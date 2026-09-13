#!/usr/bin/env python3
"""Measure same-marginal Gaussian dependence in sealed Full-22 covariances."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

PROTOCOL_SCHEMA: Final = "bayesian-phystwin/full22-dependence-compression-diagnostic-v1"
PREDICTION_MANIFEST_CONTRACT: Final = (
    "bayesian-phystwin-full22-discrepancy-prediction-manifest"
)
RESULT_SCHEMA: Final = (
    "bayesian-phystwin/full22-dependence-compression-diagnostic-result-v1"
)
EXPECTED_ARRAYS: Final = {
    "validation_mean_m",
    "validation_covariance_m2",
    "future_mean_m",
    "future_covariance_m2",
    "prediction_success",
    "fit_end",
    "train_end",
    "frame_count",
}
FloatArray: TypeAlias = NDArray[np.float64]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    """Load the content-addressed, target-free diagnostic protocol."""

    protocol = _load_json(path)
    identity = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if _require_sha256(protocol.get("protocol_id"), name="protocol_id") != (
        _canonical_sha256(identity)
    ):
        raise ValueError("protocol_id does not match canonical protocol content")
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_version") != 1
        or protocol.get("status") != "frozen-before-covariance-value-read"
    ):
        raise ValueError("unexpected diagnostic protocol identity or status")
    source = protocol.get("source")
    analysis = protocol.get("analysis")
    boundary = protocol.get("information_boundary")
    gate = protocol.get("proceed_gate")
    if not all(isinstance(value, dict) for value in (source, analysis, boundary, gate)):
        raise ValueError("protocol sections must be JSON objects")
    assert isinstance(source, dict)
    assert isinstance(analysis, dict)
    assert isinstance(boundary, dict)
    assert isinstance(gate, dict)
    if source.get("candidate_id") != "independent_endpoint_v1":
        raise ValueError("candidate donor changed")
    if source.get("case_count") != 22:
        raise ValueError("case count changed")
    if analysis.get("dense_cross_track_covariance_available") is not False:
        raise ValueError("exported covariance must remain block-only")
    if analysis.get("factor_scope") != "per-track-per-frame only":
        raise ValueError("factor scope changed")
    if boundary.get("future_outcomes_required") is not False:
        raise ValueError("diagnostic must remain target-free")
    if boundary.get("covariance_values_read_before_freeze") is not False:
        raise ValueError("protocol is not prospectively frozen for covariance values")
    if gate.get("outcome_comparison_authorized_by_this_diagnostic") is not False:
        raise ValueError("diagnostic cannot authorize outcome access")
    return protocol


def load_prediction_manifest(
    prediction_dir: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the exact frozen donor manifest and all non-value bindings."""

    source = protocol["source"]
    assert isinstance(source, Mapping)
    path = prediction_dir / "prediction_manifest.json"
    expected_file_sha = _require_sha256(
        source.get("prediction_manifest_file_sha256"),
        name="prediction_manifest_file_sha256",
    )
    if _file_sha256(path) != expected_file_sha:
        raise ValueError("prediction manifest file digest changed")
    manifest = _load_json(path)
    descriptor = {
        key: value
        for key, value in manifest.items()
        if key != "prediction_artifact_sha256"
    }
    if manifest.get("prediction_artifact_sha256") != _canonical_sha256(descriptor):
        raise ValueError("prediction manifest content identity changed")
    expected = {
        "candidate_id": source.get("candidate_id"),
        "prediction_artifact_sha256": source.get("prediction_artifact_sha256"),
        "prefix_manifest_id": source.get("prefix_manifest_id"),
        "protocol_id": source.get("source_protocol_id"),
        "source_revision": source.get("prediction_source_revision"),
        "case_count": source.get("case_count"),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"prediction manifest {key} changed")
    if (
        manifest.get("contract") != PREDICTION_MANIFEST_CONTRACT
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("unexpected prediction manifest contract")
    records = manifest.get("case_records")
    if not isinstance(records, list) or len(records) != source.get("case_count"):
        raise ValueError("prediction case roster changed")
    case_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("prediction_success") is not True:
            raise ValueError("prediction roster contains a failed or invalid record")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("prediction case identifiers must be unique strings")
        case_ids.add(case_id)
        _require_sha256(record.get("sha256"), name=f"{case_id}.sha256")
    return manifest


def total_correlation_nats(covariance: FloatArray) -> FloatArray:
    """Return KL(N(0,Sigma) || N(0,diag(Sigma))) for each matrix."""

    value = np.asarray(covariance, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError("covariance must end in a square matrix")
    dimension = value.shape[-1]
    if dimension < 2 or not np.all(np.isfinite(value)):
        raise ValueError("covariance must be finite with dimension at least two")
    symmetric = 0.5 * (value + np.swapaxes(value, -1, -2))
    diagonal = np.diagonal(symmetric, axis1=-2, axis2=-1)
    if np.any(diagonal <= 0.0):
        raise ValueError("covariance diagonal must be strictly positive")
    denominator = np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    correlation = symmetric / denominator
    sign, log_determinant = np.linalg.slogdet(correlation)
    if np.any(sign <= 0.0) or not np.all(np.isfinite(log_determinant)):
        raise ValueError("correlation matrix must be positive definite")
    result = -0.5 * log_determinant
    if float(np.min(result, initial=0.0)) < -1e-10:
        raise ValueError("total correlation became materially negative")
    return cast(FloatArray, np.asarray(np.maximum(result, 0.0), dtype=np.float64))


def _rank1_marginal_reconstruction(covariance: FloatArray) -> FloatArray:
    """Return a marginal-preserving isotropic-plus-leading-factor diagnostic."""

    value = np.asarray(covariance, dtype=np.float64)
    if value.shape[-2:] != (3, 3):
        raise ValueError("rank-one diagnostic requires 3x3 covariance blocks")
    symmetric = 0.5 * (value + np.swapaxes(value, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues, initial=0.0)) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    strength = np.maximum(eigenvalues[..., -1] - eigenvalues[..., 0], 0.0)
    direction = eigenvectors[..., :, -1]
    factor = strength[..., None, None] * (
        direction[..., :, None] * direction[..., None, :]
    )
    residual_diagonal = np.diagonal(symmetric, axis1=-2, axis2=-1) - np.diagonal(
        factor,
        axis1=-2,
        axis2=-1,
    )
    if float(np.min(residual_diagonal, initial=0.0)) < -1e-10:
        raise ValueError("rank-one residual diagonal became negative")
    result = factor.copy()
    index = np.arange(3)
    result[..., index, index] += np.maximum(residual_diagonal, 0.0)
    return cast(FloatArray, np.asarray(result, dtype=np.float64))


def _matrix_diagnostics(covariance: FloatArray) -> dict[str, FloatArray]:
    value = np.asarray(covariance, dtype=np.float64)
    total = total_correlation_nats(value)
    rank1 = _rank1_marginal_reconstruction(value)
    rank1_total = total_correlation_nats(rank1)
    diagonal = np.diagonal(value, axis1=-2, axis2=-1)
    denominator = np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    correlation = value / denominator
    off_diagonal = correlation - np.eye(3, dtype=np.float64)
    off_diagonal_rms = np.sqrt(np.sum(off_diagonal**2, axis=(-2, -1)) / 6.0)
    eigenvalues = np.linalg.eigvalsh(0.5 * (value + np.swapaxes(value, -1, -2)))
    anisotropic = np.maximum(eigenvalues - eigenvalues[..., :1], 0.0)
    anisotropic_total = np.sum(anisotropic, axis=-1)
    rank1_fraction = np.divide(
        anisotropic[..., -1],
        anisotropic_total,
        out=np.ones_like(anisotropic_total),
        where=anisotropic_total > 0.0,
    )
    return {
        "total_correlation_nats": total,
        "rank1_total_correlation_nats": rank1_total,
        "rank1_absolute_total_correlation_error_nats": np.abs(rank1_total - total),
        "off_diagonal_correlation_rms": off_diagonal_rms,
        "rank1_anisotropic_trace_fraction": rank1_fraction,
    }


def _scaled_covariance(
    covariance: FloatArray,
    *,
    scales: Sequence[float],
    observation_std_m: float,
) -> FloatArray:
    value = np.asarray(covariance, dtype=np.float64)
    if value.ndim != 4 or value.shape[-2:] != (3, 3):
        raise ValueError("future covariance must have shape (T,N,3,3)")
    if len(scales) != 3 or any(not math.isfinite(x) or x <= 0.0 for x in scales):
        raise ValueError("three finite positive horizon scales are required")
    schedule = np.empty(value.shape[0], dtype=np.float64)
    for scale, indices in zip(
        scales,
        np.array_split(np.arange(len(value)), 3),
        strict=True,
    ):
        schedule[indices] = float(scale)
    result = value * schedule[:, None, None, None]
    result = result.copy()
    index = np.arange(3)
    result[..., index, index] += float(observation_std_m) ** 2
    return cast(FloatArray, np.asarray(result, dtype=np.float64))


def analyze_case_covariance(
    future_covariance_m2: FloatArray,
    *,
    scales: Sequence[float],
    observation_std_m: float,
) -> dict[str, float | int]:
    """Summarize one case without loading any prediction mean or outcome."""

    raw = _scaled_covariance(
        future_covariance_m2,
        scales=(1.0, 1.0, 1.0),
        observation_std_m=observation_std_m,
    )
    primary = _scaled_covariance(
        future_covariance_m2,
        scales=scales,
        observation_std_m=observation_std_m,
    )
    raw_diagnostics = _matrix_diagnostics(raw)
    diagnostics = _matrix_diagnostics(primary)
    total = diagnostics["total_correlation_nats"]
    rank1_error = diagnostics["rank1_absolute_total_correlation_error_nats"]
    mean_total = float(np.mean(total))
    relative_error = float(np.mean(rank1_error) / max(mean_total, 1e-15))
    return {
        "block_count": int(total.size),
        "future_frame_count": int(primary.shape[0]),
        "track_count": int(primary.shape[1]),
        "mean_raw_total_correlation_nats": float(
            np.mean(raw_diagnostics["total_correlation_nats"])
        ),
        "mean_total_correlation_nats": mean_total,
        "median_total_correlation_nats": float(np.median(total)),
        "maximum_total_correlation_nats": float(np.max(total)),
        "mean_off_diagonal_correlation_rms": float(
            np.mean(diagnostics["off_diagonal_correlation_rms"])
        ),
        "mean_rank1_anisotropic_trace_fraction": float(
            np.mean(diagnostics["rank1_anisotropic_trace_fraction"])
        ),
        "mean_rank1_total_correlation_nats": float(
            np.mean(diagnostics["rank1_total_correlation_nats"])
        ),
        "rank1_relative_total_correlation_error": relative_error,
    }


def _bootstrap_interval(
    values: FloatArray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> list[float]:
    if values.ndim != 1 or len(values) < 2 or replicates < 1000:
        raise ValueError("bootstrap requires at least two cases and 1000 replicates")
    rng = np.random.default_rng(seed)
    means: FloatArray = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 5000):
        stop = min(start + 5000, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = np.mean(values[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    quantiles = np.asarray(
        np.quantile(means, [tail, 1.0 - tail]),
        dtype=np.float64,
    )
    return [float(quantiles[0]), float(quantiles[1])]


def _load_case(path: Path, expected_sha256: str) -> FloatArray:
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"prediction case digest changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARRAYS:
            raise ValueError(f"prediction case array roster changed: {path}")
        if not bool(np.asarray(archive["prediction_success"]).item()):
            raise ValueError(f"prediction case was not successful: {path}")
        fit_end = int(np.asarray(archive["fit_end"]).item())
        train_end = int(np.asarray(archive["train_end"]).item())
        frame_count = int(np.asarray(archive["frame_count"]).item())
        covariance = np.asarray(archive["future_covariance_m2"], dtype=np.float64)
    if not 0 < fit_end < train_end < frame_count:
        raise ValueError(f"prediction split is invalid: {path}")
    if covariance.shape[0] != frame_count - train_end:
        raise ValueError(f"future covariance length differs from split: {path}")
    return cast(FloatArray, np.asarray(covariance, dtype=np.float64))


def analyze(
    prediction_dir: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    manifest = load_prediction_manifest(prediction_dir, protocol)
    analysis = protocol["analysis"]
    gate = protocol["proceed_gate"]
    assert isinstance(analysis, dict)
    assert isinstance(gate, dict)
    scales = tuple(float(value) for value in analysis["frozen_horizon_scales"])
    observation_std_m = float(analysis["observation_std_m"])
    cases: list[dict[str, Any]] = []
    for record in manifest["case_records"]:
        case_id = str(record["case_id"])
        covariance = _load_case(
            prediction_dir / str(record["path"]),
            str(record["sha256"]),
        )
        row = analyze_case_covariance(
            covariance,
            scales=scales,
            observation_std_m=observation_std_m,
        )
        cases.append({"case_id": case_id, **row})
    case_total = np.asarray(
        [row["mean_total_correlation_nats"] for row in cases],
        dtype=np.float64,
    )
    case_raw = np.asarray(
        [row["mean_raw_total_correlation_nats"] for row in cases],
        dtype=np.float64,
    )
    rank1_error = np.asarray(
        [row["rank1_relative_total_correlation_error"] for row in cases],
        dtype=np.float64,
    )
    threshold_fraction = float(np.mean(case_total >= 0.01))
    mean_total = float(np.mean(case_total))
    median_rank1_error = float(np.median(rank1_error))
    dependence_signal_supported = bool(
        mean_total >= float(gate["minimum_equal_case_mean_total_correlation_nats"])
        and threshold_fraction
        >= float(
            gate["minimum_case_fraction_with_mean_total_correlation_at_least_0_01_nats"]
        )
    )
    local_rank1_supported = bool(
        median_rank1_error
        <= float(gate["maximum_median_case_rank1_relative_total_correlation_error"])
    )
    dense_available = bool(analysis["dense_cross_track_covariance_available"])
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "completed-target-free",
        "protocol_id": protocol["protocol_id"],
        "source": {
            "artifact_id": protocol["source"]["artifact_id"],
            "artifact_sha256": protocol["source"]["artifact_sha256"],
            "candidate_id": protocol["source"]["candidate_id"],
            "prediction_artifact_sha256": manifest["prediction_artifact_sha256"],
            "prediction_manifest_file_sha256": _file_sha256(
                prediction_dir / "prediction_manifest.json"
            ),
        },
        "representation": {
            "covariance_block": analysis["covariance_block"],
            "factor_scope": analysis["factor_scope"],
            "dense_cross_track_covariance_available": dense_available,
            "full_symmetric_parameters_per_block": 6,
            "diagonal_plus_rank1_parameters_per_block": 6,
            "strict_parameter_compression_per_block": False,
        },
        "aggregate": {
            "case_count": len(cases),
            "block_count": int(sum(int(row["block_count"]) for row in cases)),
            "equal_case_mean_raw_total_correlation_nats": float(np.mean(case_raw)),
            "equal_case_mean_total_correlation_nats": mean_total,
            "equal_case_mean_total_correlation_bits": mean_total / math.log(2.0),
            "case_bootstrap_95_interval_nats": _bootstrap_interval(
                case_total,
                replicates=int(analysis["bootstrap_replicates"]),
                seed=int(analysis["bootstrap_seed"]),
                confidence=float(analysis["bootstrap_confidence"]),
            ),
            "case_median_mean_total_correlation_nats": float(np.median(case_total)),
            "case_minimum_mean_total_correlation_nats": float(np.min(case_total)),
            "case_maximum_mean_total_correlation_nats": float(np.max(case_total)),
            "case_fraction_with_mean_total_correlation_at_least_0_01_nats": (
                threshold_fraction
            ),
            "median_case_rank1_relative_total_correlation_error": (median_rank1_error),
            "mean_case_rank1_anisotropic_trace_fraction": float(
                np.mean([row["mean_rank1_anisotropic_trace_fraction"] for row in cases])
            ),
        },
        "gates": {
            "dependence_signal_supported": dependence_signal_supported,
            "local_rank1_fidelity_supported": local_rank1_supported,
            "whole_object_dependence_testable": dense_available,
            "strict_compression_supported": False,
            "headline_fused_claim_supported": bool(
                dependence_signal_supported
                and local_rank1_supported
                and dense_available
            ),
            "realized_outcome_comparison_authorized": False,
        },
        "cases": cases,
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    report["result_id"] = _canonical_sha256(report)
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    gates = report["gates"]
    representation = report["representation"]
    assert isinstance(aggregate, Mapping)
    assert isinstance(gates, Mapping)
    assert isinstance(representation, Mapping)
    return "\n".join(
        (
            "# Full-22 dependence/compression diagnostic v1",
            "",
            "This diagnostic reads sealed covariance predictions only; it reads no future",
            "truth, score, prefix array, protected target, or held-v8 artifact.",
            "",
            "## Result",
            "",
            f"- Cases: `{aggregate['case_count']}`",
            f"- 3D covariance blocks: `{aggregate['block_count']}`",
            "- Equal-case mean total correlation: "
            f"`{float(aggregate['equal_case_mean_total_correlation_nats']):.9g}` nats",
            "- 95% case-bootstrap interval: "
            f"`{aggregate['case_bootstrap_95_interval_nats']}`",
            "- Median case rank-one relative total-correlation error: "
            f"`{float(aggregate['median_case_rank1_relative_total_correlation_error']):.6g}`",
            "",
            "## Gate",
            "",
            f"- Dependence signal supported: `{str(gates['dependence_signal_supported']).lower()}`",
            f"- Local rank-one fidelity supported: `{str(gates['local_rank1_fidelity_supported']).lower()}`",
            f"- Whole-object dependence testable: `{str(gates['whole_object_dependence_testable']).lower()}`",
            f"- Fused headline supported: `{str(gates['headline_fused_claim_supported']).lower()}`",
            "",
            "## Representation boundary",
            "",
            "The archive contains independent per-point 3x3 covariance blocks, not one",
            "dense covariance over a physical object. A symmetric 3x3 covariance needs six",
            "parameters, and a free diagonal plus one 3-vector factor also needs six; local",
            "rank-one fidelity therefore is not strict parameter compression by itself.",
            "",
            str(report["claim_boundary"]),
            "",
        )
    )


def write_result(output_dir: Path, report: Mapping[str, Any]) -> None:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        result_path = temporary / "result.json"
        report_path = temporary / "report.md"
        result_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(_markdown(report), encoding="utf-8")
        checksums = {
            path.name: _file_sha256(path) for path in (result_path, report_path)
        }
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
            ),
            encoding="ascii",
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = analyze(arguments.prediction_dir, arguments.protocol)
    write_result(arguments.output, report)
    print(json.dumps(report["aggregate"], sort_keys=True))
    print(json.dumps(report["gates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
