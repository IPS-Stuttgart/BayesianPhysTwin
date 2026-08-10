#!/usr/bin/env python3
"""Run the prediction-first released-PhysTwin full-22 discrepancy tournament.

The released 22-case cohort is already open development evidence. This runner
uses it only as a source-only candidate-selection panel. It never authorizes a
claim, never opens any still-sealed confirmation payload, and never lets a
candidate predictor receive the scored future arrays.

The information order is explicit:

1. ``prepare-prefix`` serializes only causal fit and validation prefixes.
2. ``predict`` runs one exact candidate revision and seals raw forecasts.
3. ``admit`` applies the paper's metric-specific baseline-relative guard using
   only the validation prefix.
4. ``score`` opens the already-public future, applies exact fallback, runs one
   tournament per primary metric, and requires metric agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_CONTRACT = "bayesian-phystwin-full22-discrepancy-candidate-tournament"
PREFIX_MANIFEST_CONTRACT = "bayesian-phystwin-full22-discrepancy-prefix-manifest"
PREDICTION_MANIFEST_CONTRACT = (
    "bayesian-phystwin-full22-discrepancy-prediction-manifest"
)
ADMISSION_MANIFEST_CONTRACT = "bayesian-phystwin-full22-discrepancy-admission-manifest"
ARBITRATION_REPORT_CONTRACT = "bayesian-phystwin-full22-discrepancy-metric-arbitration"
HORIZON_LABELS = ("early", "middle", "late")
PRIMARY_METRICS = ("chamfer_distance_m", "track_error_m")
STATISTICAL_UNIT = "physical-object-session"
LOWER_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _literal_sha(value: object, *, name: str, length: int) -> str:
    _require(type(value) is str, f"{name} must be a literal string")
    text = str(value)
    _require(
        len(text) == length and set(text) <= LOWER_HEX,
        f"{name} must be {length} lowercase hexadecimal characters",
    )
    return text


def _canonical_string(value: object, *, name: str) -> str:
    _require(
        type(value) is str and bool(value) and value.strip() == value,
        f"{name} must be a nonempty canonical string",
    )
    return str(value)


@contextmanager
def _atomic_output_directory(
    target: Path,
    *,
    force: bool,
) -> Iterator[Path]:
    target = target.resolve()
    if target.exists():
        if not force:
            raise FileExistsError(f"output already exists: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        yield temporary
        if target.exists():
            raise FileExistsError(f"output appeared during publication: {target}")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict),
        "protocol root must be a JSON object",
    )
    _require(
        payload.get("contract") == PROTOCOL_CONTRACT,
        "unexpected full-22 tournament protocol contract",
    )
    _require(
        payload.get("schema_version") == 1,
        "unsupported full-22 tournament protocol version",
    )
    _require(
        payload.get("status") == "retrospective-source-only-non-claim-bearing",
        "protocol must retain the retrospective source-only boundary",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("prediction_manifests_sealed_before_admission") is True
        and boundary.get("admission_manifest_sealed_before_future_scoring") is True
        and boundary.get("confirmation_payload_used") is False
        and boundary.get("target_outcome_used") is False
        and boundary.get("replacement_allowed") is False
        and boundary.get("claim_authorized") is False,
        "protocol information boundary changed",
    )
    candidates = payload.get("candidates")
    _require(
        isinstance(candidates, list) and len(candidates) >= 3,
        "protocol candidates must be a nonempty list",
    )
    candidate_ids: list[str] = []
    for index, raw in enumerate(candidates):
        _require(
            isinstance(raw, dict),
            f"candidates[{index}] must be a JSON object",
        )
        candidate_id = _canonical_string(
            raw.get("candidate_id"),
            name=f"candidates[{index}].candidate_id",
        )
        _canonical_string(
            raw.get("family"),
            name=f"candidates[{index}].family",
        )
        _canonical_string(
            raw.get("runner"),
            name=f"candidates[{index}].runner",
        )
        _literal_sha(
            raw.get("source_revision"),
            name=f"candidates[{index}].source_revision",
            length=40,
        )
        _require(
            type(raw.get("declared_parameter_count")) is int
            and int(raw["declared_parameter_count"]) >= 0,
            f"candidates[{index}].declared_parameter_count is invalid",
        )
        _require(
            isinstance(raw.get("configuration"), dict),
            f"candidates[{index}].configuration must be a JSON object",
        )
        candidate_ids.append(candidate_id)
    _require(
        len(set(candidate_ids)) == len(candidate_ids),
        "protocol candidate identifiers must be unique",
    )
    selection = payload.get("selection")
    _require(
        isinstance(selection, dict)
        and selection.get("reference_candidate") in candidate_ids
        and selection.get("physical_fallback_candidate") in candidate_ids,
        "protocol baseline candidates changed",
    )
    scoring = payload.get("scoring")
    _require(
        isinstance(scoring, dict)
        and tuple(scoring.get("horizon_bins", ())) == HORIZON_LABELS,
        "protocol horizon bins changed",
    )
    point_tournaments = scoring.get("point_tournaments")
    _require(
        isinstance(point_tournaments, list)
        and tuple(row.get("name") for row in point_tournaments) == ("track", "chamfer"),
        "protocol metric-specific tournament roster changed",
    )
    return payload, _canonical_sha256(payload)


def _candidate_spec(
    protocol: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    for raw in protocol["candidates"]:
        if raw["candidate_id"] == candidate_id:
            return dict(raw)
    raise KeyError(candidate_id)


def _candidate_ids(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(raw["candidate_id"]) for raw in protocol["candidates"])


def _configuration_sha256(candidate: Mapping[str, Any]) -> str:
    return _canonical_sha256(candidate["configuration"])


def _download_trajectory_subset(data_root: Path) -> dict[str, Any]:
    """Retrieve only the compact released trajectory-evaluation files."""

    from bayesian_phystwin.phystwin_data import (
        DEFAULT_DATA_ARCHIVE,
        DEFAULT_EXPERIMENTS_ARCHIVE,
        EVALUATION_FILENAMES,
        _archive_factory,
        _available_cases,
        _retrieve_member,
    )

    records: dict[str, dict[str, object]] = {}
    with ExitStack() as stack:
        data_archive = stack.enter_context(_archive_factory(DEFAULT_DATA_ARCHIVE))
        experiments_archive = stack.enter_context(
            _archive_factory(DEFAULT_EXPERIMENTS_ARCHIVE)
        )
        available = _available_cases(
            data_archive,
            experiments_archive,
        )
        if len(available) != 22:
            raise ValueError(
                "the released tournament requires exactly 22 complete cases; "
                f"found {len(available)}"
            )
        for case in available:
            case_dir = data_root / case
            files: dict[str, object] = {}
            for filename in EVALUATION_FILENAMES:
                member = f"data/different_types/{case}/{filename}"
                files[filename] = _retrieve_member(
                    data_archive,
                    member,
                    case_dir / filename,
                )
            member = f"experiments/{case}/inference.pkl"
            files["inference.pkl"] = _retrieve_member(
                experiments_archive,
                member,
                case_dir / "inference.pkl",
            )
            records[case] = {"files": files}

    manifest = {
        "schema": "bayesian-phystwin-trajectory-evaluation-subset",
        "schema_version": 1,
        "sources": {
            "data": DEFAULT_DATA_ARCHIVE,
            "experiments": DEFAULT_EXPERIMENTS_ARCHIVE,
        },
        "selected_cases": list(available),
        "cases": records,
    }
    manifest_path = data_root / "trajectory_evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _file_sha256(manifest_path)
    return manifest


def _last_valid_residual(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> np.ndarray:
    residual = np.asarray(residual_m, dtype=np.float64)
    validity = np.asarray(valid)
    _require(
        residual.ndim == 3 and residual.shape[2] == 3,
        "residual_m must have shape (T, N, 3)",
    )
    _require(
        validity.dtype.kind == "b" and validity.shape == residual.shape[:2],
        "valid must be a matching Boolean matrix",
    )
    _require(
        0 < end_frame <= len(residual),
        "end_frame lies outside residual_m",
    )
    result = np.zeros((residual.shape[1], 3), dtype=np.float64)
    for track in range(residual.shape[1]):
        support = np.flatnonzero(validity[:end_frame, track])
        if len(support):
            result[track] = residual[support[-1], track]
    return result


def _horizon_groups(frame_count: int) -> dict[str, np.ndarray]:
    if frame_count < 1:
        raise ValueError("future interval must contain at least one frame")
    chunks = np.array_split(
        np.arange(frame_count, dtype=np.int64),
        len(HORIZON_LABELS),
    )
    return {
        label: chunk
        for label, chunk in zip(HORIZON_LABELS, chunks, strict=True)
        if len(chunk)
    }


def _geometry_kernel_basis(
    geometry_m: np.ndarray,
    *,
    rank: int,
    diagonal_tie_break: float,
) -> np.ndarray:
    """Build a deterministic compact frame-zero geometry-kernel basis."""

    geometry = np.asarray(geometry_m, dtype=np.float64)
    _require(
        geometry.ndim == 2 and geometry.shape[1] == 3 and len(geometry) >= rank >= 1,
        "geometry_m and rank are incompatible",
    )
    _require(
        np.all(np.isfinite(geometry)),
        "geometry_m must be finite",
    )
    difference = geometry[:, None, :] - geometry[None, :, :]
    squared = np.sum(np.square(difference), axis=2)
    nonzero = np.sqrt(squared[squared > 0.0])
    scale = float(np.median(nonzero)) if len(nonzero) else 1.0
    scale = max(scale, np.finfo(np.float64).eps)
    kernel = np.exp(-squared / (2.0 * scale**2))
    tie = diagonal_tie_break * (
        np.arange(1, len(geometry) + 1, dtype=np.float64) / len(geometry)
    )
    kernel = 0.5 * (kernel + kernel.T) + np.diag(tie)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(eigenvalues, kind="stable")[::-1][:rank]
    basis = eigenvectors[:, order].copy()
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    _require(
        np.allclose(
            basis.T @ basis,
            np.eye(rank),
            atol=1e-10,
            rtol=1e-10,
        ),
        "constructed geometry basis is not orthonormal",
    )
    return basis


def _repeat_endpoint(
    mean_m: np.ndarray,
    covariance_m2: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    return (
        np.repeat(mean[None], count, axis=0),
        np.repeat(covariance[None], count, axis=0),
    )


def _forecast_independent_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    cutoff: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from bayesian_phystwin.endpoint_model_average import (
        infer_model_averaged_endpoint,
        predict_model_averaged_endpoint,
    )

    posterior = infer_model_averaged_endpoint(
        residual_m,
        valid,
        end_frame=cutoff,
    )
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for horizon in range(1, count + 1):
        prediction = predict_model_averaged_endpoint(
            posterior,
            horizon_steps=horizon,
        )
        means.append(np.asarray(prediction.mean_m, dtype=np.float64))
        covariances.append(np.asarray(prediction.covariance_m2, dtype=np.float64))
    return (
        np.stack(means),
        np.stack(covariances),
        {
            "state_dimension": int(np.asarray(posterior.component_mean_m).size),
            "covariance_bytes": int(np.asarray(posterior.component_variance_m2).nbytes),
        },
    )


def _forecast_dynamic_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    cutoff: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from bayesian_phystwin.dynamic_endpoint_model_average import (
        DynamicEndpointModelAverageConfigV2,
        infer_dynamic_endpoint_model_average,
        predict_dynamic_endpoint_model_average,
    )

    config = DynamicEndpointModelAverageConfigV2(
        evidence_pooling="object",
    )
    posterior = infer_dynamic_endpoint_model_average(
        residual_m,
        valid,
        end_frame=cutoff,
        config=config,
    )
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for horizon in range(1, count + 1):
        prediction = predict_dynamic_endpoint_model_average(
            posterior,
            horizon_steps=horizon,
        )
        means.append(np.asarray(prediction.mean_m, dtype=np.float64))
        covariances.append(np.asarray(prediction.covariance_m2, dtype=np.float64))
    return (
        np.stack(means),
        np.stack(covariances),
        {
            "state_dimension": int(np.asarray(posterior.component_state_mean).size),
            "covariance_bytes": int(
                np.asarray(posterior.component_state_covariance).nbytes
            ),
        },
    )


def _forecast_structured_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    basis: np.ndarray,
    *,
    cutoff: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from bayesian_phystwin.structured_discrepancy import (
        infer_structured_discrepancy,
        predict_structured_discrepancy,
    )

    posterior = infer_structured_discrepancy(
        residual_m,
        valid,
        basis,
        end_frame=cutoff,
    )
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for horizon in range(1, count + 1):
        prediction = predict_structured_discrepancy(
            posterior,
            horizon_steps=horizon,
        )
        means.append(np.asarray(prediction.mean_m, dtype=np.float64))
        covariances.append(
            np.asarray(
                prediction.marginal_covariance_m2,
                dtype=np.float64,
            )
        )
    return (
        np.stack(means),
        np.stack(covariances),
        {
            "state_dimension": int(
                np.asarray(posterior.component_coefficient_mean_m).size
                + np.asarray(posterior.component_local_variance_m2).size
            ),
            "covariance_bytes": int(
                np.asarray(posterior.component_coefficient_covariance_m2).nbytes
                + np.asarray(posterior.component_local_variance_m2).nbytes
            ),
        },
    )


def _forecast_graph_dynamic(
    residual_m: np.ndarray,
    valid: np.ndarray,
    basis: np.ndarray,
    configuration: Mapping[str, Any],
    *,
    cutoff: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from bayesian_phystwin.graph_dynamic_discrepancy import (
        GraphDynamicDiscrepancyConfigV1,
        fit_graph_dynamic_discrepancy,
    )

    field_names = (
        "initial_position_std_m",
        "initial_velocity_std_mps",
        "process_position_std_m",
        "process_acceleration_std_mps2",
        "velocity_retention",
        "observation_std_m",
        "degrees_of_freedom",
        "effective_samples_per_correlation_group",
        "minimum_robust_weight",
        "maximum_iterations",
        "convergence_tolerance",
        "maximum_condition_number",
        "maximum_node_position_m",
        "maximum_node_velocity_mps",
    )
    config = GraphDynamicDiscrepancyConfigV1(
        **{name: configuration[name] for name in field_names}
    )
    belief = fit_graph_dynamic_discrepancy(
        residual_m[:cutoff],
        valid[:cutoff],
        basis,
        frame_dt_s=float(configuration["frame_dt_s"]),
        config=config,
    )
    horizons = np.arange(1, count + 1, dtype=np.int64)
    forecast = belief.forecast(horizons)
    return (
        np.asarray(forecast.mean_m, dtype=np.float64),
        np.asarray(
            forecast.marginal_covariance_m2,
            dtype=np.float64,
        ),
        {
            "state_dimension": int(np.asarray(belief.state_mean).size),
            "covariance_bytes": int(np.asarray(belief.state_covariance).nbytes),
            "accepted_update_count": int(belief.accepted_update_count),
        },
    )


def _forecast_one_candidate(
    candidate: Mapping[str, Any],
    residual_m: np.ndarray,
    valid: np.ndarray,
    geometry_m: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    cutoff: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    runner = str(candidate["runner"])
    track_count = residual_m.shape[1]
    if runner == "physical_fallback":
        return (
            np.zeros((count, track_count, 3), dtype=np.float64),
            np.zeros(
                (count, track_count, 3, 3),
                dtype=np.float64,
            ),
            {"state_dimension": 0, "covariance_bytes": 0},
        )
    if runner == "last_residual":
        mean = _last_valid_residual(
            residual_m,
            valid,
            end_frame=cutoff,
        )
        covariance = np.zeros((track_count, 3, 3), dtype=np.float64)
        repeated = _repeat_endpoint(mean, covariance, count)
        return (
            repeated[0],
            repeated[1],
            {
                "state_dimension": int(mean.size),
                "covariance_bytes": 0,
            },
        )
    if runner == "independent_endpoint_v1":
        return _forecast_independent_endpoint(
            residual_m,
            valid,
            cutoff=cutoff,
            count=count,
        )
    if runner == "dynamic_endpoint_v2":
        return _forecast_dynamic_endpoint(
            residual_m,
            valid,
            cutoff=cutoff,
            count=count,
        )
    basis_spec = protocol["basis"]
    basis = _geometry_kernel_basis(
        geometry_m,
        rank=int(basis_spec["rank"]),
        diagonal_tie_break=float(basis_spec["diagonal_tie_break"]),
    )
    if runner == "structured_kernel_rank4_v1":
        return _forecast_structured_endpoint(
            residual_m,
            valid,
            basis,
            cutoff=cutoff,
            count=count,
        )
    if runner == "graph_dynamic_kernel_rank4_v1":
        return _forecast_graph_dynamic(
            residual_m,
            valid,
            basis,
            candidate["configuration"],
            cutoff=cutoff,
            count=count,
        )
    raise ValueError(f"unsupported candidate runner: {runner}")


def _safe_forecast_one_candidate(
    candidate: Mapping[str, Any],
    residual_m: np.ndarray,
    valid: np.ndarray,
    geometry_m: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    cutoff: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], bool]:
    try:
        mean, covariance, diagnostics = _forecast_one_candidate(
            candidate,
            residual_m,
            valid,
            geometry_m,
            protocol,
            cutoff=cutoff,
            count=count,
        )
        _require(
            mean.shape == (count, residual_m.shape[1], 3),
            "candidate mean shape changed",
        )
        _require(
            covariance.shape == (count, residual_m.shape[1], 3, 3),
            "candidate covariance shape changed",
        )
        _require(
            np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance)),
            "candidate forecast is nonfinite",
        )
        return mean, covariance, diagnostics, True
    except Exception as error:
        track_count = residual_m.shape[1]
        return (
            np.zeros((count, track_count, 3), dtype=np.float64),
            np.zeros(
                (count, track_count, 3, 3),
                dtype=np.float64,
            ),
            {
                "state_dimension": 0,
                "covariance_bytes": 0,
                "technical_failure_type": type(error).__name__,
                "technical_failure_message": str(error),
            },
            False,
        )


def _load_prefix_manifest(prefix_dir: Path) -> dict[str, Any]:
    path = prefix_dir / "prefix_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("contract") == PREFIX_MANIFEST_CONTRACT
        and payload.get("schema_version") == 1,
        "invalid prefix manifest",
    )
    supplied = payload.get("prefix_manifest_id")
    descriptor = {
        key: value for key, value in payload.items() if key != "prefix_manifest_id"
    }
    _require(
        supplied == _canonical_sha256(descriptor),
        "prefix manifest identity changed",
    )
    return payload


def _load_prediction_manifest(
    prediction_dir: Path,
) -> dict[str, Any]:
    path = prediction_dir / "prediction_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("contract") == PREDICTION_MANIFEST_CONTRACT
        and payload.get("schema_version") == 1,
        "invalid prediction manifest",
    )
    supplied = payload.get("prediction_artifact_sha256")
    descriptor = {
        key: value
        for key, value in payload.items()
        if key != "prediction_artifact_sha256"
    }
    _require(
        supplied == _canonical_sha256(descriptor),
        "prediction manifest identity changed",
    )
    return payload


def _load_admission_manifest(
    admission_dir: Path,
) -> dict[str, Any]:
    path = admission_dir / "admission_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("contract") == ADMISSION_MANIFEST_CONTRACT
        and payload.get("schema_version") == 1,
        "invalid admission manifest",
    )
    supplied = payload.get("admission_manifest_id")
    descriptor = {
        key: value for key, value in payload.items() if key != "admission_manifest_id"
    }
    _require(
        supplied == _canonical_sha256(descriptor),
        "admission manifest identity changed",
    )
    return payload


def _load_case_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    _require(
        _file_sha256(path) == expected_sha256,
        f"case artifact digest changed: {path}",
    )
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def prepare_prefix(
    data_root: Path,
    output_dir: Path,
    *,
    protocol_path: Path,
    force: bool,
) -> dict[str, Any]:
    """Materialize only fit/validation-prefix inputs for every candidate."""

    from bayesian_phystwin.phystwin_confirmatory import _split_for_case
    from bayesian_phystwin.phystwin_residual_dynamics import (
        _lift_map,
        _load_pickle,
        _target_validity,
    )

    protocol, protocol_id = _load_protocol(protocol_path)
    data_manifest = _download_trajectory_subset(data_root)
    cases = tuple(data_manifest["selected_cases"])
    _require(
        len(cases) == int(protocol["cohort"]["expected_case_count"]),
        "downloaded case roster differs from the protocol",
    )
    case_records: list[dict[str, object]] = []
    with _atomic_output_directory(output_dir, force=force) as temporary:
        case_root = temporary / "cases"
        case_root.mkdir()
        for case in cases:
            case_dir = data_root / case
            fit_end, train_end, frame_count = _split_for_case(
                case_dir,
                float(protocol["cohort"]["fit_fraction"]),
            )
            data = _load_pickle(case_dir / "final_data.pkl")
            baseline = np.asarray(
                _load_pickle(case_dir / "inference.pkl"),
                dtype=np.float64,
            )[:frame_count]
            gt_track = np.asarray(
                _load_pickle(case_dir / "gt_track_3d.pkl"),
                dtype=np.float64,
            )[:train_end]
            observed = np.asarray(
                data["object_points"],
                dtype=np.float64,
            )[:train_end]
            visible = np.asarray(
                data["object_visibilities"],
                dtype=bool,
            )[:train_end]
            motion_valid = np.asarray(
                data["object_motions_valid"],
                dtype=bool,
            )
            frame_count_observed, original_count, _ = np.asarray(
                data["object_points"]
            ).shape
            _require(
                frame_count_observed >= frame_count,
                f"observations do not cover {case}",
            )
            _require(
                baseline.shape[0] == frame_count
                and baseline.shape[1] >= original_count,
                f"baseline does not cover {case}",
            )
            valid = _target_validity(
                np.asarray(data["object_visibilities"], dtype=bool),
                motion_valid,
            )[:train_end]
            residual = observed - baseline[:train_end, :original_count]
            lift_indices, lift_weights = _lift_map(
                baseline[0],
                original_count,
                4,
            )
            num_surface_points = original_count + int(
                np.asarray(data["surface_points"]).shape[0]
            )
            path = case_root / f"{case}.npz"
            np.savez_compressed(
                path,
                residual_m=residual,
                valid=valid,
                geometry_m=baseline[0, :original_count],
                baseline_prefix_m=baseline[:train_end],
                observed_prefix_m=observed,
                visible_prefix=visible,
                gt_track_prefix_m=gt_track,
                lift_indices=lift_indices,
                lift_weights=lift_weights,
                fit_end=np.asarray(fit_end, dtype=np.int64),
                train_end=np.asarray(train_end, dtype=np.int64),
                frame_count=np.asarray(frame_count, dtype=np.int64),
                original_count=np.asarray(
                    original_count,
                    dtype=np.int64,
                ),
                num_surface_points=np.asarray(
                    num_surface_points,
                    dtype=np.int64,
                ),
            )
            case_records.append(
                {
                    "case_id": case,
                    "path": f"cases/{case}.npz",
                    "sha256": _file_sha256(path),
                    "fit_end": fit_end,
                    "train_end": train_end,
                    "frame_count": frame_count,
                    "track_count": original_count,
                    "future_arrays_serialized": False,
                    "source_files_sha256": {
                        filename: _file_sha256(case_dir / filename)
                        for filename in (
                            "final_data.pkl",
                            "inference.pkl",
                            "gt_track_3d.pkl",
                            "split.json",
                        )
                    },
                }
            )
        descriptor: dict[str, object] = {
            "contract": PREFIX_MANIFEST_CONTRACT,
            "schema_version": 1,
            "protocol_id": protocol_id,
            "source_archives": data_manifest["sources"],
            "case_count": len(case_records),
            "cases": case_records,
            "information_boundary": {
                "contains_fit_prefix": True,
                "contains_guard_validation_prefix": True,
                "contains_scored_future": False,
                "candidate_prediction_receives_future": False,
                "confirmation_payload_opened": False,
                "target_outcome_opened": False,
            },
        }
        descriptor["prefix_manifest_id"] = _canonical_sha256(descriptor)
        _write_json(
            temporary / "prefix_manifest.json",
            descriptor,
        )
    return _load_prefix_manifest(output_dir)


def predict_candidate(
    prefix_dir: Path,
    output_dir: Path,
    *,
    protocol_path: Path,
    candidate_id: str,
    source_revision: str,
    force: bool,
) -> dict[str, Any]:
    """Seal one candidate's validation and future forecasts."""

    protocol, protocol_id = _load_protocol(protocol_path)
    prefix_manifest = _load_prefix_manifest(prefix_dir)
    _require(
        prefix_manifest["protocol_id"] == protocol_id,
        "prefix and protocol identities differ",
    )
    candidate = _candidate_spec(protocol, candidate_id)
    expected_revision = _literal_sha(
        candidate["source_revision"],
        name="candidate source revision",
        length=40,
    )
    supplied_revision = _literal_sha(
        source_revision,
        name="source_revision",
        length=40,
    )
    _require(
        supplied_revision == expected_revision,
        "candidate source revision differs from the protocol",
    )
    case_records: list[dict[str, object]] = []
    elapsed_total = 0.0
    maximum_state_dimension = 0
    maximum_covariance_bytes = 0
    with _atomic_output_directory(output_dir, force=force) as temporary:
        case_root = temporary / "cases"
        case_root.mkdir()
        for case_record in prefix_manifest["cases"]:
            case_id = str(case_record["case_id"])
            prefix_path = prefix_dir / str(case_record["path"])
            case = _load_case_npz(
                prefix_path,
                str(case_record["sha256"]),
            )
            residual = np.asarray(
                case["residual_m"],
                dtype=np.float64,
            )
            valid = np.asarray(case["valid"])
            geometry = np.asarray(
                case["geometry_m"],
                dtype=np.float64,
            )
            fit_end = int(case["fit_end"])
            train_end = int(case["train_end"])
            frame_count = int(case["frame_count"])
            validation_count = train_end - fit_end
            future_count = frame_count - train_end
            started = time.perf_counter()
            validation = _safe_forecast_one_candidate(
                candidate,
                residual,
                valid,
                geometry,
                protocol,
                cutoff=fit_end,
                count=validation_count,
            )
            future = _safe_forecast_one_candidate(
                candidate,
                residual,
                valid,
                geometry,
                protocol,
                cutoff=train_end,
                count=future_count,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            elapsed_total += elapsed_ms
            successful = bool(validation[3] and future[3])
            if candidate_id in {"physical_fallback", "last_residual"}:
                _require(
                    successful,
                    f"required baseline candidate failed: {candidate_id}/{case_id}",
                )
            state_dimension = max(
                int(validation[2].get("state_dimension", 0)),
                int(future[2].get("state_dimension", 0)),
            )
            covariance_bytes = max(
                int(validation[2].get("covariance_bytes", 0)),
                int(future[2].get("covariance_bytes", 0)),
            )
            maximum_state_dimension = max(
                maximum_state_dimension,
                state_dimension,
            )
            maximum_covariance_bytes = max(
                maximum_covariance_bytes,
                covariance_bytes,
            )
            path = case_root / f"{case_id}.npz"
            np.savez_compressed(
                path,
                validation_mean_m=validation[0],
                validation_covariance_m2=validation[1],
                future_mean_m=future[0],
                future_covariance_m2=future[1],
                prediction_success=np.asarray(
                    successful,
                    dtype=bool,
                ),
                fit_end=np.asarray(fit_end, dtype=np.int64),
                train_end=np.asarray(train_end, dtype=np.int64),
                frame_count=np.asarray(
                    frame_count,
                    dtype=np.int64,
                ),
            )
            case_records.append(
                {
                    "case_id": case_id,
                    "path": f"cases/{case_id}.npz",
                    "sha256": _file_sha256(path),
                    "prediction_success": successful,
                    "runtime_milliseconds": elapsed_ms,
                    "state_dimension": state_dimension,
                    "covariance_bytes": covariance_bytes,
                    "validation_diagnostics": validation[2],
                    "future_diagnostics": future[2],
                }
            )
        descriptor: dict[str, object] = {
            "contract": PREDICTION_MANIFEST_CONTRACT,
            "schema_version": 1,
            "protocol_id": protocol_id,
            "prefix_manifest_id": prefix_manifest["prefix_manifest_id"],
            "candidate_id": candidate_id,
            "family": candidate["family"],
            "source_revision": supplied_revision,
            "configuration_sha256": _configuration_sha256(candidate),
            "declared_parameter_count": int(candidate["declared_parameter_count"]),
            "case_count": len(case_records),
            "case_records": case_records,
            "runtime_milliseconds": (elapsed_total / len(case_records)),
            "state_dimension": maximum_state_dimension,
            "covariance_bytes": maximum_covariance_bytes,
            "information_boundary": {
                "input_artifact": "prefix-manifest-only",
                "scored_future_arrays_received": False,
                "future_observations_used": False,
                "confirmation_payload_opened": False,
                "target_outcome_opened": False,
                "prediction_sealed_before_admission": True,
            },
        }
        descriptor["prediction_artifact_sha256"] = _canonical_sha256(descriptor)
        _write_json(
            temporary / "prediction_manifest.json",
            descriptor,
        )
    return _load_prediction_manifest(output_dir)


def _candidate_trajectory(
    baseline: np.ndarray,
    tracked_mean_m: np.ndarray,
    *,
    start_frame: int,
    lift_indices: np.ndarray,
    lift_weights: np.ndarray,
    maximum_residual_m: float,
) -> np.ndarray:
    from bayesian_phystwin.phystwin_residual_dynamics import (
        _lift_residual,
    )

    result = np.asarray(baseline, dtype=np.float64).copy()
    tracked = np.asarray(tracked_mean_m, dtype=np.float64)
    original_count = result.shape[1] - len(lift_indices)
    _require(
        tracked.ndim == 3
        and tracked.shape == (len(result) - start_frame, original_count, 3),
        "tracked candidate correction has an invalid shape",
    )
    correction = _lift_residual(
        tracked,
        result.shape[1],
        lift_indices,
        lift_weights,
        maximum_norm=maximum_residual_m,
    )
    result[start_frame:] += correction
    return result


def _metrics_by_frame(
    trajectory: np.ndarray,
    observed: np.ndarray,
    visible: np.ndarray,
    gt_track: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, np.ndarray]:
    from bayesian_phystwin.phystwin_official_evaluation import (
        official_phystwin_metrics_by_frame,
    )

    return official_phystwin_metrics_by_frame(
        trajectory,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim == 1 and len(array) > 0 and np.all(np.isfinite(array)),
        "regret values must be a finite nonempty vector",
    )
    _require(
        0.0 <= quantile <= 1.0,
        "regret quantile must lie in [0, 1]",
    )
    return float(np.quantile(array, quantile, method="higher"))


def admit_predictions(
    prefix_dir: Path,
    predictions_root: Path,
    output_dir: Path,
    *,
    protocol_path: Path,
    force: bool,
) -> dict[str, Any]:
    """Apply the paper guard without reading the scored future."""

    protocol, protocol_id = _load_protocol(protocol_path)
    prefix_manifest = _load_prefix_manifest(prefix_dir)
    _require(
        prefix_manifest["protocol_id"] == protocol_id,
        "prefix and protocol identities differ",
    )
    prediction_manifests = {
        candidate_id: _load_prediction_manifest(predictions_root / candidate_id)
        for candidate_id in _candidate_ids(protocol)
    }
    for candidate_id, manifest in prediction_manifests.items():
        _require(
            manifest["candidate_id"] == candidate_id
            and manifest["protocol_id"] == protocol_id
            and manifest["prefix_manifest_id"] == prefix_manifest["prefix_manifest_id"],
            f"prediction manifest lineage changed: {candidate_id}",
        )
    guard = protocol["guard"]
    maximum_residual_m = float(guard["maximum_residual_m"])
    regret_quantile = float(guard["within_execution_regret_quantile"])
    maximum_regret = float(guard["maximum_allowed_regret_m"])
    minimum_frames = int(guard["minimum_validation_frame_count"])
    fallback_candidate = str(protocol["selection"]["physical_fallback_candidate"])
    decisions: list[dict[str, object]] = []
    with _atomic_output_directory(output_dir, force=force) as temporary:
        for prefix_record in prefix_manifest["cases"]:
            case_id = str(prefix_record["case_id"])
            prefix_case = _load_case_npz(
                prefix_dir / str(prefix_record["path"]),
                str(prefix_record["sha256"]),
            )
            fit_end = int(prefix_case["fit_end"])
            train_end = int(prefix_case["train_end"])
            validation_count = train_end - fit_end
            _require(
                validation_count >= minimum_frames,
                f"validation prefix is too short: {case_id}",
            )
            baseline = np.asarray(
                prefix_case["baseline_prefix_m"],
                dtype=np.float64,
            )
            observed = np.asarray(
                prefix_case["observed_prefix_m"],
                dtype=np.float64,
            )
            visible = np.asarray(prefix_case["visible_prefix"])
            gt_track = np.asarray(
                prefix_case["gt_track_prefix_m"],
                dtype=np.float64,
            )
            lift_indices = np.asarray(
                prefix_case["lift_indices"],
                dtype=np.int64,
            )
            lift_weights = np.asarray(
                prefix_case["lift_weights"],
                dtype=np.float64,
            )
            num_surface_points = int(prefix_case["num_surface_points"])
            fallback_metrics = _metrics_by_frame(
                baseline,
                observed,
                visible,
                gt_track,
                num_surface_points=num_surface_points,
                start_frame=fit_end,
                end_frame=train_end,
            )
            for candidate_id, manifest in prediction_manifests.items():
                candidate_record = next(
                    row for row in manifest["case_records"] if row["case_id"] == case_id
                )
                prediction = _load_case_npz(
                    predictions_root / candidate_id / str(candidate_record["path"]),
                    str(candidate_record["sha256"]),
                )
                successful = bool(np.asarray(prediction["prediction_success"]).item())
                reasons: list[str] = []
                if candidate_id == fallback_candidate:
                    accepted = False
                    reasons.append("registered-physical-fallback")
                    raw_metrics = fallback_metrics
                elif not successful:
                    accepted = False
                    reasons.append("candidate-technical-failure")
                    raw_metrics = fallback_metrics
                else:
                    trajectory = _candidate_trajectory(
                        baseline,
                        np.asarray(
                            prediction["validation_mean_m"],
                            dtype=np.float64,
                        ),
                        start_frame=fit_end,
                        lift_indices=lift_indices,
                        lift_weights=lift_weights,
                        maximum_residual_m=maximum_residual_m,
                    )
                    raw_metrics = _metrics_by_frame(
                        trajectory,
                        observed,
                        visible,
                        gt_track,
                        num_surface_points=num_surface_points,
                        start_frame=fit_end,
                        end_frame=train_end,
                    )
                    accepted = True
                metric_rows: dict[str, object] = {}
                for metric in PRIMARY_METRICS:
                    raw_values = np.asarray(
                        raw_metrics[metric],
                        dtype=np.float64,
                    )
                    fallback_values = np.asarray(
                        fallback_metrics[metric],
                        dtype=np.float64,
                    )
                    regret = raw_values - fallback_values
                    quantile_value = _higher_quantile(
                        regret,
                        regret_quantile,
                    )
                    metric_rows[metric] = {
                        "candidate_mean_m": float(np.mean(raw_values)),
                        "fallback_mean_m": float(np.mean(fallback_values)),
                        "regret_quantile_m": quantile_value,
                    }
                    if (
                        candidate_id != fallback_candidate
                        and successful
                        and quantile_value > maximum_regret + 1e-12
                    ):
                        accepted = False
                        reasons.append(f"{metric}-validation-regret")
                if accepted:
                    reasons.append("metric-specific-guard-passed")
                decisions.append(
                    {
                        "case_id": case_id,
                        "candidate_id": candidate_id,
                        "accepted": accepted,
                        "reasons": sorted(set(reasons)),
                        "validation_frame_count": validation_count,
                        "metrics": metric_rows,
                    }
                )
        descriptor: dict[str, object] = {
            "contract": ADMISSION_MANIFEST_CONTRACT,
            "schema_version": 1,
            "protocol_id": protocol_id,
            "prefix_manifest_id": prefix_manifest["prefix_manifest_id"],
            "prediction_artifact_sha256": {
                candidate_id: manifest["prediction_artifact_sha256"]
                for candidate_id, manifest in sorted(prediction_manifests.items())
            },
            "guard": guard,
            "decision_count": len(decisions),
            "decisions": decisions,
            "information_boundary": {
                "only_prefix_validation_scored": True,
                "future_outcomes_read": False,
                "confirmation_payload_opened": False,
                "target_outcome_opened": False,
                "admission_sealed_before_future_scoring": True,
            },
        }
        descriptor["admission_manifest_id"] = _canonical_sha256(descriptor)
        _write_json(
            temporary / "admission_manifest.json",
            descriptor,
        )
    return _load_admission_manifest(output_dir)


def _regularized_gaussian_nll(
    error_m: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    observation_std_m: float,
    eigenvalue_floor_m2: float,
) -> np.ndarray:
    error = np.asarray(error_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    _require(
        error.ndim == 2
        and error.shape[1] == 3
        and covariance.shape == (len(error), 3, 3),
        "proper-score arrays have incompatible shapes",
    )
    identity = np.eye(3, dtype=np.float64)
    result = np.empty(len(error), dtype=np.float64)
    for index in range(len(error)):
        matrix = 0.5 * (covariance[index] + covariance[index].T)
        matrix = matrix + observation_std_m**2 * identity
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(
            eigenvalues,
            eigenvalue_floor_m2,
        )
        inverse = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
        mahalanobis = float(error[index] @ inverse @ error[index])
        log_determinant = float(np.sum(np.log(eigenvalues)))
        result[index] = 0.5 * (
            3.0 * math.log(2.0 * math.pi) + log_determinant + mahalanobis
        )
    return result


def _proper_score_by_horizon(
    residual_future_m: np.ndarray,
    valid_future: np.ndarray,
    mean_future_m: np.ndarray,
    covariance_future_m2: np.ndarray,
    *,
    observation_std_m: float,
    eigenvalue_floor_m2: float,
) -> dict[str, float]:
    residual = np.asarray(
        residual_future_m,
        dtype=np.float64,
    )
    validity = np.asarray(valid_future)
    mean = np.asarray(mean_future_m, dtype=np.float64)
    covariance = np.asarray(
        covariance_future_m2,
        dtype=np.float64,
    )
    _require(
        validity.dtype.kind == "b"
        and validity.shape == residual.shape[:2]
        and mean.shape == residual.shape
        and covariance.shape == residual.shape[:2] + (3, 3),
        "proper-score future arrays differ",
    )
    output: dict[str, float] = {}
    for label, indices in _horizon_groups(len(residual)).items():
        event_mask = validity[indices]
        errors = (residual[indices] - mean[indices])[event_mask]
        covariances = covariance[indices][event_mask]
        _require(
            len(errors) > 0,
            f"no proper-score events in horizon {label}",
        )
        values = _regularized_gaussian_nll(
            errors,
            covariances,
            observation_std_m=observation_std_m,
            eigenvalue_floor_m2=eigenvalue_floor_m2,
        )
        output[label] = float(np.mean(values))
    return output


def _point_metrics_by_horizon(
    trajectory: np.ndarray,
    observed: np.ndarray,
    visible: np.ndarray,
    gt_track: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, dict[str, float]]:
    by_frame = _metrics_by_frame(
        trajectory,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    output: dict[str, dict[str, float]] = {}
    for label, indices in _horizon_groups(end_frame - start_frame).items():
        output[label] = {
            metric: float(np.mean(np.asarray(by_frame[metric])[indices]))
            for metric in PRIMARY_METRICS
        }
    return output


def _selection_payload(
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    source = protocol["selection"]
    keys = (
        "minimum_group_count",
        "minimum_relative_point_improvement",
        "maximum_worst_group_relative_regression",
        "maximum_harmful_accepted_count",
        "maximum_mean_proper_score_regression",
        "require_paired_point_upper_bound_nonpositive",
        "bootstrap_samples",
        "bootstrap_seed",
        "require_crossfit_stability",
        "maximum_interval_coverage_shortfall",
        "numerical_tolerance",
    )
    result = {key: source[key] for key in keys}
    result["nominal_interval_coverage"] = protocol["scoring"][
        "nominal_interval_coverage"
    ]
    return result


def _metric_point_loss_id(
    protocol: Mapping[str, Any],
    name: str,
) -> str:
    for row in protocol["scoring"]["point_tournaments"]:
        if row["name"] == name:
            return str(row["point_loss_id"])
    raise KeyError(name)


def _metric_arbitration(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    reference_candidate: str,
) -> dict[str, object]:
    selected = {
        name: str(report["selected_candidate"]) for name, report in reports.items()
    }
    passed = {
        name: bool(report["source_gate_passed"]) for name, report in reports.items()
    }
    unique = set(selected.values())
    advance = (
        all(passed.values())
        and len(unique) == 1
        and next(iter(unique)) != reference_candidate
    )
    final = next(iter(unique)) if advance else reference_candidate
    return {
        "contract": ARBITRATION_REPORT_CONTRACT,
        "schema_version": 1,
        "metric_selected_candidates": selected,
        "metric_source_gate_passed": passed,
        "metrics_agree": len(unique) == 1,
        "source_gate_passed": advance,
        "status": ("selected" if advance else "completed_no_selection"),
        "selected_candidate": final,
        "decision": (
            "advance-metric-agreed-candidate"
            if advance
            else "retain-reference-candidate"
        ),
        "claim_authorized": False,
    }


def score_tournament(
    data_root: Path,
    prefix_dir: Path,
    predictions_root: Path,
    admission_dir: Path,
    output_dir: Path,
    *,
    protocol_path: Path,
    evaluator_revision: str,
    force: bool,
) -> dict[str, Any]:
    """Open the already-public future only after all barriers are sealed."""

    from bayesian_phystwin.discrepancy_candidate_tournament import (
        DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
        analyze_discrepancy_candidate_tournament,
    )
    from bayesian_phystwin.phystwin_confirmatory import _split_for_case
    from bayesian_phystwin.phystwin_residual_dynamics import (
        _lift_map,
        _load_pickle,
        _target_validity,
    )

    evaluator_sha = _literal_sha(
        evaluator_revision,
        name="evaluator_revision",
        length=40,
    )
    protocol, protocol_id = _load_protocol(protocol_path)
    prefix_manifest = _load_prefix_manifest(prefix_dir)
    admission = _load_admission_manifest(admission_dir)
    _require(
        prefix_manifest["protocol_id"] == protocol_id
        and admission["protocol_id"] == protocol_id
        and admission["prefix_manifest_id"] == prefix_manifest["prefix_manifest_id"],
        "score-stage lineage differs from the sealed barriers",
    )
    candidate_ids = _candidate_ids(protocol)
    prediction_manifests = {
        candidate_id: _load_prediction_manifest(predictions_root / candidate_id)
        for candidate_id in candidate_ids
    }
    for candidate_id, manifest in prediction_manifests.items():
        _require(
            admission["prediction_artifact_sha256"][candidate_id]
            == manifest["prediction_artifact_sha256"],
            f"admission did not bind {candidate_id}",
        )
    decisions = {
        (str(row["case_id"]), str(row["candidate_id"])): bool(row["accepted"])
        for row in admission["decisions"]
    }
    _require(
        len(decisions) == len(prefix_manifest["cases"]) * len(candidate_ids),
        "admission decision roster is incomplete",
    )
    scoring = protocol["scoring"]
    observation_std_m = float(scoring["proper_score_observation_std_m"])
    eigenvalue_floor_m2 = float(scoring["covariance_eigenvalue_floor_m2"])
    maximum_residual_m = float(protocol["guard"]["maximum_residual_m"])
    raw_rows: list[dict[str, object]] = []
    case_ids = tuple(str(row["case_id"]) for row in prefix_manifest["cases"])
    for case_id in case_ids:
        case_dir = data_root / case_id
        fit_end, train_end, frame_count = _split_for_case(
            case_dir,
            float(protocol["cohort"]["fit_fraction"]),
        )
        data = _load_pickle(case_dir / "final_data.pkl")
        baseline = np.asarray(
            _load_pickle(case_dir / "inference.pkl"),
            dtype=np.float64,
        )[:frame_count]
        gt_track = np.asarray(
            _load_pickle(case_dir / "gt_track_3d.pkl"),
            dtype=np.float64,
        )[:frame_count]
        observed = np.asarray(
            data["object_points"],
            dtype=np.float64,
        )[:frame_count]
        visible = np.asarray(
            data["object_visibilities"],
            dtype=bool,
        )[:frame_count]
        motion_valid = np.asarray(
            data["object_motions_valid"],
            dtype=bool,
        )
        valid = _target_validity(
            np.asarray(data["object_visibilities"], dtype=bool),
            motion_valid,
        )[:frame_count]
        original_count = observed.shape[1]
        residual = observed - baseline[:, :original_count]
        lift_indices, lift_weights = _lift_map(
            baseline[0],
            original_count,
            4,
        )
        num_surface_points = original_count + int(
            np.asarray(data["surface_points"]).shape[0]
        )
        fallback_point = _point_metrics_by_horizon(
            baseline,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=train_end,
            end_frame=frame_count,
        )
        future_count = frame_count - train_end
        zero_mean = np.zeros(
            (future_count, original_count, 3),
            dtype=np.float64,
        )
        zero_covariance = np.zeros(
            (future_count, original_count, 3, 3),
            dtype=np.float64,
        )
        fallback_proper = _proper_score_by_horizon(
            residual[train_end:],
            valid[train_end:],
            zero_mean,
            zero_covariance,
            observation_std_m=observation_std_m,
            eigenvalue_floor_m2=eigenvalue_floor_m2,
        )
        for candidate_id in candidate_ids:
            manifest = prediction_manifests[candidate_id]
            record = next(
                row for row in manifest["case_records"] if row["case_id"] == case_id
            )
            prediction = _load_case_npz(
                predictions_root / candidate_id / str(record["path"]),
                str(record["sha256"]),
            )
            _require(
                int(prediction["fit_end"]) == fit_end
                and int(prediction["train_end"]) == train_end
                and int(prediction["frame_count"]) == frame_count,
                f"prediction split changed: {candidate_id}/{case_id}",
            )
            mean = np.asarray(
                prediction["future_mean_m"],
                dtype=np.float64,
            )
            covariance = np.asarray(
                prediction["future_covariance_m2"],
                dtype=np.float64,
            )
            trajectory = _candidate_trajectory(
                baseline,
                mean,
                start_frame=train_end,
                lift_indices=lift_indices,
                lift_weights=lift_weights,
                maximum_residual_m=maximum_residual_m,
            )
            point = _point_metrics_by_horizon(
                trajectory,
                observed,
                visible,
                gt_track,
                num_surface_points=num_surface_points,
                start_frame=train_end,
                end_frame=frame_count,
            )
            proper = _proper_score_by_horizon(
                residual[train_end:],
                valid[train_end:],
                mean,
                covariance,
                observation_std_m=observation_std_m,
                eigenvalue_floor_m2=eigenvalue_floor_m2,
            )
            accepted = decisions[(case_id, candidate_id)]
            for horizon in HORIZON_LABELS:
                raw_rows.append(
                    {
                        "case_id": case_id,
                        "candidate_id": candidate_id,
                        "horizon": horizon,
                        "accepted": accepted,
                        "point": point[horizon],
                        "fallback_point": fallback_point[horizon],
                        "proper_score": proper[horizon],
                        "fallback_proper_score": fallback_proper[horizon],
                    }
                )
    unit_roster = [
        f"{case_id}/{horizon}" for case_id in case_ids for horizon in HORIZON_LABELS
    ]
    barrier_descriptor = {
        "protocol_id": protocol_id,
        "prefix_manifest_id": prefix_manifest["prefix_manifest_id"],
        "prediction_artifact_sha256": {
            candidate_id: prediction_manifests[candidate_id][
                "prediction_artifact_sha256"
            ]
            for candidate_id in candidate_ids
        },
        "admission_manifest_id": admission["admission_manifest_id"],
        "future_opened_after_barrier": True,
    }
    prediction_barrier_sha256 = _canonical_sha256(barrier_descriptor)
    candidates_payload = []
    for candidate_id in candidate_ids:
        candidate = _candidate_spec(protocol, candidate_id)
        manifest = prediction_manifests[candidate_id]
        candidates_payload.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "state_dimension": int(manifest["state_dimension"]),
                "parameter_count": int(candidate["declared_parameter_count"]),
                "runtime_milliseconds": 0.0,
                "covariance_bytes": int(manifest["covariance_bytes"]),
                "source_revision": candidate["source_revision"],
                "configuration_sha256": manifest["configuration_sha256"],
                "prediction_artifact_sha256": manifest["prediction_artifact_sha256"],
            }
        )
    metric_map = {
        "track": "track_error_m",
        "chamfer": "chamfer_distance_m",
    }
    reports: dict[str, dict[str, Any]] = {}
    with _atomic_output_directory(output_dir, force=force) as temporary:
        _write_json(
            temporary / "prediction_barrier.json",
            {
                **barrier_descriptor,
                "prediction_barrier_sha256": (prediction_barrier_sha256),
            },
        )
        _write_json(
            temporary / "raw_scored_rows.json",
            {
                "protocol_id": protocol_id,
                "claim_authorized": False,
                "rows": raw_rows,
            },
        )
        for tournament_name, metric in metric_map.items():
            records: list[dict[str, object]] = []
            for row in raw_rows:
                point_loss = float(row["point"][metric])
                fallback_point_loss = float(row["fallback_point"][metric])
                proper_score = float(row["proper_score"])
                fallback_proper_score = float(row["fallback_proper_score"])
                accepted = bool(row["accepted"])
                records.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "unit_id": (f"{row['case_id']}/{row['horizon']}"),
                        "group_id": row["case_id"],
                        "horizon": row["horizon"],
                        "accepted": accepted,
                        "point_loss": point_loss,
                        "fallback_point_loss": fallback_point_loss,
                        "deployed_point_loss": (
                            point_loss if accepted else fallback_point_loss
                        ),
                        "proper_score": proper_score,
                        "fallback_proper_score": (fallback_proper_score),
                        "deployed_proper_score": (
                            proper_score if accepted else fallback_proper_score
                        ),
                        "interval_covered": None,
                        "interval_width": None,
                    }
                )
            payload: dict[str, object] = {
                "contract": DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
                "schema_version": 1,
                "protocol_id": (f"{protocol_id}-{tournament_name}"),
                "statistical_unit": STATISTICAL_UNIT,
                "split": "source-only",
                "reference_candidate": protocol["selection"]["reference_candidate"],
                "physical_fallback_candidate": protocol["selection"][
                    "physical_fallback_candidate"
                ],
                "information_boundary": {
                    "candidate_predictions_sealed_before_scoring": True,
                    "candidate_generation_used_scored_targets": False,
                    "future_observations_used": False,
                    "confirmation_payloads_opened": False,
                    "replacement_allowed": False,
                },
                "evaluation": {
                    "evaluator_revision": evaluator_sha,
                    "scoring_policy_sha256": _canonical_sha256(
                        {
                            "scoring": scoring,
                            "metric": tournament_name,
                            "guard": protocol["guard"],
                        }
                    ),
                    "scored_unit_roster_sha256": (_canonical_sha256(unit_roster)),
                    "physical_fallback_artifact_sha256": (
                        prediction_manifests[
                            protocol["selection"]["physical_fallback_candidate"]
                        ]["prediction_artifact_sha256"]
                    ),
                    "prediction_barrier_sha256": (prediction_barrier_sha256),
                    "point_loss_id": _metric_point_loss_id(
                        protocol,
                        tournament_name,
                    ),
                    "proper_score_id": scoring["proper_score_id"],
                    "interval_semantics_id": scoring["interval_semantics_id"],
                },
                "selection": _selection_payload(protocol),
                "candidates": candidates_payload,
                "records": records,
            }
            report = analyze_discrepancy_candidate_tournament(payload)
            reports[tournament_name] = report
            _write_json(
                temporary / f"tournament-{tournament_name}-input.json",
                payload,
            )
            _write_json(
                temporary / f"tournament-{tournament_name}-report.json",
                report,
            )
        arbitration = _metric_arbitration(
            reports,
            reference_candidate=str(protocol["selection"]["reference_candidate"]),
        )
        arbitration.update(
            {
                "protocol_id": protocol_id,
                "prediction_barrier_sha256": (prediction_barrier_sha256),
                "admission_manifest_id": admission["admission_manifest_id"],
                "metric_report_ids": {
                    name: report["report_id"] for name, report in reports.items()
                },
                "claim_boundary": protocol["claim_boundary"],
            }
        )
        arbitration["report_id"] = _canonical_sha256(arbitration)
        _write_json(
            temporary / "metric_arbitration_report.json",
            arbitration,
        )
    return json.loads(
        (output_dir / "metric_arbitration_report.json").read_text(encoding="utf-8")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare = subparsers.add_parser(
        "prepare-prefix",
        help="Publish prefix-only full-22 candidate inputs.",
    )
    prepare.add_argument("data_root", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--force", action="store_true")

    predict = subparsers.add_parser(
        "predict",
        help="Seal one candidate's raw validation and future forecasts.",
    )
    predict.add_argument("prefix_dir", type=Path)
    predict.add_argument("output_dir", type=Path)
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--candidate", required=True)
    predict.add_argument("--source-revision", required=True)
    predict.add_argument("--force", action="store_true")

    admit = subparsers.add_parser(
        "admit",
        help="Seal prefix-only metric-specific guard decisions.",
    )
    admit.add_argument("prefix_dir", type=Path)
    admit.add_argument("predictions_root", type=Path)
    admit.add_argument("output_dir", type=Path)
    admit.add_argument("--protocol", type=Path, required=True)
    admit.add_argument("--force", action="store_true")

    score = subparsers.add_parser(
        "score",
        help="Open the public future and run both metric tournaments.",
    )
    score.add_argument("data_root", type=Path)
    score.add_argument("prefix_dir", type=Path)
    score.add_argument("predictions_root", type=Path)
    score.add_argument("admission_dir", type=Path)
    score.add_argument("output_dir", type=Path)
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--evaluator-revision", required=True)
    score.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-prefix":
        result = prepare_prefix(
            args.data_root,
            args.output_dir,
            protocol_path=args.protocol,
            force=args.force,
        )
    elif args.command == "predict":
        result = predict_candidate(
            args.prefix_dir,
            args.output_dir,
            protocol_path=args.protocol,
            candidate_id=args.candidate,
            source_revision=args.source_revision,
            force=args.force,
        )
    elif args.command == "admit":
        result = admit_predictions(
            args.prefix_dir,
            args.predictions_root,
            args.output_dir,
            protocol_path=args.protocol,
            force=args.force,
        )
    else:
        result = score_tournament(
            args.data_root,
            args.prefix_dir,
            args.predictions_root,
            args.admission_dir,
            args.output_dir,
            protocol_path=args.protocol,
            evaluator_revision=args.evaluator_revision,
            force=args.force,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
