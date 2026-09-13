#!/usr/bin/env python3
"""Audit deterministic point-mean equivalence for frozen DEFORM residual models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    build_deform_local_residual_features,
)

Array = np.ndarray


@dataclass(frozen=True)
class PanelFiles:
    """Resolved immutable inputs for one DEFORM evaluation panel."""

    dlo: str
    predictions: Path
    model: Path
    manifest: Path
    base_key: str
    candidate_key: str
    shrinkage: float
    provenance: Mapping[str, object]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_path(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    label: str,
) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    if expected_size is not None and resolved.stat().st_size != expected_size:
        raise ValueError(f"{label} size differs: {resolved}")
    if expected_sha256 is not None and _sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} digest differs: {resolved}")
    return resolved


def _search_file_by_sha256(
    roots: Sequence[Path],
    *,
    filename: str,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
) -> Path:
    pruned = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "data_set",
        "datasets",
        "node_modules",
        "site-packages",
        "venv",
    }
    examined = 0
    for root in roots:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            continue
        for directory, names, files in os.walk(
            resolved_root,
            topdown=True,
            onerror=lambda _error: None,
        ):
            names[:] = [name for name in names if name not in pruned]
            if filename not in files:
                continue
            candidate = Path(directory) / filename
            examined += 1
            if expected_size is not None and candidate.stat().st_size != expected_size:
                continue
            if _sha256_file(candidate) == expected_sha256:
                return candidate.resolve()
    raise FileNotFoundError(
        f"could not locate {label} by digest after examining {examined} candidates"
    )


def _verified_identity(
    value: object,
    *,
    label: str,
    search_roots: Sequence[Path] = (),
) -> Path:
    identity = _mapping(value, label=label)
    expected_sha256 = str(identity.get("sha256", ""))
    if len(expected_sha256) != 64:
        raise ValueError(f"{label} has no valid SHA-256")
    expected_size_value = identity.get("size_bytes")
    expected_size = (
        int(cast(Any, expected_size_value))
        if expected_size_value is not None
        else None
    )
    declared = Path(str(identity.get("path", ""))).expanduser()
    if declared.is_file():
        return _verify_path(
            declared,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            label=label,
        )
    if not search_roots:
        raise FileNotFoundError(f"{label} declared path is unavailable: {declared}")
    return _search_file_by_sha256(
        search_roots,
        filename=declared.name,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        label=label,
    )


def _resolve_dlo2(
    spec: Mapping[str, object], search_roots: Sequence[Path]
) -> PanelFiles:
    prediction_sha256 = str(spec.get("prediction_sha256", ""))
    prediction_size = int(cast(Any, spec.get("prediction_size_bytes", -1)))
    explicit_root = str(spec.get("root", "")).strip()
    if explicit_root:
        predictions = _verify_path(
            Path(explicit_root) / "official_prediction.npz",
            expected_sha256=prediction_sha256,
            expected_size=prediction_size,
            label="DLO2 official prediction",
        )
    else:
        predictions = _search_file_by_sha256(
            search_roots,
            filename="official_prediction.npz",
            expected_sha256=prediction_sha256,
            expected_size=prediction_size,
            label="DLO2 official prediction",
        )
    root = predictions.parent
    authorization_path = _verify_path(
        root / "authorization.json", label="DLO2 authorization"
    )
    authorization = _read_json(authorization_path)
    model = _verified_identity(
        authorization.get("local_residual_model"),
        label="DLO2 local residual model",
        search_roots=search_roots,
    )
    expected_model_sha256 = str(spec.get("model_sha256", ""))
    if expected_model_sha256 and _sha256_file(model) != expected_model_sha256:
        raise ValueError("DLO2 local residual model differs from the request")
    fixed = _mapping(authorization.get("fixed_arm"), label="DLO2 fixed arm")
    manifest = _verify_path(
        root / "evaluation_manifest.json", label="DLO2 evaluation manifest"
    )
    return PanelFiles(
        dlo="DLO2",
        predictions=predictions,
        model=model,
        manifest=manifest,
        base_key="baseline_predictions",
        candidate_key="candidate_predictions",
        shrinkage=float(cast(Any, fixed["shrinkage"])),
        provenance={
            "authorization": _identity(authorization_path),
            "discovery": "content-addressed-search",
        },
    )


def _resolve_dlo3(
    spec: Mapping[str, object], search_roots: Sequence[Path]
) -> PanelFiles:
    root = Path(str(spec["root"])).expanduser().resolve()
    seal_path = _verify_path(root / "prediction_seal.json", label="DLO3 seal")
    seal = _read_json(seal_path)
    predictions = _verified_identity(
        seal.get("predictions"),
        label="DLO3 predictions",
        search_roots=search_roots,
    )
    manifest = _verified_identity(
        seal.get("panel_manifest"),
        label="DLO3 evaluation manifest",
        search_roots=search_roots,
    )
    authorization_path = _verify_path(
        root / "authorization.json", label="DLO3 authorization"
    )
    authorization = _read_json(authorization_path)
    final_method_path = _verified_identity(
        authorization.get("final_method"),
        label="DLO3 final method",
        search_roots=search_roots,
    )
    final_method = _read_json(final_method_path)
    model = _verified_identity(
        final_method.get("full_covariance_model"),
        label="DLO3 full covariance model",
        search_roots=search_roots,
    )
    return PanelFiles(
        dlo="DLO3",
        predictions=predictions,
        model=model,
        manifest=manifest,
        base_key="baseline",
        candidate_key="candidate",
        shrinkage=float(cast(Any, final_method["shrinkage"])),
        provenance={
            "prediction_seal": _identity(seal_path),
            "authorization": _identity(authorization_path),
            "final_method": _identity(final_method_path),
        },
    )


def _resolve_dlo45(
    dlo: str,
    spec: Mapping[str, object],
    search_roots: Sequence[Path],
) -> PanelFiles:
    root = Path(str(spec["root"])).expanduser().resolve()
    seal_path = _verify_path(root / "prediction_seal.json", label=f"{dlo} seal")
    seal = _read_json(seal_path)
    if seal.get("dlo") != dlo:
        raise ValueError(f"{dlo} prediction seal label differs")
    predictions = _verified_identity(
        seal.get("predictions"),
        label=f"{dlo} predictions",
        search_roots=search_roots,
    )
    manifest = _verified_identity(
        seal.get("eval_manifest"),
        label=f"{dlo} evaluation manifest",
        search_roots=search_roots,
    )
    method_path = _verified_identity(
        seal.get("method_seal"),
        label=f"{dlo} method seal",
        search_roots=search_roots,
    )
    method = _read_json(method_path)
    model = _verified_identity(
        method.get("full_covariance_model"),
        label=f"{dlo} full covariance model",
        search_roots=search_roots,
    )
    return PanelFiles(
        dlo=dlo,
        predictions=predictions,
        model=model,
        manifest=manifest,
        base_key="physical",
        candidate_key="candidate",
        shrinkage=float(cast(Any, method["shrinkage"])),
        provenance={
            "prediction_seal": _identity(seal_path),
            "method_seal": _identity(method_path),
        },
    )


def _load_trajectory(path: Path, *, frame_count: int, node_count: int) -> Array:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    array = np.asarray(raw, dtype=np.float32)
    expected_shape = (frame_count, 3, node_count)
    if array.shape != expected_shape:
        raise ValueError(f"{path}: expected {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: trajectory contains non-finite values")
    nodes = np.transpose(array, (0, 2, 1)).copy()
    nodes[:, :, 2] = np.clip(nodes[:, :, 2], 2e-3 + 1e-6, 10000.0)
    return nodes


def _load_truth_and_inputs(
    manifest_path: Path, names: Sequence[str]
) -> tuple[Array, Array, Array]:
    manifest = _read_json(manifest_path)
    ordered_names = [
        str(value)
        for value in _sequence(manifest.get("ordered_names"), label="ordered names")
    ]
    if ordered_names != list(names):
        raise ValueError("prediction names differ from the sealed manifest")
    identities = _mapping(manifest.get("trajectories"), label="trajectories")
    full = []
    for name in names:
        identity = _mapping(identities.get(name), label=f"trajectory {name}")
        path = _verify_path(
            Path(str(identity.get("path", ""))),
            expected_sha256=str(identity.get("sha256", "")),
            expected_size=(
                int(cast(Any, identity["size_bytes"]))
                if "size_bytes" in identity
                else None
            ),
            label=f"trajectory {name}",
        )
        full.append(_load_trajectory(path, frame_count=500, node_count=12))
    trajectories = np.stack(full)
    clamped = np.asarray((0, 1, 10, 11), dtype=np.int64)
    initial = trajectories[:, :2].copy()
    action = trajectories[:, 2:, clamped].copy()
    truth = trajectories[:, 2:].copy()
    return initial, action, truth


def _load_point_model(path: Path) -> dict[str, Array | int]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "node_count",
            "prediction_horizon",
            "feature_location",
            "feature_scale",
            "coefficients",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"point model omits keys: {missing}")
        return {
            "node_count": int(np.asarray(archive["node_count"]).reshape(-1)[0]),
            "prediction_horizon": int(
                np.asarray(archive["prediction_horizon"]).reshape(-1)[0]
            ),
            "feature_location": np.asarray(
                archive["feature_location"], dtype=np.float64
            ),
            "feature_scale": np.asarray(archive["feature_scale"], dtype=np.float64),
            "coefficients": np.asarray(archive["coefficients"], dtype=np.float64),
        }


def _apply_deterministic_point_mean(
    features: Array,
    frames: Array,
    baseline: Array,
    *,
    feature_location: Array,
    feature_scale: Array,
    coefficients: Array,
    shrinkage: float,
) -> Array:
    """Apply the deterministic ridge mean without reading covariance arrays."""

    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    internal_count = baseline.shape[2] - 4
    feature_count = features.shape[3]
    if (
        feature_location.shape != (internal_count, feature_count)
        or feature_scale.shape != feature_location.shape
        or coefficients.shape != (internal_count, feature_count + 1, 3)
        or frames.shape != (baseline.shape[0], 3, 3)
    ):
        raise ValueError("deterministic point-mean arrays do not align")
    canonical = []
    for node in range(internal_count):
        standardized = (
            features[:, :, node] - feature_location[node]
        ) / feature_scale[node]
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        canonical.append(np.einsum("ntd,dc->ntc", design, coefficients[node]))
    correction_canonical = np.stack(canonical, axis=2)
    correction_global = np.einsum("ntvj,nij->ntvi", correction_canonical, frames)
    result = np.asarray(baseline, dtype=np.float64).copy()
    internal = np.arange(2, baseline.shape[2] - 2, dtype=np.int64)
    result[:, :, internal] += shrinkage * correction_global
    return result


def _reconstruct_deterministic(
    model_path: Path,
    initial: Array,
    action: Array,
    baseline: Array,
    *,
    shrinkage: float,
) -> tuple[Array, float]:
    model = _load_point_model(model_path)
    if (
        cast(int, model["node_count"]) != baseline.shape[2]
        or cast(int, model["prediction_horizon"]) != baseline.shape[1]
    ):
        raise ValueError("point model dimensions differ from the prediction panel")
    started = time.perf_counter()
    features, frames = build_deform_local_residual_features(
        initial,
        action,
        baseline,
    )
    candidate = _apply_deterministic_point_mean(
        features,
        frames,
        baseline,
        feature_location=cast(Array, model["feature_location"]),
        feature_scale=cast(Array, model["feature_scale"]),
        coefficients=cast(Array, model["coefficients"]),
        shrinkage=shrinkage,
    )
    return candidate, time.perf_counter() - started


def _case_l1(prediction: Array, truth: Array) -> Array:
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes differ")
    return np.mean(np.abs(prediction - truth), axis=(1, 2, 3))


def _horizon_l1(prediction: Array, truth: Array) -> dict[str, float]:
    thirds = np.array_split(np.arange(prediction.shape[1]), 3)
    return {
        label: float(np.mean(np.abs(prediction[:, indices] - truth[:, indices])))
        for label, indices in zip(("early", "middle", "late"), thirds, strict=True)
    }


def _load_prediction_panel(files: PanelFiles) -> tuple[list[str], Array, Array]:
    with np.load(files.predictions, allow_pickle=False) as archive:
        for key in ("names", files.base_key, files.candidate_key):
            if key not in archive:
                raise ValueError(f"{files.dlo} predictions omit {key}")
        names = [str(value) for value in np.asarray(archive["names"])]
        baseline = np.asarray(archive[files.base_key], dtype=np.float64)
        candidate = np.asarray(archive[files.candidate_key], dtype=np.float64)
    expected_shape = (14, 498, 12, 3)
    if baseline.shape != expected_shape or candidate.shape != expected_shape:
        raise ValueError(f"{files.dlo} prediction shape differs")
    return names, baseline, candidate


def _audit_panel(
    files: PanelFiles,
    expected: Mapping[str, object],
    *,
    point_tolerance_m: float,
    score_tolerance_m: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    names, baseline, bayesian = _load_prediction_panel(files)
    initial, action, truth = _load_truth_and_inputs(files.manifest, names)
    deterministic, elapsed_seconds = _reconstruct_deterministic(
        files.model,
        initial,
        action,
        baseline,
        shrinkage=files.shrinkage,
    )
    difference = np.abs(deterministic - bayesian)
    max_abs = float(np.max(difference))
    exact = bool(np.array_equal(deterministic, bayesian))
    within_tolerance = max_abs <= point_tolerance_m
    base_cases = _case_l1(baseline, truth)
    deterministic_cases = _case_l1(deterministic, truth)
    bayesian_cases = _case_l1(bayesian, truth)
    case_rows = []
    for index, name in enumerate(names):
        case_rows.append(
            {
                "dlo": files.dlo,
                "name": name,
                "base_l1_m": float(base_cases[index]),
                "deterministic_l1_m": float(deterministic_cases[index]),
                "bayesian_l1_m": float(bayesian_cases[index]),
                "deterministic_to_base_ratio": float(
                    deterministic_cases[index] / base_cases[index]
                ),
                "deterministic_minus_bayesian_l1_m": float(
                    deterministic_cases[index] - bayesian_cases[index]
                ),
            }
        )
    base_mean = float(np.mean(base_cases))
    deterministic_mean = float(np.mean(deterministic_cases))
    bayesian_mean = float(np.mean(bayesian_cases))
    expected_base = float(cast(Any, expected["base_mean_l1_m"]))
    expected_candidate = float(cast(Any, expected["candidate_mean_l1_m"]))
    expected_wins = int(cast(Any, expected["wins"]))
    wins = int(np.sum(deterministic_cases < base_cases))
    checks = {
        "deterministic_bayesian_point_equivalence": within_tolerance,
        "base_score_matches_frozen_result": abs(base_mean - expected_base)
        <= score_tolerance_m,
        "candidate_score_matches_frozen_result": abs(
            bayesian_mean - expected_candidate
        )
        <= score_tolerance_m,
        "deterministic_score_matches_bayesian": abs(
            deterministic_mean - bayesian_mean
        )
        <= score_tolerance_m,
        "win_count_matches_frozen_result": wins == expected_wins,
    }
    result = {
        "dlo": files.dlo,
        "case_count": len(names),
        "base_mean_l1_m": base_mean,
        "deterministic_mean_l1_m": deterministic_mean,
        "bayesian_mean_l1_m": bayesian_mean,
        "relative_improvement": (base_mean - deterministic_mean) / base_mean,
        "wins": wins,
        "maximum_case_ratio": float(
            np.max(deterministic_cases / base_cases)
        ),
        "point_equivalence": {
            "array_equal": exact,
            "max_abs_difference_m": max_abs,
            "tolerance_m": point_tolerance_m,
            "within_tolerance": within_tolerance,
        },
        "horizon_l1_m": {
            "base": _horizon_l1(baseline, truth),
            "deterministic": _horizon_l1(deterministic, truth),
            "bayesian": _horizon_l1(bayesian, truth),
        },
        "deterministic_reconstruction": {
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_trajectory": elapsed_seconds / len(names),
            "covariance_arrays_read": False,
        },
        "artifacts": {
            "predictions": _identity(files.predictions),
            "point_model": _identity(files.model),
            "evaluation_manifest": _identity(files.manifest),
        },
        "provenance": dict(files.provenance),
        "checks": checks,
        "passed": all(checks.values()),
    }
    return result, case_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty trajectory table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    request_path = args.request.resolve()
    request = _read_json(request_path)
    if (
        request.get("schema_version") != 1
        or request.get("contract")
        != "deform-deterministic-bayes-audit-request-v1"
        or request.get("runner_label") != "gpuserver4090"
    ):
        raise ValueError("unsupported deterministic/Bayesian audit request")
    search_roots = [
        Path(str(value)).expanduser()
        for value in _sequence(request.get("search_roots"), label="search roots")
    ]
    tolerances = _mapping(request.get("tolerances"), label="tolerances")
    point_tolerance_m = float(
        cast(Any, tolerances["point_mean_max_abs_difference_m"])
    )
    score_tolerance_m = float(cast(Any, tolerances["score_max_abs_difference_m"]))
    if (
        not math.isfinite(point_tolerance_m)
        or point_tolerance_m <= 0.0
        or not math.isfinite(score_tolerance_m)
        or score_tolerance_m <= 0.0
    ):
        raise ValueError("audit tolerances must be positive")

    panels_spec = _mapping(request.get("panels"), label="panels")
    expected = _mapping(request.get("expected"), label="expected results")
    resolved = (
        _resolve_dlo2(
            _mapping(panels_spec.get("DLO2"), label="DLO2 panel"), search_roots
        ),
        _resolve_dlo3(
            _mapping(panels_spec.get("DLO3"), label="DLO3 panel"), search_roots
        ),
        _resolve_dlo45(
            "DLO4",
            _mapping(panels_spec.get("DLO4"), label="DLO4 panel"),
            search_roots,
        ),
        _resolve_dlo45(
            "DLO5",
            _mapping(panels_spec.get("DLO5"), label="DLO5 panel"),
            search_roots,
        ),
    )

    panel_results = []
    trajectory_rows: list[dict[str, object]] = []
    for files in resolved:
        result, rows = _audit_panel(
            files,
            _mapping(expected.get(files.dlo), label=f"{files.dlo} expected result"),
            point_tolerance_m=point_tolerance_m,
            score_tolerance_m=score_tolerance_m,
        )
        panel_results.append(result)
        trajectory_rows.extend(rows)

    equal_dlo_base = float(
        np.mean([float(panel["base_mean_l1_m"]) for panel in panel_results])
    )
    equal_dlo_deterministic = float(
        np.mean(
            [float(panel["deterministic_mean_l1_m"]) for panel in panel_results]
        )
    )
    wins = sum(int(panel["wins"]) for panel in panel_results)
    case_count = sum(int(panel["case_count"]) for panel in panel_results)
    max_abs = max(
        float(_mapping(panel["point_equivalence"], label="point equivalence")[
            "max_abs_difference_m"
        ])
        for panel in panel_results
    )
    point_supported = all(
        bool(_mapping(panel["point_equivalence"], label="point equivalence")[
            "within_tolerance"
        ])
        for panel in panel_results
    )
    residual_supported = wins == case_count and all(
        float(panel["deterministic_mean_l1_m"]) < float(panel["base_mean_l1_m"])
        for panel in panel_results
    )
    all_panels_passed = all(bool(panel["passed"]) for panel in panel_results)
    result = {
        "schema_version": 1,
        "contract": "deform-deterministic-bayes-point-equivalence-audit-v1",
        "audit_id": request.get("audit_id"),
        "request": _identity(request_path),
        "source_revision": os.environ.get("GITHUB_SHA", "unknown"),
        "runner": {
            "required_label": "gpuserver4090",
            "actual_name": os.environ.get("RUNNER_NAME", "unknown"),
        },
        "information_boundary": {
            "public_real_data_only": True,
            "existing_sealed_predictions_only": True,
            "physical_model_retrained": False,
            "residual_model_refit": False,
            "target_selection": False,
            "target_calibration": False,
            "original_artifacts_mutated": False,
            "audit_class": "retrospective-matched-deterministic-equivalence",
        },
        "panels": panel_results,
        "aggregate": {
            "dlo_count": len(panel_results),
            "trajectory_count": case_count,
            "wins": wins,
            "equal_dlo_base_mean_l1_m": equal_dlo_base,
            "equal_dlo_deterministic_mean_l1_m": equal_dlo_deterministic,
            "equal_dlo_relative_improvement": (
                equal_dlo_base - equal_dlo_deterministic
            )
            / equal_dlo_base,
            "maximum_deterministic_bayesian_point_difference_m": max_abs,
        },
        "decision": {
            "matched_deterministic_point_equivalence_supported": point_supported,
            "residual_adapter_accuracy_supported": residual_supported,
            "bayesian_point_accuracy_unique": not point_supported,
            "bayesian_value_attribution": (
                "predictive covariance and joint dependence, not the ridge point mean"
            ),
            "paper_claim_supported": point_supported
            and residual_supported
            and all_panels_passed,
        },
        "passed": point_supported and residual_supported and all_panels_passed,
    }
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"audit output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trajectory_metrics.csv", trajectory_rows)
    _write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
