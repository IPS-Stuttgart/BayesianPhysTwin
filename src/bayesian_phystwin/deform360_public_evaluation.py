"""Read-only public Deform360 trajectory evaluation.

The evaluator discovers released trajectory-like NPZ artifacts without assuming a
single private staging layout. Fixed-identity trajectories receive a rolling
one-step comparison between persistence, last residual, and the frozen endpoint
model average. Packed visual-hull archives receive the same comparison on global
translation with correspondence-free Chamfer scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from .endpoint_model_average import (
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

DEFORM360_PUBLIC_EVALUATION_SCHEMA = "bayesian-phystwin/deform360-public-v1"
CHI_SQUARE_3D_90 = 6.251388631170325
_OBJECT_PATTERN = re.compile(r"^\d{3}-.+")
_TRAJECTORY_KEY_HINTS = (
    "positions_world_m",
    "points_world_m",
    "positions_m",
    "control_points",
    "particles",
    "particle_tracks",
    "tracks",
    "trajectory",
    "positions",
    "points",
)
_VALID_KEY_HINTS = ("valid_mask", "track_valid", "visibility", "valid")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _result_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return None if len(finite) == 0 else float(np.mean(finite))


def _deterministic_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _object_id(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        if _OBJECT_PATTERN.match(part):
            return part
    return "unknown"


def _unit_scale(key: str, points: np.ndarray) -> tuple[float, str]:
    lowered = key.lower()
    if lowered.endswith("_mm") or "millimet" in lowered:
        return 1e-3, "declared_mm"
    if lowered.endswith("_m") or "world_m" in lowered:
        return 1.0, "declared_m"
    magnitude = float(np.nanmedian(np.linalg.norm(points.reshape(-1, 3), axis=1)))
    if magnitude > 20.0:
        return 1e-3, "heuristic_mm"
    return 1.0, "heuristic_m"


def _symmetric_chamfer_rmse(
    first: np.ndarray,
    second: np.ndarray,
    *,
    point_limit: int = 512,
    block_size: int = 128,
) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left = left[_deterministic_indices(len(left), point_limit)]
    right = right[_deterministic_indices(len(right), point_limit)]
    if len(left) == 0 or len(right) == 0:
        return float("nan")

    def directed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        minima = np.full(len(source), np.inf, dtype=np.float64)
        for start in range(0, len(target), block_size):
            block = target[start : start + block_size]
            squared = np.sum(
                np.square(source[:, None, :] - block[None, :, :]),
                axis=2,
            )
            minima = np.minimum(minima, np.min(squared, axis=1))
        return minima

    squared = 0.5 * (
        float(np.mean(directed(left, right)))
        + float(np.mean(directed(right, left)))
    )
    return float(np.sqrt(max(squared, 0.0)))


def _identity_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> float:
    mask = np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return float("nan")
    squared = np.sum(np.square(prediction[mask] - target[mask]), axis=1)
    return float(np.sqrt(np.mean(squared)))


def _coverage_and_nll(
    error: np.ndarray,
    covariance: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    mask = np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return float("nan"), float("nan")
    selected_error = np.asarray(error[mask], dtype=np.float64)
    selected_covariance = np.asarray(covariance[mask], dtype=np.float64)
    jitter = np.eye(3, dtype=np.float64)[None] * 1e-12
    selected_covariance = selected_covariance + jitter
    sign, logdet = np.linalg.slogdet(selected_covariance)
    usable = sign > 0
    if not np.any(usable):
        return float("nan"), float("nan")
    selected_error = selected_error[usable]
    selected_covariance = selected_covariance[usable]
    logdet = logdet[usable]
    solved = np.linalg.solve(selected_covariance, selected_error[..., None])[..., 0]
    quadratic = np.sum(selected_error * solved, axis=1)
    coverage = float(np.mean(quadratic <= CHI_SQUARE_3D_90))
    nll = 0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + quadratic)
    return coverage, float(np.mean(nll))


def _component_effective_count(weights: np.ndarray) -> float:
    probabilities = np.asarray(weights, dtype=np.float64)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-300)),
        axis=1,
    )
    return float(np.mean(np.exp(entropy)))


@dataclass(frozen=True, slots=True)
class EvaluationLimits:
    max_archives: int = 64
    max_frames_per_archive: int = 96
    max_tracks: int = 2048

    def __post_init__(self) -> None:
        for name in ("max_archives", "max_frames_per_archive", "max_tracks"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _iter_npz_paths(root: Path, *, limit: int) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for directory, names, files in os.walk(root):
        names[:] = sorted(
            name
            for name in names
            if name not in {".git", "__pycache__", "node_modules"}
        )
        for name in sorted(files):
            if name.endswith(".npz"):
                candidates.append(Path(directory) / name)
                if len(candidates) >= limit:
                    return tuple(candidates)
    return tuple(candidates)


def _trajectory_candidate(stored: Any) -> tuple[str, np.ndarray] | None:
    candidates: list[tuple[int, str, np.ndarray]] = []
    for key in stored.files:
        value = np.asarray(stored[key])
        if value.ndim != 3 or value.shape[-1] != 3:
            continue
        if value.shape[0] < 4 or value.shape[1] < 1:
            continue
        lowered = key.lower()
        rank = next(
            (index for index, hint in enumerate(_TRAJECTORY_KEY_HINTS) if hint in lowered),
            len(_TRAJECTORY_KEY_HINTS),
        )
        candidates.append((rank, key, value))
    if not candidates:
        return None
    _, key, value = min(candidates, key=lambda item: (item[0], item[1]))
    return key, value


def _validity(stored: Any, shape: tuple[int, int]) -> np.ndarray:
    for hint in _VALID_KEY_HINTS:
        for key in stored.files:
            if hint not in key.lower():
                continue
            value = np.asarray(stored[key])
            if value.shape == shape:
                return np.asarray(value, dtype=bool)
    return np.ones(shape, dtype=bool)


def _evaluate_fixed_trajectory(
    path: Path,
    root: Path,
    stored: Any,
    key: str,
    raw_points: np.ndarray,
    limits: EvaluationLimits,
) -> dict[str, Any]:
    scale, unit_source = _unit_scale(key, raw_points)
    points = np.asarray(raw_points, dtype=np.float64) * scale
    finite = np.all(np.isfinite(points), axis=2)
    valid = _validity(stored, points.shape[:2]) & finite
    frames = min(len(points), limits.max_frames_per_archive)
    tracks = _deterministic_indices(points.shape[1], limits.max_tracks)
    points = points[:frames, tracks]
    valid = valid[:frames, tracks]
    residual = np.diff(points, axis=0)
    residual_valid = valid[:-1] & valid[1:]
    steps: list[dict[str, Any]] = []

    for current in range(2, frames - 1):
        prefix = residual[:current]
        prefix_valid = residual_valid[:current]
        posterior = infer_model_averaged_endpoint(
            prefix,
            prefix_valid,
            end_frame=len(prefix),
        )
        predictive = predict_model_averaged_endpoint(posterior, horizon_steps=1)
        target = points[current + 1]
        target_valid = valid[current] & valid[current + 1]
        persistence = points[current]
        last_residual = points[current] + residual[current - 1]
        model_average = points[current] + predictive.mean_m
        coverage, nll = _coverage_and_nll(
            target - model_average,
            predictive.covariance_m2,
            target_valid,
        )
        steps.append(
            {
                "current_frame": current,
                "target_frame": current + 1,
                "valid_track_count": int(np.count_nonzero(target_valid)),
                "identity_rmse_m": {
                    "persistence": _identity_rmse(
                        persistence,
                        target,
                        target_valid,
                    ),
                    "last_residual": _identity_rmse(
                        last_residual,
                        target,
                        target_valid,
                    ),
                    "model_average": _identity_rmse(
                        model_average,
                        target,
                        target_valid,
                    ),
                },
                "chamfer_rmse_m": {
                    "persistence": _symmetric_chamfer_rmse(
                        persistence[target_valid],
                        target[target_valid],
                    ),
                    "last_residual": _symmetric_chamfer_rmse(
                        last_residual[target_valid],
                        target[target_valid],
                    ),
                    "model_average": _symmetric_chamfer_rmse(
                        model_average[target_valid],
                        target[target_valid],
                    ),
                },
                "model_average_raw_coverage_90": coverage,
                "model_average_mean_nll": nll,
                "model_average_effective_component_count": (
                    _component_effective_count(predictive.component_weights)
                ),
            }
        )

    return {
        "path": str(path.relative_to(root)),
        "object_id": _object_id(path, root),
        "representation": "fixed_identity_trajectory",
        "array_key": key,
        "unit_source": unit_source,
        "frame_count_used": frames,
        "track_count_used": len(tracks),
        "step_count": len(steps),
        "steps": steps,
    }


def _packed_hulls(stored: Any) -> tuple[np.ndarray, tuple[np.ndarray, ...]] | None:
    required = {"frame_indices", "point_offsets", "points_world_m"}
    if not required.issubset(set(stored.files)):
        return None
    frames = np.asarray(stored["frame_indices"], dtype=np.int64)
    offsets = np.asarray(stored["point_offsets"], dtype=np.int64)
    points = np.asarray(stored["points_world_m"], dtype=np.float64)
    if frames.ndim != 1 or len(frames) < 4:
        return None
    if offsets.shape != (len(frames) + 1,) or offsets[0] != 0:
        return None
    if offsets[-1] != len(points) or points.ndim != 2 or points.shape[1] != 3:
        return None
    hulls = tuple(
        points[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(frames))
    )
    if not all(len(hull) > 0 and np.all(np.isfinite(hull)) for hull in hulls):
        return None
    return frames, hulls


def _evaluate_packed_hulls(
    path: Path,
    root: Path,
    frames: np.ndarray,
    hulls: tuple[np.ndarray, ...],
    limits: EvaluationLimits,
) -> dict[str, Any]:
    count = min(len(hulls), limits.max_frames_per_archive)
    selected_hulls = tuple(
        hull[_deterministic_indices(len(hull), limits.max_tracks)]
        for hull in hulls[:count]
    )
    centroids = np.asarray([np.mean(hull, axis=0) for hull in selected_hulls])
    residual = np.diff(centroids, axis=0)[:, None, :]
    valid = np.ones(residual.shape[:2], dtype=bool)
    steps: list[dict[str, Any]] = []

    for current in range(2, count - 1):
        prefix = residual[:current]
        posterior = infer_model_averaged_endpoint(
            prefix,
            valid[:current],
            end_frame=len(prefix),
        )
        predictive = predict_model_averaged_endpoint(posterior, horizon_steps=1)
        current_hull = selected_hulls[current]
        target_hull = selected_hulls[current + 1]
        last_translation = residual[current - 1, 0]
        model_translation = predictive.mean_m[0]
        methods = {
            "persistence": current_hull,
            "last_residual": current_hull + last_translation,
            "model_average": current_hull + model_translation,
        }
        target_translation = centroids[current + 1] - centroids[current]
        coverage, nll = _coverage_and_nll(
            (target_translation - model_translation)[None],
            predictive.covariance_m2,
            np.ones(1, dtype=bool),
        )
        steps.append(
            {
                "current_frame": int(frames[current]),
                "target_frame": int(frames[current + 1]),
                "centroid_error_m": {
                    name: float(
                        np.linalg.norm(np.mean(prediction, axis=0) - centroids[current + 1])
                    )
                    for name, prediction in methods.items()
                },
                "chamfer_rmse_m": {
                    name: _symmetric_chamfer_rmse(prediction, target_hull)
                    for name, prediction in methods.items()
                },
                "model_average_raw_coverage_90": coverage,
                "model_average_mean_nll": nll,
                "model_average_effective_component_count": (
                    _component_effective_count(predictive.component_weights)
                ),
            }
        )

    return {
        "path": str(path.relative_to(root)),
        "object_id": _object_id(path, root),
        "representation": "packed_visual_hulls",
        "array_key": "points_world_m",
        "unit_source": "declared_m",
        "frame_count_used": count,
        "track_count_used": None,
        "step_count": len(steps),
        "steps": steps,
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("identity_rmse_m", "centroid_error_m", "chamfer_rmse_m")
    methods = ("persistence", "last_residual", "model_average")
    summary: dict[str, Any] = {}
    for representation in sorted({case["representation"] for case in cases}):
        selected = [case for case in cases if case["representation"] == representation]
        representation_summary: dict[str, Any] = {
            "archive_count": len(selected),
            "object_count": len({case["object_id"] for case in selected}),
            "step_count": int(sum(case["step_count"] for case in selected)),
        }
        for metric in metric_names:
            method_values: dict[str, list[float]] = {method: [] for method in methods}
            for case in selected:
                per_case: dict[str, list[float]] = {method: [] for method in methods}
                for step in case["steps"]:
                    if metric not in step:
                        continue
                    for method in methods:
                        per_case[method].append(float(step[metric][method]))
                for method in methods:
                    value = _finite_mean(per_case[method])
                    if value is not None:
                        method_values[method].append(value)
            if any(method_values.values()):
                representation_summary[metric] = {
                    method: _finite_mean(values)
                    for method, values in method_values.items()
                }
        coverage_values = [
            float(step["model_average_raw_coverage_90"])
            for case in selected
            for step in case["steps"]
        ]
        component_values = [
            float(step["model_average_effective_component_count"])
            for case in selected
            for step in case["steps"]
        ]
        representation_summary["model_average_raw_coverage_90"] = _finite_mean(
            coverage_values
        )
        representation_summary["model_average_effective_component_count"] = (
            _finite_mean(component_values)
        )
        summary[representation] = representation_summary
    return summary


def evaluate_deform360_public_data(
    data_root: Path,
    *,
    limits: EvaluationLimits | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Evaluate all supported public Deform360 trajectory artifacts read-only."""

    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Deform360 data root is missing: {root}")
    settings = EvaluationLimits() if limits is None else limits
    paths = _iter_npz_paths(root, limit=settings.max_archives * 8)
    cases: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []

    for path in paths:
        if len(cases) >= settings.max_archives:
            break
        record: dict[str, Any] = {"path": str(path.relative_to(root))}
        try:
            with np.load(path, allow_pickle=False) as stored:
                record["keys"] = sorted(stored.files)
                packed = _packed_hulls(stored)
                if packed is not None:
                    case = _evaluate_packed_hulls(
                        path,
                        root,
                        packed[0],
                        packed[1],
                        settings,
                    )
                else:
                    candidate = _trajectory_candidate(stored)
                    if candidate is None:
                        record["status"] = "unsupported_npz_schema"
                        inspected.append(record)
                        continue
                    case = _evaluate_fixed_trajectory(
                        path,
                        root,
                        stored,
                        candidate[0],
                        candidate[1],
                        settings,
                    )
        except (OSError, ValueError, TypeError) as error:
            record["status"] = "unreadable_or_invalid"
            record["error"] = f"{type(error).__name__}: {error}"
            inspected.append(record)
            continue
        if case["step_count"] == 0:
            record["status"] = "insufficient_temporal_support"
            inspected.append(record)
            continue
        record["status"] = "evaluated"
        inspected.append(record)
        cases.append(case)

    result: dict[str, Any] = {
        "schema": DEFORM360_PUBLIC_EVALUATION_SCHEMA,
        "revision": revision,
        "data_root": str(root),
        "limits": {
            "max_archives": settings.max_archives,
            "max_frames_per_archive": settings.max_frames_per_archive,
            "max_tracks": settings.max_tracks,
        },
        "information_boundary": {
            "dataset_read_only": True,
            "rolling_prefix_only": True,
            "future_used_for_scoring_only": True,
            "method_parameters_refit": False,
            "official_deform360_table_parity": False,
        },
        "inventory": {
            "npz_paths_considered": len(paths),
            "npz_paths_inspected": len(inspected),
            "archives_evaluated": len(cases),
            "inspection": inspected,
        },
        "summary": _aggregate(cases),
        "cases": cases,
    }
    result["result_sha256"] = _result_sha256(result)
    return result


def write_evaluation(path: Path, result: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
