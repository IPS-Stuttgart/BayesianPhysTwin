#!/usr/bin/env python3
"""Read-only rolling evaluation on public Deform360 trajectory artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.endpoint_model_average import (
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

DEFORM360_PUBLIC_EVALUATION_SCHEMA = "bayesian-phystwin/deform360-public-v1"
CHI_SQUARE_3D_90 = 6.251388631170325
_OBJECT_PATTERN = re.compile(r"^\d{3}-.+")
_TRAJECTORY_HINTS = (
    "positions_world_m",
    "points_world_m",
    "positions_m",
    "control_points",
    "particle_tracks",
    "particles",
    "tracks",
    "trajectory",
    "positions",
    "points",
)
_VALID_HINTS = ("valid_mask", "track_valid", "visibility", "valid")
_PATH_HINTS = ("hull", "control", "track", "particle", "trajectory", "position")


@dataclass(frozen=True, slots=True)
class EvaluationLimits:
    """Deterministic resource bounds for one evaluation run."""

    max_archives: int = 64
    max_frames_per_archive: int = 96
    max_tracks: int = 2048

    def __post_init__(self) -> None:
        for name in ("max_archives", "max_frames_per_archive", "max_tracks"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _result_sha256(value: dict[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    finite = np.asarray(tuple(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if len(finite) == 0 else float(np.mean(finite))


def _indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _object_id(path: Path, root: Path) -> str:
    return next(
        (
            part
            for part in path.relative_to(root).parts
            if _OBJECT_PATTERN.match(part)
        ),
        "unknown",
    )


def _scale_to_meters(key: str, points: np.ndarray) -> tuple[float, str]:
    lowered = key.lower()
    if lowered.endswith("_mm") or "millimet" in lowered:
        return 1e-3, "declared_mm"
    if lowered.endswith("_m") or "world_m" in lowered:
        return 1.0, "declared_m"
    norms = np.linalg.norm(points.reshape(-1, 3), axis=1)
    if float(np.nanmedian(norms)) > 20.0:
        return 1e-3, "heuristic_mm"
    return 1.0, "heuristic_m"


def _chamfer_rmse(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left = left[_indices(len(left), 512)]
    right = right[_indices(len(right), 512)]
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Chamfer inputs must be nonempty")

    def directed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        minimum = np.full(len(source), np.inf, dtype=np.float64)
        for start in range(0, len(target), 128):
            block = target[start : start + 128]
            squared = np.sum(
                np.square(source[:, None, :] - block[None, :, :]),
                axis=2,
            )
            minimum = np.minimum(minimum, np.min(squared, axis=1))
        return minimum

    mean_squared = 0.5 * (
        float(np.mean(directed(left, right)))
        + float(np.mean(directed(right, left)))
    )
    return float(np.sqrt(max(mean_squared, 0.0)))


def _rmse(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    error = prediction[valid] - target[valid]
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def _coverage_nll(
    error: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float]:
    covariance = covariance + np.eye(3, dtype=np.float64)[None] * 1e-12
    sign, logdet = np.linalg.slogdet(covariance)
    usable = sign > 0
    if not np.any(usable):  # pragma: no cover - contract covariance is PSD
        raise ValueError("predictive covariance is not positive definite")
    selected_error = error[usable]
    selected_covariance = covariance[usable]
    solved = np.linalg.solve(
        selected_covariance,
        selected_error[..., None],
    )[..., 0]
    quadratic = np.sum(selected_error * solved, axis=1)
    nll = 0.5 * (3.0 * np.log(2.0 * np.pi) + logdet[usable] + quadratic)
    return (
        float(np.mean(quadratic <= CHI_SQUARE_3D_90)),
        float(np.mean(nll)),
    )


def _effective_components(weights: np.ndarray) -> float:
    entropy = -np.sum(
        weights * np.log(np.maximum(weights, 1e-300)),
        axis=1,
    )
    return float(np.mean(np.exp(entropy)))


def _discover_npz(root: Path, limit: int) -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory, names, files in os.walk(root):
        names[:] = sorted(
            name
            for name in names
            if name not in {".git", "__pycache__", "node_modules"}
        )
        paths.extend(
            Path(directory) / name
            for name in sorted(files)
            if name.lower().endswith(".npz")
        )
    paths.sort(
        key=lambda path: (
            0 if any(hint in path.name.lower() for hint in _PATH_HINTS) else 1,
            str(path.relative_to(root)),
        )
    )
    return tuple(paths[:limit])


def _trajectory(stored: Any) -> tuple[str, np.ndarray] | None:
    candidates: list[tuple[int, str, np.ndarray]] = []
    for key in stored.files:
        lowered = key.lower()
        ranks = [
            index
            for index, hint in enumerate(_TRAJECTORY_HINTS)
            if hint in lowered
        ]
        if not ranks:
            continue
        value = np.asarray(stored[key])
        if value.ndim == 3 and value.shape[-1] == 3 and value.shape[0] >= 4:
            candidates.append((min(ranks), key, value))
    if not candidates:
        return None
    _, key, value = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return key, value


def _validity(stored: Any, shape: tuple[int, int]) -> np.ndarray:
    for key in stored.files:
        if not any(hint in key.lower() for hint in _VALID_HINTS):
            continue
        value = np.asarray(stored[key])
        if value.shape == shape:
            return np.asarray(value, dtype=bool)
    return np.ones(shape, dtype=bool)


def _packed_hulls(stored: Any) -> tuple[np.ndarray, tuple[np.ndarray, ...]] | None:
    keys = set(stored.files)
    if not {"frame_indices", "point_offsets", "points_world_m"}.issubset(keys):
        return None
    frames = np.asarray(stored["frame_indices"], dtype=np.int64)
    offsets = np.asarray(stored["point_offsets"], dtype=np.int64)
    points = np.asarray(stored["points_world_m"], dtype=np.float64)
    valid = (
        frames.ndim == 1
        and len(frames) >= 4
        and offsets.shape == (len(frames) + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(points)
        and points.ndim == 2
        and points.shape[1] == 3
        and np.all(np.isfinite(points))
    )
    if not valid:
        return None
    hulls = tuple(
        points[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(frames))
    )
    return None if any(len(hull) == 0 for hull in hulls) else (frames, hulls)


def _rolling_prediction(
    residual: np.ndarray,
    valid: np.ndarray,
    current: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    posterior = infer_model_averaged_endpoint(
        residual[:current],
        valid[:current],
        end_frame=current,
    )
    prediction = predict_model_averaged_endpoint(posterior, horizon_steps=1)
    return (
        prediction.mean_m,
        prediction.covariance_m2,
        _effective_components(prediction.component_weights),
    )


def _evaluate_fixed(
    path: Path,
    root: Path,
    stored: Any,
    key: str,
    raw: np.ndarray,
    limits: EvaluationLimits,
) -> dict[str, Any] | None:
    scale, unit_source = _scale_to_meters(key, raw)
    points = np.asarray(raw, dtype=np.float64) * scale
    validity = _validity(stored, points.shape[:2]) & np.all(
        np.isfinite(points), axis=2
    )
    track_indices = _indices(len(points[0]), limits.max_tracks)
    points = points[: limits.max_frames_per_archive, track_indices]
    validity = validity[: len(points), track_indices]
    if len(points) < 4:
        return None
    residual = np.diff(points, axis=0)
    residual_valid = validity[:-1] & validity[1:]
    steps: list[dict[str, Any]] = []
    for current in range(2, len(points) - 1):
        target_valid = validity[current] & validity[current + 1]
        if not np.any(target_valid):
            continue
        mean, covariance, effective = _rolling_prediction(
            residual,
            residual_valid,
            current,
        )
        target = points[current + 1]
        methods = {
            "persistence": points[current],
            "last_residual": points[current] + residual[current - 1],
            "model_average": points[current] + mean,
        }
        coverage, nll = _coverage_nll(
            target[target_valid] - methods["model_average"][target_valid],
            covariance[target_valid],
        )
        steps.append(
            {
                "identity_rmse_m": {
                    name: _rmse(value, target, target_valid)
                    for name, value in methods.items()
                },
                "chamfer_rmse_m": {
                    name: _chamfer_rmse(value[target_valid], target[target_valid])
                    for name, value in methods.items()
                },
                "raw_coverage_90": coverage,
                "mean_nll": nll,
                "effective_component_count": effective,
            }
        )
    if not steps:
        return None
    return {
        "path": str(path.relative_to(root)),
        "object_id": _object_id(path, root),
        "representation": "fixed_identity_trajectory",
        "array_key": key,
        "unit_source": unit_source,
        "frame_count": len(points),
        "track_count": points.shape[1],
        "steps": steps,
    }


def _evaluate_hulls(
    path: Path,
    root: Path,
    frames: np.ndarray,
    hulls: tuple[np.ndarray, ...],
    limits: EvaluationLimits,
) -> dict[str, Any] | None:
    count = min(len(hulls), limits.max_frames_per_archive)
    selected = tuple(
        hull[_indices(len(hull), limits.max_tracks)] for hull in hulls[:count]
    )
    centroids = np.asarray([np.mean(hull, axis=0) for hull in selected])
    residual = np.diff(centroids, axis=0)[:, None, :]
    valid = np.ones(residual.shape[:2], dtype=bool)
    steps: list[dict[str, Any]] = []
    for current in range(2, count - 1):
        mean, covariance, effective = _rolling_prediction(residual, valid, current)
        translations = {
            "persistence": np.zeros(3),
            "last_residual": residual[current - 1, 0],
            "model_average": mean[0],
        }
        target = selected[current + 1]
        target_translation = centroids[current + 1] - centroids[current]
        coverage, nll = _coverage_nll(
            (target_translation - mean[0])[None],
            covariance,
        )
        steps.append(
            {
                "centroid_error_m": {
                    name: float(np.linalg.norm(value - target_translation))
                    for name, value in translations.items()
                },
                "chamfer_rmse_m": {
                    name: _chamfer_rmse(selected[current] + value, target)
                    for name, value in translations.items()
                },
                "raw_coverage_90": coverage,
                "mean_nll": nll,
                "effective_component_count": effective,
            }
        )
    if not steps:
        return None
    return {
        "path": str(path.relative_to(root)),
        "object_id": _object_id(path, root),
        "representation": "packed_visual_hulls",
        "array_key": "points_world_m",
        "unit_source": "declared_m",
        "frame_count": count,
        "track_count": None,
        "steps": steps,
        "frame_indices": frames[:count].astype(int).tolist(),
    }


def _case_means(case: dict[str, Any], metric: str, method: str) -> float | None:
    return _mean(
        float(step[metric][method])
        for step in case["steps"]
        if metric in step
    )


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    methods = ("persistence", "last_residual", "model_average")
    for representation in sorted({case["representation"] for case in cases}):
        selected = [
            case for case in cases if case["representation"] == representation
        ]
        summary: dict[str, Any] = {
            "archive_count": len(selected),
            "object_count": len({case["object_id"] for case in selected}),
            "step_count": sum(len(case["steps"]) for case in selected),
        }
        for metric in (
            "identity_rmse_m",
            "centroid_error_m",
            "chamfer_rmse_m",
        ):
            values = {
                method: _mean(
                    value
                    for case in selected
                    if (value := _case_means(case, metric, method)) is not None
                )
                for method in methods
            }
            if any(value is not None for value in values.values()):
                summary[metric] = values
        summary["raw_coverage_90"] = _mean(
            float(step["raw_coverage_90"])
            for case in selected
            for step in case["steps"]
        )
        summary["effective_component_count"] = _mean(
            float(step["effective_component_count"])
            for case in selected
            for step in case["steps"]
        )
        result[representation] = summary
    return result


def evaluate_deform360_public_data(
    data_root: Path,
    *,
    limits: EvaluationLimits | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Run the deterministic rolling diagnostic on supported NPZ artifacts."""

    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Deform360 data root is missing: {root}")
    settings = EvaluationLimits() if limits is None else limits
    paths = _discover_npz(root, settings.max_archives * 16)
    cases: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    for path in paths:
        if len(cases) >= settings.max_archives:
            break
        inspection: dict[str, Any] = {"path": str(path.relative_to(root))}
        try:
            with np.load(path, allow_pickle=False) as stored:
                inspection["keys"] = sorted(stored.files)
                packed = _packed_hulls(stored)
                if packed is not None:
                    case = _evaluate_hulls(
                        path,
                        root,
                        packed[0],
                        packed[1],
                        settings,
                    )
                else:
                    candidate = _trajectory(stored)
                    case = (
                        None
                        if candidate is None
                        else _evaluate_fixed(
                            path,
                            root,
                            stored,
                            candidate[0],
                            candidate[1],
                            settings,
                        )
                    )
        except (OSError, TypeError, ValueError) as error:
            inspection["status"] = "invalid"
            inspection["error"] = f"{type(error).__name__}: {error}"
            inspected.append(inspection)
            continue
        inspection["status"] = "evaluated" if case is not None else "unsupported"
        inspected.append(inspection)
        if case is not None:
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
            "npz_paths_discovered": len(paths),
            "archives_evaluated": len(cases),
            "inspection": inspected,
        },
        "summary": _aggregate(cases),
        "cases": cases,
    }
    result["result_sha256"] = _result_sha256(result)
    return result


def write_evaluation(path: Path, result: dict[str, Any]) -> None:
    """Write one canonical, newline-terminated JSON evaluation artifact."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-archives", type=int, default=64)
    parser.add_argument("--max-frames-per-archive", type=int, default=96)
    parser.add_argument("--max-tracks", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_deform360_public_data(
        args.data_root,
        limits=EvaluationLimits(
            max_archives=args.max_archives,
            max_frames_per_archive=args.max_frames_per_archive,
            max_tracks=args.max_tracks,
        ),
        revision=os.environ.get("GITHUB_SHA"),
    )
    write_evaluation(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if result["inventory"]["archives_evaluated"] == 0:
        raise SystemExit(
            "no supported Deform360 trajectory archive was found; "
            "inspect the uploaded inventory"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
