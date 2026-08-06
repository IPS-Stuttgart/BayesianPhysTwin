"""Conservative multiview fusion for prefix-only TAPIP3D identities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_tapip3d_competence import (
    IdentityTrajectory,
    Tapip3dPrediction,
)


@dataclass(frozen=True)
class MultiviewTapip3dPrediction:
    """World-space identity observations with conservative uncertainty."""

    coords_world_m: np.ndarray
    valid: np.ndarray
    query_points: np.ndarray
    observation_covariance_m2: np.ndarray
    view_count: np.ndarray
    max_pairwise_disagreement_m: np.ndarray


def _validate_queries(query_points: np.ndarray) -> np.ndarray:
    queries = np.asarray(query_points, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 4 or len(queries) == 0:
        raise ValueError("query_points must have nonempty shape (N, 4)")
    if not np.all(np.isfinite(queries)):
        raise ValueError("query_points must be finite")
    if not np.all(queries[:, 0] == 0.0):
        raise ValueError("multiview competence permits frame-zero queries only")
    return queries


def _map_view_queries(
    view_queries: np.ndarray,
    global_queries: np.ndarray,
    *,
    tolerance_m: float,
) -> np.ndarray:
    local = _validate_queries(view_queries)
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be nonnegative")
    distances = np.linalg.norm(
        local[:, None, 1:] - global_queries[None, :, 1:],
        axis=2,
    )
    mapping = np.argmin(distances, axis=1).astype(np.int64)
    selected = distances[np.arange(len(local)), mapping]
    if np.any(selected > tolerance_m):
        raise ValueError("view query does not match a locked global identity")
    if len(np.unique(mapping)) != len(mapping):
        raise ValueError("view contains duplicate global query identities")
    return mapping


def _geometric_median(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return points[0].copy()
    if len(points) == 2:
        return np.mean(points, axis=0)
    estimate = np.median(points, axis=0)
    for _ in range(32):
        distances = np.linalg.norm(points - estimate[None, :], axis=1)
        coincident = distances <= 1e-12
        if np.any(coincident):
            return points[np.flatnonzero(coincident)[0]].copy()
        weights = 1.0 / distances
        updated = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(updated - estimate) <= 1e-12:
            return updated
        estimate = updated
    return estimate


def _maximum_pairwise_distance(points: np.ndarray) -> float:
    differences = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=2)))


def expand_tapip3d_view(
    prediction: Tapip3dPrediction,
    global_query_points: np.ndarray,
    *,
    query_tolerance_m: float = 1e-7,
) -> IdentityTrajectory:
    """Re-anchor one view and expand its query subset to locked identities."""

    queries = _validate_queries(global_query_points)
    coords = np.asarray(prediction.coords_world_m, dtype=np.float64)
    valid = np.asarray(prediction.valid, dtype=bool)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("view coordinates must have shape (T, N, 3)")
    if valid.shape != coords.shape[:2]:
        raise ValueError("view validity must have shape (T, N)")
    mapping = _map_view_queries(
        prediction.query_points,
        queries,
        tolerance_m=query_tolerance_m,
    )
    finite = np.all(np.isfinite(coords), axis=2)
    anchored_valid = valid & finite & (valid[0] & finite[0])[None, :]
    reanchored = (
        queries[mapping, 1:][None, :, :] + coords - coords[0][None, :, :]
    )
    expanded = np.full((len(coords), len(queries), 3), np.nan, dtype=np.float64)
    expanded_valid = np.zeros((len(coords), len(queries)), dtype=bool)
    expanded[:, mapping] = reanchored
    expanded_valid[:, mapping] = anchored_valid
    expanded[~expanded_valid] = np.nan
    return IdentityTrajectory(expanded, expanded_valid)


def fuse_tapip3d_views(
    predictions: Sequence[Tapip3dPrediction],
    global_query_points: np.ndarray,
    *,
    minimum_view_count: int = 2,
    maximum_pairwise_disagreement_m: float = 0.02,
    shared_bias_floor_m: float = 0.005,
    query_tolerance_m: float = 1e-7,
) -> MultiviewTapip3dPrediction:
    """Fuse re-anchored views without assuming independent camera errors."""

    if len(predictions) < minimum_view_count:
        raise ValueError("fewer predictions than minimum_view_count")
    if minimum_view_count < 2:
        raise ValueError("minimum_view_count must be at least two")
    if maximum_pairwise_disagreement_m <= 0.0:
        raise ValueError("maximum_pairwise_disagreement_m must be positive")
    if shared_bias_floor_m <= 0.0:
        raise ValueError("shared_bias_floor_m must be positive")
    queries = _validate_queries(global_query_points)
    frame_counts = {prediction.coords_world_m.shape[0] for prediction in predictions}
    if len(frame_counts) != 1:
        raise ValueError("all views must contain the same frame count")
    frame_count = frame_counts.pop()
    query_count = len(queries)
    view_coords = np.full(
        (len(predictions), frame_count, query_count, 3),
        np.nan,
        dtype=np.float64,
    )
    view_valid = np.zeros(
        (len(predictions), frame_count, query_count),
        dtype=bool,
    )

    for view_index, prediction in enumerate(predictions):
        expanded = expand_tapip3d_view(
            prediction,
            queries,
            query_tolerance_m=query_tolerance_m,
        )
        if expanded.coords_world_m.shape[0] != frame_count:
            raise ValueError("all views must contain the same frame count")
        view_coords[view_index] = expanded.coords_world_m
        view_valid[view_index] = expanded.valid

    coords = np.full((frame_count, query_count, 3), np.nan, dtype=np.float64)
    valid = np.zeros((frame_count, query_count), dtype=bool)
    covariance = np.full(
        (frame_count, query_count, 3, 3),
        np.nan,
        dtype=np.float64,
    )
    view_count = np.sum(view_valid, axis=0, dtype=np.int16)
    disagreement = np.full((frame_count, query_count), np.nan, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    floor_variance = shared_bias_floor_m**2

    for frame_index in range(frame_count):
        for query_index in range(query_count):
            selected = view_coords[
                view_valid[:, frame_index, query_index],
                frame_index,
                query_index,
            ]
            if len(selected) < minimum_view_count:
                continue
            pairwise = _maximum_pairwise_distance(selected)
            disagreement[frame_index, query_index] = pairwise
            if pairwise > maximum_pairwise_disagreement_m:
                continue
            center = _geometric_median(selected)
            radius = float(np.max(np.linalg.norm(selected - center, axis=1)))
            # Unknown cross-view correlation forbids 1 / view_count shrinkage.
            variance = floor_variance + radius**2
            coords[frame_index, query_index] = center
            covariance[frame_index, query_index] = identity * variance
            valid[frame_index, query_index] = True

    return MultiviewTapip3dPrediction(
        coords_world_m=coords,
        valid=valid,
        query_points=queries,
        observation_covariance_m2=covariance,
        view_count=view_count,
        max_pairwise_disagreement_m=disagreement,
    )


def save_multiview_tapip3d_prediction(
    path: str | Path,
    prediction: MultiviewTapip3dPrediction,
) -> None:
    """Write the compact score-blind multiview carrier."""

    np.savez_compressed(
        path,
        coords_world_m=prediction.coords_world_m,
        valid=prediction.valid,
        query_points=prediction.query_points,
        observation_covariance_m2=prediction.observation_covariance_m2,
        view_count=prediction.view_count,
        max_pairwise_disagreement_m=prediction.max_pairwise_disagreement_m,
    )


def load_multiview_tapip3d_prediction(
    path: str | Path,
) -> MultiviewTapip3dPrediction:
    """Load and validate a compact multiview carrier."""

    required = {
        "coords_world_m",
        "valid",
        "query_points",
        "observation_covariance_m2",
        "view_count",
        "max_pairwise_disagreement_m",
    }
    with np.load(path) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                "multiview TAPIP3D carrier lacks required fields: "
                + ", ".join(sorted(missing))
            )
        result = MultiviewTapip3dPrediction(
            coords_world_m=np.asarray(archive["coords_world_m"], dtype=np.float64),
            valid=np.asarray(archive["valid"]),
            query_points=np.asarray(archive["query_points"], dtype=np.float64),
            observation_covariance_m2=np.asarray(
                archive["observation_covariance_m2"], dtype=np.float64
            ),
            view_count=np.asarray(archive["view_count"]),
            max_pairwise_disagreement_m=np.asarray(
                archive["max_pairwise_disagreement_m"], dtype=np.float64
            ),
        )
    _validate_multiview_prediction(result)
    return result


def _validate_multiview_prediction(
    prediction: MultiviewTapip3dPrediction,
) -> None:
    coords = prediction.coords_world_m
    valid = prediction.valid
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("multiview coords must have shape (T, N, 3)")
    if valid.dtype != np.bool_ or valid.shape != coords.shape[:2]:
        raise ValueError("multiview valid must be boolean with shape (T, N)")
    queries = _validate_queries(prediction.query_points)
    if len(queries) != coords.shape[1]:
        raise ValueError("multiview query count is inconsistent")
    expected_covariance = (*coords.shape[:2], 3, 3)
    if prediction.observation_covariance_m2.shape != expected_covariance:
        raise ValueError("multiview covariance must have shape (T, N, 3, 3)")
    if prediction.view_count.shape != coords.shape[:2]:
        raise ValueError("multiview view_count must have shape (T, N)")
    if prediction.max_pairwise_disagreement_m.shape != coords.shape[:2]:
        raise ValueError("multiview disagreement must have shape (T, N)")
    if not np.all(np.isfinite(coords[valid])):
        raise ValueError("valid multiview coordinates must be finite")
    selected_covariance = prediction.observation_covariance_m2[valid]
    if not np.all(np.isfinite(selected_covariance)):
        raise ValueError("valid multiview covariance must be finite")
    if not np.allclose(
        selected_covariance,
        np.swapaxes(selected_covariance, -1, -2),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("multiview covariance must be symmetric")
    if len(selected_covariance) and np.min(
        np.linalg.eigvalsh(selected_covariance)
    ) <= 0.0:
        raise ValueError("multiview covariance must be positive definite")


def multiview_identity_trajectory(
    prediction: MultiviewTapip3dPrediction,
) -> IdentityTrajectory:
    """Expose supported observations without counting fallback as evidence."""

    _validate_multiview_prediction(prediction)
    return IdentityTrajectory(
        coords_world_m=prediction.coords_world_m,
        valid=prediction.valid,
    )


def apply_exact_identity_fallback(
    prediction: MultiviewTapip3dPrediction,
    baseline_coords_world_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the baseline bit-for-bit wherever the provider is unsupported."""

    _validate_multiview_prediction(prediction)
    baseline = np.asarray(baseline_coords_world_m)
    if baseline.shape != prediction.coords_world_m.shape:
        raise ValueError("baseline coordinates must match the provider shape")
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline coordinates must be finite")
    output = baseline.copy()
    np.copyto(
        output,
        prediction.coords_world_m.astype(output.dtype, copy=False),
        where=prediction.valid[..., None],
    )
    return output, prediction.valid.astype(np.float64)


def multiview_support_diagnostics(
    prediction: MultiviewTapip3dPrediction,
) -> dict[str, Any]:
    """Summarize target-free support and disagreement."""

    _validate_multiview_prediction(prediction)
    finite_disagreement = prediction.max_pairwise_disagreement_m[
        np.isfinite(prediction.max_pairwise_disagreement_m)
    ]
    return {
        "supported_count": int(np.sum(prediction.valid)),
        "total_count": int(prediction.valid.size),
        "support_fraction": float(np.mean(prediction.valid)),
        "view_count_histogram": {
            str(count): int(np.sum(prediction.view_count == count))
            for count in np.unique(prediction.view_count)
        },
        "median_pairwise_disagreement_m": (
            float(np.median(finite_disagreement))
            if len(finite_disagreement)
            else None
        ),
        "p90_pairwise_disagreement_m": (
            float(np.quantile(finite_disagreement, 0.9))
            if len(finite_disagreement)
            else None
        ),
    }


def evaluate_multiview_tapip3d_gates(
    metrics: dict[str, Any],
    late_metrics: dict[str, Any],
    best_single_shared_improvement_fraction: float | None,
    *,
    minimum_support_fraction: float,
    maximum_displacement_rmse_m: float,
    maximum_frame_zero_anchor_rmse_m: float,
    minimum_late_support_fraction: float,
    maximum_late_displacement_rmse_m: float,
    minimum_best_single_shared_improvement_fraction: float,
) -> dict[str, bool]:
    """Apply the frozen association-oracle multiview competence gate."""

    gates = {
        "prefix_support_at_least_minimum": (
            metrics["support_fraction"] is not None
            and metrics["support_fraction"] >= minimum_support_fraction
        ),
        "displacement_rmse_at_most_maximum": (
            metrics["displacement_rmse_m"] is not None
            and metrics["displacement_rmse_m"] <= maximum_displacement_rmse_m
        ),
        "frame_zero_anchor_rmse_at_most_maximum": (
            metrics["frame_zero_anchor_rmse_m"] is not None
            and metrics["frame_zero_anchor_rmse_m"]
            <= maximum_frame_zero_anchor_rmse_m
        ),
        "late_support_at_least_minimum": (
            late_metrics["support_fraction"] is not None
            and late_metrics["support_fraction"] >= minimum_late_support_fraction
        ),
        "late_displacement_rmse_at_most_maximum": (
            late_metrics["displacement_rmse_m"] is not None
            and late_metrics["displacement_rmse_m"]
            <= maximum_late_displacement_rmse_m
        ),
        "best_single_shared_improvement_at_least_minimum": (
            best_single_shared_improvement_fraction is not None
            and best_single_shared_improvement_fraction
            >= minimum_best_single_shared_improvement_fraction
        ),
    }
    gates["competence_gate_passed"] = all(gates.values())
    return gates


__all__ = [
    "MultiviewTapip3dPrediction",
    "apply_exact_identity_fallback",
    "evaluate_multiview_tapip3d_gates",
    "expand_tapip3d_view",
    "fuse_tapip3d_views",
    "load_multiview_tapip3d_prediction",
    "multiview_identity_trajectory",
    "multiview_support_diagnostics",
    "save_multiview_tapip3d_prediction",
]
