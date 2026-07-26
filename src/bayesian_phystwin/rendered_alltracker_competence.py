"""Scoring for the sealed render-to-real AllTracker source competence control."""

from __future__ import annotations

from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rmse(vectors: np.ndarray) -> float | None:
    values = np.asarray(vectors, dtype=np.float64)
    if not len(values):
        return None
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=-1))))


def validate_cotracker_prefix_quality_shape(
    quality_probability: np.ndarray,
    *,
    full_track_shape: tuple[int, int],
    scored_frames: np.ndarray,
) -> None:
    """Accept a prefix-only quality tensor for a longer full trajectory."""

    quality = np.asarray(quality_probability)
    frames = np.asarray(scored_frames, dtype=np.int64)
    _require(
        quality.ndim == 3
        and quality.shape[2] == full_track_shape[1],
        "CoTracker3 quality must have shape (camera, prefix_frame, track)",
    )
    _require(
        frames.ndim == 1
        and len(frames) > 0
        and np.all(frames >= 0)
        and int(np.max(frames)) < quality.shape[1],
        "CoTracker3 quality does not cover every scored prefix frame",
    )


def trajectory_metrics(
    prediction_world_m: np.ndarray,
    valid: np.ndarray,
    target_world_m: np.ndarray,
) -> dict[str, Any]:
    """Score one frame-identity panel on its exact finite support."""

    prediction = np.asarray(prediction_world_m, dtype=np.float64)
    supplied_valid = np.asarray(valid, dtype=bool)
    target = np.asarray(target_world_m, dtype=np.float64)
    _require(
        prediction.ndim == 3
        and prediction.shape[2] == 3
        and prediction.shape == target.shape,
        "prediction and target must share shape (frame, identity, 3)",
    )
    _require(
        supplied_valid.shape == prediction.shape[:2],
        "validity shape changed",
    )
    finite_target = np.all(np.isfinite(target), axis=2)
    finite_prediction = np.all(np.isfinite(prediction), axis=2)
    support = supplied_valid & finite_target & finite_prediction
    errors = prediction - target
    return {
        "supported_count": int(np.sum(support)),
        "target_count": int(np.sum(finite_target)),
        "support_fraction": (
            float(np.sum(support) / np.sum(finite_target))
            if np.any(finite_target)
            else None
        ),
        "position_rmse_m": _rmse(errors[support]),
        "per_frame": [
            {
                "frame_position": frame,
                "supported_count": int(np.sum(support[frame])),
                "target_count": int(np.sum(finite_target[frame])),
                "position_rmse_m": _rmse(errors[frame][support[frame]]),
            }
            for frame in range(len(prediction))
        ],
    }


def shared_support_metrics(
    candidate_world_m: np.ndarray,
    candidate_valid: np.ndarray,
    comparator_world_m: np.ndarray,
    comparator_valid: np.ndarray,
    target_world_m: np.ndarray,
) -> dict[str, Any]:
    """Compare candidate and comparator on identical material point-frames."""

    candidate = np.asarray(candidate_world_m, dtype=np.float64)
    comparator = np.asarray(comparator_world_m, dtype=np.float64)
    target = np.asarray(target_world_m, dtype=np.float64)
    first_valid = np.asarray(candidate_valid, dtype=bool)
    second_valid = np.asarray(comparator_valid, dtype=bool)
    _require(
        candidate.shape == comparator.shape == target.shape,
        "shared-support coordinate shapes changed",
    )
    _require(
        first_valid.shape == second_valid.shape == target.shape[:2],
        "shared-support validity shapes changed",
    )
    shared = (
        first_valid
        & second_valid
        & np.all(np.isfinite(candidate), axis=2)
        & np.all(np.isfinite(comparator), axis=2)
        & np.all(np.isfinite(target), axis=2)
    )
    candidate_rmse = _rmse((candidate - target)[shared])
    comparator_rmse = _rmse((comparator - target)[shared])
    improvement = None
    if (
        candidate_rmse is not None
        and comparator_rmse is not None
        and comparator_rmse > 0.0
    ):
        improvement = float(1.0 - candidate_rmse / comparator_rmse)
    return {
        "shared_count": int(np.sum(shared)),
        "candidate_rmse_m": candidate_rmse,
        "comparator_rmse_m": comparator_rmse,
        "candidate_relative_improvement_fraction": improvement,
    }


def covariance_diagnostics(
    prediction_world_m: np.ndarray,
    covariance_m2: np.ndarray,
    valid: np.ndarray,
    target_world_m: np.ndarray,
    *,
    chi_square_90_df3: float = 6.251388631170325,
) -> dict[str, float | int | None]:
    """Report conditional covariance behavior without claiming calibration."""

    prediction = np.asarray(prediction_world_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    target = np.asarray(target_world_m, dtype=np.float64)
    supported = (
        np.asarray(valid, dtype=bool)
        & np.all(np.isfinite(prediction), axis=2)
        & np.all(np.isfinite(target), axis=2)
    )
    _require(
        covariance.shape == (*prediction.shape[:2], 3, 3),
        "covariance shape changed",
    )
    if not np.any(supported):
        return {"count": 0, "mean_nees": None, "coverage_90": None}
    errors = (prediction - target)[supported]
    matrices = covariance[supported]
    nees = np.einsum(
        "ni,nij,nj->n",
        errors,
        np.linalg.inv(matrices),
        errors,
    )
    return {
        "count": int(len(nees)),
        "mean_nees": float(np.mean(nees)),
        "coverage_90": float(np.mean(nees <= chi_square_90_df3)),
    }


def evaluate_competence_gates(
    candidate_metrics: dict[str, Any],
    final_frame_metrics: dict[str, Any],
    physical_shared: dict[str, Any],
    cotracker_shared: dict[str, Any],
    *,
    minimum_support_fraction: float,
    maximum_position_rmse_m: float,
    maximum_final_frame_rmse_m: float,
    minimum_physical_improvement_fraction: float,
    minimum_cotracker_improvement_fraction: float,
) -> dict[str, bool]:
    """Apply the frozen conjunction of source competence requirements."""

    support = candidate_metrics["support_fraction"]
    rmse = candidate_metrics["position_rmse_m"]
    final_rmse = final_frame_metrics["position_rmse_m"]
    physical_gain = physical_shared[
        "candidate_relative_improvement_fraction"
    ]
    cotracker_gain = cotracker_shared[
        "candidate_relative_improvement_fraction"
    ]
    gates = {
        "support_at_least_minimum": (
            support is not None and support >= minimum_support_fraction
        ),
        "position_rmse_at_most_maximum": (
            rmse is not None and rmse <= maximum_position_rmse_m
        ),
        "final_frame_rmse_at_most_maximum": (
            final_rmse is not None
            and final_rmse <= maximum_final_frame_rmse_m
        ),
        "physical_improvement_at_least_minimum": (
            physical_gain is not None
            and physical_gain >= minimum_physical_improvement_fraction
        ),
        "cotracker_improvement_at_least_minimum": (
            cotracker_gain is not None
            and cotracker_gain >= minimum_cotracker_improvement_fraction
        ),
    }
    gates["competence_gate_passed"] = all(gates.values())
    return gates


__all__ = [
    "covariance_diagnostics",
    "evaluate_competence_gates",
    "shared_support_metrics",
    "trajectory_metrics",
    "validate_cotracker_prefix_quality_shape",
]
