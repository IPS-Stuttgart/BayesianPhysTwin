"""Prefix-only TAPIP3D competence diagnostics for PhysTwin identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Tapip3dPrediction:
    """Canonical world-space TAPIP3D prediction."""

    coords_world_m: np.ndarray
    valid: np.ndarray
    query_points: np.ndarray


@dataclass(frozen=True)
class IdentityTrajectory:
    """World-space trajectories attached to fixed frame-zero identities."""

    coords_world_m: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class FrameZeroAssociation:
    """Nearest graph identities selected without later trajectory evidence."""

    node_indices: np.ndarray
    distance_m: np.ndarray


def _validate_query_points(query_points: np.ndarray) -> np.ndarray:
    queries = np.asarray(query_points)
    if queries.ndim != 2 or queries.shape[1] != 4 or len(queries) == 0:
        raise ValueError("query_points must have nonempty shape (N, 4)")
    if not np.all(np.isfinite(queries)):
        raise ValueError("query_points must be finite")
    return queries


def load_tapip3d_prediction(path: str | Path) -> Tapip3dPrediction:
    """Load and validate the minimal fields emitted by official TAPIP3D."""

    required = {"coords", "visibs", "query_points"}
    with np.load(path) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"TAPIP3D result lacks required fields: {names}")
        coords = np.asarray(archive["coords"])
        valid = np.asarray(archive["visibs"])
        query_points = np.asarray(archive["query_points"])

    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (T, N, 3)")
    if valid.shape != coords.shape[:2]:
        raise ValueError("visibs must have shape (T, N)")
    if valid.dtype != np.bool_:
        raise ValueError("visibs must contain thresholded boolean visibility")
    queries = _validate_query_points(query_points)
    if len(queries) != coords.shape[1]:
        raise ValueError("query count differs between coords and query_points")
    if coords.shape[0] == 0:
        raise ValueError("TAPIP3D prediction must contain at least one frame")

    finite = np.all(np.isfinite(coords), axis=2)
    return Tapip3dPrediction(
        coords_world_m=np.asarray(coords, dtype=np.float64),
        valid=np.asarray(valid & finite, dtype=bool),
        query_points=np.asarray(queries, dtype=np.float64),
    )


def validate_tapip3d_prediction_contract(
    prediction: Tapip3dPrediction,
    expected_query_points: np.ndarray,
    *,
    expected_frame_count: int,
    query_tolerance_m: float = 1e-7,
) -> None:
    """Bind a model result to the prefix input without opening score targets."""

    if expected_frame_count <= 0:
        raise ValueError("expected_frame_count must be positive")
    expected = _validate_query_points(expected_query_points).astype(float)
    if prediction.coords_world_m.shape[0] != expected_frame_count:
        raise ValueError("TAPIP3D frame count differs from the locked prefix")
    if prediction.query_points.shape != expected.shape:
        raise ValueError("TAPIP3D query shape differs from the locked input")
    if not np.allclose(
        prediction.query_points,
        expected,
        rtol=0.0,
        atol=query_tolerance_m,
    ):
        raise ValueError("TAPIP3D queries differ from the locked input")
    query_frames = prediction.query_points[:, 0]
    if not np.all(query_frames == 0.0):
        raise ValueError("competence-v1 permits frame-zero queries only")


def save_canonical_tapip3d_prediction(
    path: str | Path,
    prediction: Tapip3dPrediction,
) -> None:
    """Save a compact carrier that excludes copied RGB-D inputs."""

    np.savez_compressed(
        path,
        coords_world_m=prediction.coords_world_m,
        valid=prediction.valid,
        query_points=prediction.query_points,
    )


def load_canonical_tapip3d_prediction(
    path: str | Path,
) -> Tapip3dPrediction:
    """Load a sealed compact prediction carrier."""

    required = {"coords_world_m", "valid", "query_points"}
    with np.load(path) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"canonical TAPIP3D carrier lacks required fields: {names}"
            )
        coords = np.asarray(archive["coords_world_m"])
        valid = np.asarray(archive["valid"])
        queries = np.asarray(archive["query_points"])
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("canonical coords_world_m must have shape (T, N, 3)")
    if valid.dtype != np.bool_ or valid.shape != coords.shape[:2]:
        raise ValueError("canonical valid must be boolean with shape (T, N)")
    query_points = _validate_query_points(queries)
    if len(query_points) != coords.shape[1]:
        raise ValueError("canonical query count is inconsistent")
    if not np.all(np.isfinite(coords[valid])):
        raise ValueError("canonical valid coordinates must be finite")
    return Tapip3dPrediction(
        coords_world_m=np.asarray(coords, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        query_points=np.asarray(query_points, dtype=np.float64),
    )


def associate_frame_zero_queries(
    initial_node_positions_m: np.ndarray,
    query_positions_m: np.ndarray,
) -> FrameZeroAssociation:
    """Fix nearest graph identities from frame-zero geometry only."""

    nodes = np.asarray(initial_node_positions_m, dtype=float)
    queries = np.asarray(query_positions_m, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
        raise ValueError(
            "initial_node_positions_m must have nonempty shape (M, 3)"
        )
    if queries.ndim != 2 or queries.shape[1] != 3 or len(queries) == 0:
        raise ValueError("query_positions_m must have nonempty shape (N, 3)")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(queries)):
        raise ValueError("frame-zero nodes and queries must be finite")
    distances = np.linalg.norm(
        nodes[:, None, :] - queries[None, :, :],
        axis=2,
    )
    node_indices = np.argmin(distances, axis=0).astype(np.int64)
    query_indices = np.arange(len(queries))
    return FrameZeroAssociation(
        node_indices=node_indices,
        distance_m=distances[node_indices, query_indices],
    )


def build_same_query_cotracker3_trajectory(
    multiview_points_world_m: np.ndarray,
    multiview_valid: np.ndarray,
    initial_node_positions_m: np.ndarray,
    query_positions_m: np.ndarray,
) -> tuple[IdentityTrajectory, FrameZeroAssociation]:
    """Anchor nearest-node CoTracker3 displacements to identical queries."""

    points = np.asarray(multiview_points_world_m, dtype=float)
    valid = np.asarray(multiview_valid, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("multiview_points_world_m must have shape (T, M, 3)")
    if valid.shape != points.shape[:2]:
        raise ValueError("multiview_valid must have shape (T, M)")
    association = associate_frame_zero_queries(
        initial_node_positions_m,
        query_positions_m,
    )
    if points.shape[1] != len(initial_node_positions_m):
        raise ValueError("CoTracker3 node count differs from frame-zero geometry")
    selected = points[:, association.node_indices]
    selected_valid = valid[:, association.node_indices]
    initial_valid = selected_valid[0] & np.all(
        np.isfinite(selected[0]),
        axis=1,
    )
    anchored = (
        np.asarray(query_positions_m, dtype=float)[None, :, :]
        + selected
        - selected[0][None, :, :]
    )
    trajectory_valid = (
        selected_valid
        & initial_valid[None, :]
        & np.all(np.isfinite(anchored), axis=2)
    )
    anchored[~trajectory_valid] = np.nan
    return (
        IdentityTrajectory(
            coords_world_m=anchored,
            valid=trajectory_valid,
        ),
        association,
    )


def _rmse_from_vectors(vectors: np.ndarray) -> float | None:
    if len(vectors) == 0:
        return None
    squared_distance = np.sum(np.square(vectors), axis=1)
    return float(np.sqrt(np.mean(squared_distance)))


def _translation_diagnostics(
    displacement_error: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int | None]:
    bias_vectors: list[np.ndarray] = []
    centered_vectors: list[np.ndarray] = []
    for frame_error, frame_valid in zip(displacement_error, valid, strict=True):
        values = frame_error[frame_valid]
        if len(values) == 0:
            continue
        bias = np.mean(values, axis=0)
        bias_vectors.append(bias)
        centered_vectors.extend(values - bias)
    bias_array = np.asarray(bias_vectors, dtype=float).reshape(-1, 3)
    centered_array = np.asarray(centered_vectors, dtype=float).reshape(-1, 3)
    return {
        "frame_count": len(bias_vectors),
        "common_translation_rmse_m": _rmse_from_vectors(bias_array),
        "translation_removed_rmse_m": _rmse_from_vectors(centered_array),
    }


def identity_trajectory_metrics(
    trajectory: IdentityTrajectory,
    ground_truth_world_m: np.ndarray,
    *,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> dict[str, Any]:
    """Evaluate support, position, and material-displacement errors."""

    coords = np.asarray(trajectory.coords_world_m, dtype=float)
    supplied_valid = np.asarray(trajectory.valid, dtype=bool)
    target = np.asarray(ground_truth_world_m, dtype=float)
    if coords.shape != target.shape or coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("trajectory and ground truth must share shape (T, N, 3)")
    if supplied_valid.shape != coords.shape[:2]:
        raise ValueError("trajectory validity must have shape (T, N)")
    end = len(coords) if frame_end is None else frame_end
    if not 0 <= frame_start < end <= len(coords):
        raise ValueError("invalid metric frame range")
    if not np.all(np.isfinite(target[0])):
        raise ValueError("all frame-zero identity targets must be finite")

    finite_target = np.all(np.isfinite(target), axis=2)
    finite_coords = np.all(np.isfinite(coords), axis=2)
    initial_available = supplied_valid[0] & finite_coords[0]
    valid = (
        supplied_valid
        & finite_coords
        & finite_target
        & initial_available[None, :]
    )
    frame_slice = slice(frame_start, end)
    range_valid = valid[frame_slice]
    target_available = finite_target[frame_slice]
    supported_count = int(np.sum(range_valid))
    target_count = int(np.sum(target_available))

    position_error = coords - target
    predicted_displacement = coords - coords[0][None, :, :]
    target_displacement = target - target[0][None, :, :]
    displacement_error = predicted_displacement - target_displacement
    selected_position = position_error[frame_slice][range_valid]
    selected_displacement = displacement_error[frame_slice][range_valid]
    anchor_valid = valid[0]
    anchor_error = position_error[0, anchor_valid]

    per_identity: list[dict[str, Any]] = []
    for identity in range(coords.shape[1]):
        identity_valid = range_valid[:, identity]
        per_identity.append(
            {
                "identity_index": identity,
                "supported_count": int(np.sum(identity_valid)),
                "target_count": int(np.sum(target_available[:, identity])),
                "support_fraction": (
                    float(np.mean(identity_valid[target_available[:, identity]]))
                    if np.any(target_available[:, identity])
                    else None
                ),
                "displacement_rmse_m": _rmse_from_vectors(
                    displacement_error[frame_slice, identity][identity_valid]
                ),
            }
        )

    return {
        "frame_range": [frame_start, end],
        "supported_count": supported_count,
        "target_count": target_count,
        "support_fraction": (
            float(supported_count / target_count) if target_count else None
        ),
        "position_rmse_m": _rmse_from_vectors(selected_position),
        "displacement_rmse_m": _rmse_from_vectors(selected_displacement),
        "frame_zero_anchor_rmse_m": _rmse_from_vectors(anchor_error),
        "translation_diagnostics": _translation_diagnostics(
            displacement_error[frame_slice],
            range_valid,
        ),
        "per_identity": per_identity,
    }


def shared_support_displacement_metrics(
    first: IdentityTrajectory,
    second: IdentityTrajectory,
    ground_truth_world_m: np.ndarray,
    *,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> dict[str, Any]:
    """Compare two trackers only where both supply the same identities."""

    first_coords = np.asarray(first.coords_world_m, dtype=float)
    second_coords = np.asarray(second.coords_world_m, dtype=float)
    target = np.asarray(ground_truth_world_m, dtype=float)
    if first_coords.shape != second_coords.shape or first_coords.shape != target.shape:
        raise ValueError("shared-support trajectories must have identical shapes")
    end = len(target) if frame_end is None else frame_end
    if not 0 <= frame_start < end <= len(target):
        raise ValueError("invalid shared-support frame range")
    finite_target = np.all(np.isfinite(target), axis=2)
    shared = (
        np.asarray(first.valid, dtype=bool)
        & np.asarray(second.valid, dtype=bool)
        & np.all(np.isfinite(first_coords), axis=2)
        & np.all(np.isfinite(second_coords), axis=2)
        & finite_target
    )
    shared &= shared[0][None, :]
    frame_slice = slice(frame_start, end)
    selected = shared[frame_slice]
    target_displacement = target - target[0][None, :, :]
    first_error = (
        first_coords - first_coords[0][None, :, :] - target_displacement
    )
    second_error = (
        second_coords - second_coords[0][None, :, :] - target_displacement
    )
    first_rmse = _rmse_from_vectors(first_error[frame_slice][selected])
    second_rmse = _rmse_from_vectors(second_error[frame_slice][selected])
    relative_improvement = None
    if (
        first_rmse is not None
        and second_rmse is not None
        and second_rmse > 0.0
    ):
        relative_improvement = float(1.0 - first_rmse / second_rmse)
    return {
        "frame_range": [frame_start, end],
        "shared_count": int(np.sum(selected)),
        "first_displacement_rmse_m": first_rmse,
        "second_displacement_rmse_m": second_rmse,
        "first_relative_improvement_fraction": relative_improvement,
    }


def evaluate_tapip3d_competence_gates(
    tapip3d_metrics: dict[str, Any],
    late_metrics: dict[str, Any],
    shared_metrics: dict[str, Any],
    *,
    minimum_support_fraction: float,
    minimum_shared_rmse_improvement_fraction: float,
    maximum_displacement_rmse_m: float,
    maximum_frame_zero_anchor_rmse_m: float,
    minimum_late_support_fraction: float,
    maximum_late_displacement_rmse_m: float,
) -> dict[str, bool]:
    """Apply the fully specified source-only competence gate."""

    support = tapip3d_metrics["support_fraction"]
    displacement = tapip3d_metrics["displacement_rmse_m"]
    anchor = tapip3d_metrics["frame_zero_anchor_rmse_m"]
    late_support = late_metrics["support_fraction"]
    late_displacement = late_metrics["displacement_rmse_m"]
    shared_improvement = shared_metrics[
        "first_relative_improvement_fraction"
    ]
    gates = {
        "prefix_support_at_least_minimum": (
            support is not None and support >= minimum_support_fraction
        ),
        "shared_rmse_improvement_at_least_minimum": (
            shared_improvement is not None
            and shared_improvement
            >= minimum_shared_rmse_improvement_fraction
        ),
        "displacement_rmse_at_most_maximum": (
            displacement is not None
            and displacement <= maximum_displacement_rmse_m
        ),
        "frame_zero_anchor_rmse_at_most_maximum": (
            anchor is not None and anchor <= maximum_frame_zero_anchor_rmse_m
        ),
        "late_support_at_least_minimum": (
            late_support is not None
            and late_support >= minimum_late_support_fraction
        ),
        "late_displacement_rmse_at_most_maximum": (
            late_displacement is not None
            and late_displacement <= maximum_late_displacement_rmse_m
        ),
    }
    gates["competence_gate_passed"] = all(gates.values())
    return gates


__all__ = [
    "FrameZeroAssociation",
    "IdentityTrajectory",
    "Tapip3dPrediction",
    "associate_frame_zero_queries",
    "build_same_query_cotracker3_trajectory",
    "evaluate_tapip3d_competence_gates",
    "identity_trajectory_metrics",
    "load_canonical_tapip3d_prediction",
    "load_tapip3d_prediction",
    "save_canonical_tapip3d_prediction",
    "shared_support_displacement_metrics",
    "validate_tapip3d_prediction_contract",
]
