"""Automatic dense MotionCrafter-to-PhysTwin graph association."""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_additional_confirmation import (
    apply_endpoint_transform,
    fit_endpoint_transform,
)
from .phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from .phystwin_official_evaluation import _nearest_distances


MOTIONCRAFTER_REPOSITORY = "https://github.com/TencentARC/MotionCrafter"
MOTIONCRAFTER_REVISION = "1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257"


@dataclass(frozen=True)
class MotionCrafterAssociationConfig:
    """Frozen settings for dense transport and spring-graph association."""

    camera_index: int = 0
    process_stride: int = 1
    seed_stride_pixels: int = 4
    alignment_stride_pixels: int = 4
    alignment_trim_fraction: float = 0.8
    alignment_iterations: int = 5
    maximum_transport_error_m: float = 0.02
    transport_candidate_count: int = 4
    candidate_count: int = 8
    position_scale_m: float = 0.01
    motion_scale_m: float = 0.02
    motion_strength: float = 1.0
    graph_scale_m: float = 0.015
    graph_strength: float = 0.3
    collision_strength: float = 0.1
    mean_field_iterations: int = 5
    minimum_trajectory_valid_fraction: float = 0.5
    minimum_observation_mass: float = 0.5


@dataclass(frozen=True)
class MotionCrafterPrediction:
    """Official MotionCrafter point-map and forward-flow output."""

    point_map: np.ndarray
    valid_mask: np.ndarray
    scene_flow: np.ndarray
    deform_mask: np.ndarray


@dataclass(frozen=True)
class DenseMotionTrajectories:
    """Persistent trajectories composed from Eulerian scene flow."""

    positions: np.ndarray
    valid: np.ndarray
    step_error_m: np.ndarray
    pixel_indices: np.ndarray
    seed_pixels_yx: np.ndarray


@dataclass(frozen=True)
class GraphAssociation:
    """Sparse soft mapping from graph vertices to dense trajectories."""

    trajectory_indices: np.ndarray
    weights: np.ndarray
    confidence: np.ndarray
    initial_error_m: np.ndarray
    normalized_entropy: np.ndarray
    candidate_valid_fraction: np.ndarray
    training_motion_error_m: np.ndarray


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(len(finite)),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(np.max(finite)),
    }


def _nearest_neighbors(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    k: int,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean k-nearest neighbors with a NumPy fallback."""

    reference_points = np.asarray(reference, dtype=float)
    query_points = np.asarray(query, dtype=float)
    if (
        reference_points.ndim != 2
        or query_points.ndim != 2
        or reference_points.shape[1] != query_points.shape[1]
    ):
        raise ValueError("reference and query must have matching shape (*, D)")
    if not 1 <= k <= len(reference_points):
        raise ValueError("k must lie in [1, len(reference)]")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from scipy.spatial import cKDTree
    except (ImportError, OSError, ValueError, Warning):
        cKDTree = None
    if cKDTree is not None:
        distance, indices = cKDTree(reference_points).query(query_points, k=k)
        if k == 1:
            distance = np.asarray(distance)[:, None]
            indices = np.asarray(indices)[:, None]
        return np.asarray(distance, dtype=float), np.asarray(indices, dtype=np.int64)

    distance = np.empty((len(query_points), k), dtype=float)
    indices = np.empty((len(query_points), k), dtype=np.int64)
    for start in range(0, len(query_points), chunk_size):
        stop = min(start + chunk_size, len(query_points))
        delta = query_points[start:stop, None] - reference_points[None]
        squared = np.sum(np.square(delta), axis=2)
        if k == 1:
            local = np.argmin(squared, axis=1)[:, None]
        else:
            local = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
            local_squared = np.take_along_axis(squared, local, axis=1)
            order = np.argsort(local_squared, axis=1, kind="mergesort")
            local = np.take_along_axis(local, order, axis=1)
        indices[start:stop] = local
        distance[start:stop] = np.sqrt(
            np.take_along_axis(squared, local, axis=1)
        )
    return distance, indices


def load_motioncrafter_prediction(path: str | Path) -> MotionCrafterPrediction:
    """Load and validate the official MotionCrafter inference archive."""

    with np.load(path) as archive:
        required = {"point_map", "valid_mask", "scene_flow", "deform_mask"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"MotionCrafter archive is missing {missing}")
        point_map = np.asarray(archive["point_map"], dtype=np.float32)
        valid_mask = np.asarray(archive["valid_mask"], dtype=bool)
        scene_flow = np.asarray(archive["scene_flow"], dtype=np.float32)
        deform_mask = np.asarray(archive["deform_mask"], dtype=bool)
    if point_map.ndim != 4 or point_map.shape[-1] != 3:
        raise ValueError("point_map must have shape (T, H, W, 3)")
    if valid_mask.shape != point_map.shape[:3]:
        raise ValueError("valid_mask must match point_map")
    if scene_flow.shape != point_map.shape:
        raise ValueError("scene_flow must match point_map")
    if deform_mask.shape != point_map.shape[:3]:
        raise ValueError("deform_mask must match point_map")
    if len(point_map) < 2:
        raise ValueError("MotionCrafter output must contain at least two frames")
    return MotionCrafterPrediction(
        point_map=point_map,
        valid_mask=valid_mask,
        scene_flow=scene_flow,
        deform_mask=deform_mask,
    )


def cover_resize_source_indices(
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map MotionCrafter cover-resize/crop pixels to source-image indices."""

    source_height, source_width = (int(value) for value in source_shape)
    target_height, target_width = (int(value) for value in target_shape)
    if min(source_height, source_width, target_height, target_width) < 1:
        raise ValueError("image dimensions must be positive")
    scale = max(target_height / source_height, target_width / source_width)
    resized_height = int(source_height * scale)
    resized_width = int(source_width * scale)
    if resized_height < target_height or resized_width < target_width:
        raise RuntimeError("cover resize does not cover the requested target")
    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    target_y = np.arange(target_height, dtype=float)
    target_x = np.arange(target_width, dtype=float)
    source_y = np.rint((target_y + top + 0.5) / scale - 0.5).astype(np.int64)
    source_x = np.rint((target_x + left + 0.5) / scale - 0.5).astype(np.int64)
    return (
        np.clip(source_y, 0, source_height - 1),
        np.clip(source_x, 0, source_width - 1),
    )


def resample_cover_grid(
    values: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-sample a source image/grid under MotionCrafter cover resize."""

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("values must start with image height and width")
    source_y, source_x = cover_resize_source_indices(
        (array.shape[0], array.shape[1]), target_shape
    )
    return array[source_y[:, None], source_x[None, :]]


def robust_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    trim_fraction: float = 0.8,
    iterations: int = 5,
) -> dict[str, object]:
    """Fit a trimmed Sim(3) transform for noisy same-pixel point maps."""

    source_points = np.asarray(source, dtype=float)
    target_points = np.asarray(target, dtype=float)
    if source_points.shape != target_points.shape or source_points.ndim != 2:
        raise ValueError("source and target must have matching shape (N, 3)")
    if source_points.shape[1] != 3:
        raise ValueError("similarity points must be three-dimensional")
    finite = np.all(np.isfinite(source_points), axis=1) & np.all(
        np.isfinite(target_points), axis=1
    )
    selected = np.flatnonzero(finite)
    if len(selected) < 4:
        raise ValueError("similarity fit requires at least four finite pairs")
    if not 0.5 <= trim_fraction <= 1.0:
        raise ValueError("trim_fraction must lie in [0.5, 1]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    keep_count = max(4, int(np.ceil(trim_fraction * len(selected))))
    transform: dict[str, object] | None = None
    residual = np.full(len(source_points), np.nan, dtype=float)
    for _ in range(iterations):
        transform = fit_endpoint_transform(
            source_points[selected], target_points[selected], mode="sim3"
        )
        fitted = apply_endpoint_transform(source_points[finite], transform)
        finite_residual = np.linalg.norm(fitted - target_points[finite], axis=1)
        residual[finite] = finite_residual
        finite_indices = np.flatnonzero(finite)
        order = np.argsort(finite_residual, kind="mergesort")
        next_selected = finite_indices[order[:keep_count]]
        if np.array_equal(next_selected, selected):
            break
        selected = next_selected
    transform = fit_endpoint_transform(
        source_points[selected], target_points[selected], mode="sim3"
    )
    fitted = apply_endpoint_transform(source_points[finite], transform)
    residual[finite] = np.linalg.norm(fitted - target_points[finite], axis=1)
    transform = dict(transform)
    transform.update(
        {
            "trim_fraction": trim_fraction,
            "iterations": iterations,
            "input_pair_count": int(np.sum(finite)),
            "inlier_pair_count": int(len(selected)),
            "inlier_mask": np.isin(np.arange(len(source_points)), selected),
            "all_pair_residual_m": residual,
            "inlier_rmse_m": float(
                np.sqrt(np.mean(np.square(residual[selected])))
            ),
        }
    )
    return transform


def align_motioncrafter_prediction(
    prediction: MotionCrafterPrediction,
    transform: dict[str, object],
) -> MotionCrafterPrediction:
    """Apply Sim(3) to point maps and its linear part to scene flow."""

    linear = np.asarray(transform["linear"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    point_map = prediction.point_map.astype(float) @ linear + translation
    scene_flow = prediction.scene_flow.astype(float) @ linear
    return MotionCrafterPrediction(
        point_map=point_map.astype(np.float32),
        valid_mask=prediction.valid_mask,
        scene_flow=scene_flow.astype(np.float32),
        deform_mask=prediction.deform_mask,
    )


def robust_icp_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    mode: str = "se3",
    trim_fraction: float = 0.8,
    iterations: int = 8,
    maximum_correspondence_m: float = 0.08,
) -> dict[str, object]:
    """Fit a trimmed nearest-neighbor endpoint transform."""

    source_points = np.asarray(source, dtype=float)
    target_points = np.asarray(target, dtype=float)
    if (
        source_points.ndim != 2
        or target_points.ndim != 2
        or source_points.shape[1] != 3
        or target_points.shape[1] != 3
    ):
        raise ValueError("ICP source and target must have shape (N, 3)")
    if len(source_points) < 4 or len(target_points) < 4:
        raise ValueError("ICP requires at least four points per cloud")
    if not 0.5 <= trim_fraction <= 1.0 or iterations < 1:
        raise ValueError("invalid ICP trimming or iteration count")
    if maximum_correspondence_m <= 0.0:
        raise ValueError("maximum_correspondence_m must be positive")
    current = source_points.copy()
    total_linear = np.eye(3)
    total_translation = np.zeros(3)
    initial_distance: np.ndarray | None = None
    selected_count = 0
    for _ in range(iterations):
        distance, nearest = _nearest_neighbors(target_points, current, k=1)
        distance = distance[:, 0]
        nearest = nearest[:, 0]
        if initial_distance is None:
            initial_distance = distance.copy()
        usable = np.isfinite(distance) & (distance <= maximum_correspondence_m)
        usable_indices = np.flatnonzero(usable)
        if len(usable_indices) < 4:
            raise ValueError("too few gated ICP correspondences")
        keep_count = max(4, int(np.ceil(trim_fraction * len(usable_indices))))
        selected = usable_indices[
            np.argsort(distance[usable_indices], kind="mergesort")[:keep_count]
        ]
        selected_count = len(selected)
        delta = fit_endpoint_transform(
            current[selected], target_points[nearest[selected]], mode=mode
        )
        delta_linear = np.asarray(delta["linear"], dtype=float)
        delta_translation = np.asarray(delta["translation"], dtype=float)
        current = current @ delta_linear + delta_translation
        total_translation = total_translation @ delta_linear + delta_translation
        total_linear = total_linear @ delta_linear
    final_distance, _ = _nearest_neighbors(target_points, current, k=1)
    return {
        "mode": mode,
        "linear": total_linear,
        "translation": total_translation,
        "iterations": iterations,
        "trim_fraction": trim_fraction,
        "selected_pair_count": selected_count,
        "initial_nearest_residual_m": np.asarray(initial_distance),
        "final_nearest_residual_m": final_distance[:, 0],
    }


def reverse_dense_trajectories(
    reverse_time_trajectories: DenseMotionTrajectories,
) -> DenseMotionTrajectories:
    """Put trajectories composed on a reversed video into original time order."""

    return DenseMotionTrajectories(
        positions=reverse_time_trajectories.positions[::-1].copy(),
        valid=reverse_time_trajectories.valid[::-1].copy(),
        step_error_m=reverse_time_trajectories.step_error_m[::-1].copy(),
        pixel_indices=reverse_time_trajectories.pixel_indices[::-1].copy(),
        seed_pixels_yx=reverse_time_trajectories.seed_pixels_yx.copy(),
    )


def compose_dense_trajectories(
    prediction: MotionCrafterPrediction,
    object_masks: np.ndarray,
    *,
    seed_stride_pixels: int = 4,
    maximum_transport_error_m: float = 0.02,
    transport_candidate_count: int = 4,
) -> DenseMotionTrajectories:
    """Compose Eulerian flow with gated one-to-one point-map transport."""

    if seed_stride_pixels < 1:
        raise ValueError("seed_stride_pixels must be positive")
    if maximum_transport_error_m <= 0.0:
        raise ValueError("maximum_transport_error_m must be positive")
    if transport_candidate_count < 1:
        raise ValueError("transport_candidate_count must be positive")
    masks = np.asarray(object_masks, dtype=bool)
    if masks.shape != prediction.valid_mask.shape:
        raise ValueError("object_masks must match MotionCrafter frame/image shape")
    frame_count, height, width = masks.shape
    grid_y, grid_x = np.indices((height, width))
    seed_mask = (
        masks[0]
        & prediction.valid_mask[0]
        & (grid_y % seed_stride_pixels == 0)
        & (grid_x % seed_stride_pixels == 0)
        & np.all(np.isfinite(prediction.point_map[0]), axis=2)
    )
    seed_pixels = np.column_stack(np.nonzero(seed_mask)).astype(np.int32)
    if len(seed_pixels) == 0:
        raise ValueError("no valid MotionCrafter object seeds")
    track_count = len(seed_pixels)
    positions = np.full((frame_count, track_count, 3), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count, track_count), dtype=bool)
    pixel_indices = np.full((frame_count, track_count), -1, dtype=np.int64)
    step_error = np.full((frame_count - 1, track_count), np.nan, dtype=np.float32)
    initial_flat = seed_pixels[:, 0].astype(np.int64) * width + seed_pixels[:, 1]
    pixel_indices[0] = initial_flat
    flat_points = prediction.point_map.reshape(frame_count, height * width, 3)
    flat_flow = prediction.scene_flow.reshape(frame_count, height * width, 3)
    flat_valid = prediction.valid_mask.reshape(frame_count, height * width)
    flat_deform = prediction.deform_mask.reshape(frame_count, height * width)
    flat_masks = masks.reshape(frame_count, height * width)
    positions[0] = flat_points[0, initial_flat]
    valid[0] = True

    for frame in range(frame_count - 1):
        active_tracks = np.flatnonzero(valid[frame])
        if len(active_tracks) == 0:
            break
        current_pixels = pixel_indices[frame, active_tracks]
        source_usable = (
            flat_valid[frame, current_pixels]
            & flat_deform[frame, current_pixels]
            & np.all(np.isfinite(flat_flow[frame, current_pixels]), axis=1)
        )
        active_tracks = active_tracks[source_usable]
        current_pixels = current_pixels[source_usable]
        if len(active_tracks) == 0:
            continue
        next_candidate_mask = (
            flat_masks[frame + 1]
            & flat_valid[frame + 1]
            & np.all(np.isfinite(flat_points[frame + 1]), axis=1)
        )
        next_pixels = np.flatnonzero(next_candidate_mask)
        if len(next_pixels) == 0:
            continue
        next_points = flat_points[frame + 1, next_pixels]
        predicted = flat_points[frame, current_pixels] + flat_flow[frame, current_pixels]
        query_count = min(transport_candidate_count, len(next_points))
        distance, nearest = _nearest_neighbors(
            next_points, predicted, k=query_count
        )
        proposed_pixels = next_pixels[nearest]
        accepted = np.isfinite(distance) & (
            distance <= maximum_transport_error_m
        )
        if not np.any(accepted):
            continue

        proposal_track, proposal_rank = np.nonzero(accepted)
        proposal_distance = distance[proposal_track, proposal_rank]
        proposal_pixel = proposed_pixels[proposal_track, proposal_rank]
        order = np.lexsort(
            (
                proposal_rank,
                active_tracks[proposal_track],
                proposal_pixel,
                proposal_distance,
            )
        )
        assigned_track = np.zeros(len(active_tracks), dtype=bool)
        assigned_pixel = np.zeros(height * width, dtype=bool)
        winner_local: list[int] = []
        winner_pixels: list[int] = []
        winner_distance: list[float] = []
        for proposal in order:
            local = int(proposal_track[proposal])
            pixel = int(proposal_pixel[proposal])
            if assigned_track[local] or assigned_pixel[pixel]:
                continue
            assigned_track[local] = True
            assigned_pixel[pixel] = True
            winner_local.append(local)
            winner_pixels.append(pixel)
            winner_distance.append(float(proposal_distance[proposal]))
        if not winner_local:
            continue
        winner_local_array = np.asarray(winner_local, dtype=np.int64)
        winner_pixels_array = np.asarray(winner_pixels, dtype=np.int64)
        winner_tracks = active_tracks[winner_local_array]
        valid[frame + 1, winner_tracks] = True
        pixel_indices[frame + 1, winner_tracks] = winner_pixels_array
        positions[frame + 1, winner_tracks] = flat_points[
            frame + 1, winner_pixels_array
        ]
        step_error[frame, winner_tracks] = np.asarray(
            winner_distance, dtype=np.float32
        )

    return DenseMotionTrajectories(
        positions=positions,
        valid=valid,
        step_error_m=step_error,
        pixel_indices=pixel_indices,
        seed_pixels_yx=seed_pixels,
    )


def concatenate_dense_trajectories(
    trajectories_by_camera: dict[int, DenseMotionTrajectories],
) -> tuple[DenseMotionTrajectories, np.ndarray]:
    """Concatenate calibrated view trajectories and retain their camera IDs."""

    if not trajectories_by_camera:
        raise ValueError("at least one camera trajectory set is required")
    ordered = sorted(trajectories_by_camera.items())
    frame_counts = {len(trajectories.positions) for _, trajectories in ordered}
    if len(frame_counts) != 1:
        raise ValueError("all camera trajectory sets must share a frame count")
    combined = DenseMotionTrajectories(
        positions=np.concatenate(
            [trajectories.positions for _, trajectories in ordered], axis=1
        ),
        valid=np.concatenate(
            [trajectories.valid for _, trajectories in ordered], axis=1
        ),
        step_error_m=np.concatenate(
            [trajectories.step_error_m for _, trajectories in ordered], axis=1
        ),
        pixel_indices=np.concatenate(
            [trajectories.pixel_indices for _, trajectories in ordered], axis=1
        ),
        seed_pixels_yx=np.concatenate(
            [trajectories.seed_pixels_yx for _, trajectories in ordered], axis=0
        ),
    )
    camera_indices = np.concatenate(
        [
            np.full(
                trajectories.positions.shape[1], camera, dtype=np.int16
            )
            for camera, trajectories in ordered
        ]
    )
    return combined, camera_indices


def _softmax_negative_cost(cost: np.ndarray) -> np.ndarray:
    shifted = -np.asarray(cost, dtype=float)
    shifted -= np.max(shifted, axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -700.0, 0.0))
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def infer_graph_association(
    graph_initial: np.ndarray,
    springs: np.ndarray,
    trajectories: DenseMotionTrajectories,
    *,
    candidate_count: int = 8,
    position_scale_m: float = 0.01,
    motion_scale_m: float = 0.02,
    motion_strength: float = 1.0,
    graph_scale_m: float = 0.015,
    graph_strength: float = 0.3,
    collision_strength: float = 0.1,
    mean_field_iterations: int = 5,
    minimum_trajectory_valid_fraction: float = 0.5,
    association_frame_count: int | None = None,
    graph_training_trajectory: np.ndarray | None = None,
) -> GraphAssociation:
    """Infer a locally injective graph map from an allowed frame prefix."""

    graph = np.asarray(graph_initial, dtype=float)
    edges = np.asarray(springs, dtype=np.int64)
    if graph.ndim != 2 or graph.shape[1] != 3 or not np.all(np.isfinite(graph)):
        raise ValueError("graph_initial must contain finite shape (N, 3) points")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("springs must have shape (S, 2)")
    if np.any(edges < 0) or np.any(edges >= len(graph)):
        raise ValueError("spring endpoint exceeds graph_initial")
    if candidate_count < 1 or mean_field_iterations < 1:
        raise ValueError("candidate_count and mean_field_iterations must be positive")
    if min(position_scale_m, motion_scale_m, graph_scale_m) <= 0.0:
        raise ValueError("association scales must be positive")
    if min(motion_strength, graph_strength, collision_strength) < 0.0:
        raise ValueError("association strengths must be nonnegative")
    if not 0.0 <= minimum_trajectory_valid_fraction <= 1.0:
        raise ValueError("minimum trajectory valid fraction must lie in [0, 1]")
    if association_frame_count is None:
        association_frame_count = len(trajectories.valid)
    if not 1 <= association_frame_count <= len(trajectories.valid):
        raise ValueError("association_frame_count exceeds the trajectory frames")
    trajectory_valid_fraction = np.mean(
        trajectories.valid[:association_frame_count], axis=0
    )
    eligible = (
        trajectories.valid[0]
        & (trajectory_valid_fraction >= minimum_trajectory_valid_fraction)
        & np.all(np.isfinite(trajectories.positions[0]), axis=1)
    )
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < candidate_count:
        raise ValueError("too few persistent dense trajectories for association")
    dense_initial = trajectories.positions[0, eligible_indices].astype(float)
    distance, local_candidates = _nearest_neighbors(
        dense_initial, graph, k=candidate_count
    )
    candidate_indices = eligible_indices[np.asarray(local_candidates, dtype=np.int64)]
    distance = np.asarray(distance, dtype=float)
    candidate_valid_fraction = trajectory_valid_fraction[candidate_indices]
    unary = 0.5 * np.square(distance / position_scale_m) - np.log(
        np.maximum(candidate_valid_fraction, 1e-6)
    )
    candidate_points = trajectories.positions[0, candidate_indices].astype(float)
    candidate_motion_error = np.full_like(unary, np.nan)
    if graph_training_trajectory is not None:
        training_target = np.asarray(graph_training_trajectory, dtype=float)
        expected_shape = (association_frame_count, len(graph), 3)
        if training_target.shape != expected_shape:
            raise ValueError(
                "graph_training_trajectory must match the association prefix "
                f"shape {expected_shape}"
            )
        robust_sum = np.zeros_like(unary)
        error_sum = np.zeros_like(unary)
        motion_count = np.zeros_like(unary)
        target_reference = training_target[0]
        for frame in range(1, association_frame_count):
            target = training_target[frame]
            target_valid = np.all(np.isfinite(target), axis=1) & np.all(
                np.isfinite(target_reference), axis=1
            )
            candidate = trajectories.positions[frame, candidate_indices].astype(
                float
            )
            candidate_valid = trajectories.valid[frame, candidate_indices]
            usable = target_valid[:, None] & candidate_valid & np.all(
                np.isfinite(candidate), axis=2
            )
            target_delta = target - target_reference
            candidate_delta = candidate - candidate_points
            error = np.linalg.norm(
                candidate_delta - target_delta[:, None, :], axis=2
            )
            scaled = error / motion_scale_m
            robust = 2.0 * (np.sqrt(1.0 + np.square(scaled)) - 1.0)
            robust_sum += np.where(usable, robust, 0.0)
            error_sum += np.where(usable, error, 0.0)
            motion_count += usable
        motion_available = motion_count > 0.0
        motion_cost = np.zeros_like(unary)
        motion_cost[motion_available] = (
            robust_sum[motion_available] / motion_count[motion_available]
        )
        candidate_motion_error[motion_available] = (
            error_sum[motion_available] / motion_count[motion_available]
        )
        unary += motion_strength * motion_cost
    weights = _softmax_negative_cost(unary)

    for _ in range(mean_field_iterations):
        expected = np.sum(weights[:, :, None] * candidate_points, axis=1)
        pair_cost = np.zeros_like(unary)
        pair_count = np.zeros((len(graph), 1), dtype=float)
        for first, second in edges:
            graph_delta = graph[first] - graph[second]
            first_delta = candidate_points[first] - expected[second]
            second_delta = candidate_points[second] - expected[first]
            pair_cost[first] += np.sum(
                np.square(first_delta - graph_delta), axis=1
            ) / np.square(graph_scale_m)
            pair_cost[second] += np.sum(
                np.square(second_delta + graph_delta), axis=1
            ) / np.square(graph_scale_m)
            pair_count[first] += 1.0
            pair_count[second] += 1.0
        pair_cost /= np.maximum(pair_count, 1.0)

        usage = np.zeros(trajectories.positions.shape[1], dtype=float)
        np.add.at(usage, candidate_indices.reshape(-1), weights.reshape(-1))
        collision = np.maximum(usage[candidate_indices] - 1.0, 0.0)
        total_cost = (
            unary
            + 0.5 * graph_strength * pair_cost
            + collision_strength * collision
        )
        weights = _softmax_negative_cost(total_cost)

    initial_observation = np.sum(weights[:, :, None] * candidate_points, axis=1)
    initial_error = np.linalg.norm(initial_observation - graph, axis=1)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    normalized_entropy = (
        np.zeros(len(graph), dtype=float)
        if candidate_count == 1
        else entropy / np.log(candidate_count)
    )
    weighted_valid_fraction = np.sum(weights * candidate_valid_fraction, axis=1)
    confidence = (
        weighted_valid_fraction
        * np.exp(-0.5 * np.square(initial_error / position_scale_m))
        * (1.0 - 0.5 * normalized_entropy)
    )
    motion_weight = weights * np.isfinite(candidate_motion_error)
    motion_denominator = np.sum(motion_weight, axis=1)
    training_motion_error = np.full(len(graph), np.nan, dtype=float)
    motion_available = motion_denominator > 0.0
    training_motion_error[motion_available] = np.sum(
        np.where(
            np.isfinite(candidate_motion_error),
            weights * candidate_motion_error,
            0.0,
        ),
        axis=1,
    )[motion_available] / motion_denominator[motion_available]
    return GraphAssociation(
        trajectory_indices=candidate_indices.astype(np.int32),
        weights=weights.astype(np.float32),
        confidence=np.clip(confidence, 0.0, 1.0).astype(np.float32),
        initial_error_m=initial_error.astype(np.float32),
        normalized_entropy=normalized_entropy.astype(np.float32),
        candidate_valid_fraction=candidate_valid_fraction.astype(np.float32),
        training_motion_error_m=training_motion_error.astype(np.float32),
    )


def apply_graph_association(
    trajectories: DenseMotionTrajectories,
    association: GraphAssociation,
    *,
    minimum_observation_mass: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return graph observations, validity, and reliability from a soft map."""

    if not 0.0 <= minimum_observation_mass <= 1.0:
        raise ValueError("minimum_observation_mass must lie in [0, 1]")
    indices = association.trajectory_indices
    weights = association.weights.astype(float)
    candidate_positions = trajectories.positions[:, indices].astype(float)
    candidate_valid = trajectories.valid[:, indices]
    effective = weights[None] * candidate_valid
    mass = np.sum(effective, axis=2)
    numerator = np.sum(
        effective[:, :, :, None]
        * np.where(candidate_valid[:, :, :, None], candidate_positions, 0.0),
        axis=2,
    )
    observations = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = mass >= minimum_observation_mass
    observations[valid] = (numerator / np.maximum(mass[:, :, None], 1e-12))[
        valid
    ].astype(np.float32)
    reliability = mass * association.confidence[None]
    reliability[~valid] = 0.0
    return observations, valid, reliability.astype(np.float32)


def dense_graph_error_by_frame(
    graph_trajectory: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    reliability: np.ndarray,
) -> np.ndarray:
    """Compute a reliability-weighted dense graph correspondence error."""

    graph = np.asarray(graph_trajectory, dtype=float)
    observed = np.asarray(observations, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    weight = np.asarray(reliability, dtype=float)
    if graph.shape != observed.shape or mask.shape != graph.shape[:2]:
        raise ValueError("graph, observations, and valid shapes disagree")
    if weight.shape != mask.shape:
        raise ValueError("reliability must match valid")
    usable = mask & np.all(np.isfinite(graph), axis=2) & (weight > 0.0)
    distance = np.linalg.norm(graph - observed, axis=2)
    weighted = np.where(usable, weight, 0.0)
    denominator = np.sum(weighted, axis=1)
    output = np.full(len(graph), np.nan, dtype=float)
    available = denominator > 0.0
    output[available] = np.sum(
        np.where(usable, distance * weighted, 0.0), axis=1
    )[available] / denominator[available]
    return output


def manual_track_association_audit(
    graph_initial: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    manual_tracks: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, object]:
    """Audit automatic identities against sparse manual tracks after locking."""

    graph = np.asarray(graph_initial, dtype=float)
    observed = np.asarray(observations, dtype=float)
    observation_valid = np.asarray(valid, dtype=bool)
    tracks = np.asarray(manual_tracks, dtype=float)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        raise ValueError("manual_tracks must have shape (T, K, 3)")
    if len(frames) != len(observed) or np.any(frames < 0) or np.any(frames >= len(tracks)):
        raise ValueError("frame_indices do not map observations into manual_tracks")
    initial_mask = np.all(np.isfinite(tracks[0]), axis=1)
    _, graph_indices = _nearest_distances(graph, tracks[0, initial_mask], p=2)
    selected_tracks = tracks[:, initial_mask]
    by_frame = np.full(len(frames), np.nan, dtype=float)
    count_by_frame = np.zeros(len(frames), dtype=np.int32)
    for output_frame, source_frame in enumerate(frames):
        manual = selected_tracks[source_frame]
        usable = (
            np.all(np.isfinite(manual), axis=1)
            & observation_valid[output_frame, graph_indices]
            & np.all(
                np.isfinite(observed[output_frame, graph_indices]), axis=1
            )
        )
        count_by_frame[output_frame] = int(np.sum(usable))
        if np.any(usable):
            by_frame[output_frame] = float(
                np.mean(
                    np.linalg.norm(
                        observed[output_frame, graph_indices[usable]]
                        - manual[usable],
                        axis=1,
                    )
                )
            )
    return {
        "manual_track_count": int(np.sum(initial_mask)),
        "graph_vertex_indices": graph_indices.astype(int).tolist(),
        "error_by_sampled_frame_m": by_frame.tolist(),
        "available_track_count_by_sampled_frame": count_by_frame.astype(int).tolist(),
        "error_distribution_m": _distribution(by_frame),
    }


def _mean_on_frames(values: np.ndarray, selection: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=float)[np.asarray(selection, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    return None if len(selected) == 0 else float(np.mean(selected))


def associate_motioncrafter_case(
    case_dir: str | Path,
    raw_case_dir: str | Path,
    motioncrafter_npz_path: str | Path,
    output_dir: str | Path,
    *,
    config: MotionCrafterAssociationConfig,
    train_end_frame: int | None = None,
    additional_views: dict[int, str | Path] | None = None,
    reverse_motioncrafter_npz_path: str | Path | None = None,
) -> dict[str, object]:
    """Associate calibrated MotionCrafter views with a released graph."""

    if config.camera_index < 0 or config.process_stride < 1:
        raise ValueError("camera_index must be nonnegative and stride positive")
    if reverse_motioncrafter_npz_path is not None and (
        config.process_stride != 1 or additional_views
    ):
        raise ValueError(
            "reverse evaluation currently requires one native-rate camera view"
        )
    case_path = Path(case_dir)
    raw_path = Path(raw_case_dir)
    output = Path(output_dir)
    final_path = case_path / "final_data.pkl"
    baseline_path = case_path / "inference.pkl"
    optimal_path = case_path / "optimal_params.pkl"
    track_path = case_path / "gt_track_3d.pkl"
    split_path = case_path / "split.json"
    data = _load_pickle(final_path)
    baseline = np.asarray(_load_pickle(baseline_path), dtype=float)
    optimal = _load_pickle(optimal_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_end = int(split["test"][0]) if train_end_frame is None else train_end_frame
    frame_count = int(split["frame_len"])
    if not 1 < train_end < frame_count:
        raise ValueError("train_end_frame must leave a future interval")
    view_paths = {config.camera_index: Path(motioncrafter_npz_path)}
    for camera, path in (additional_views or {}).items():
        camera_index = int(camera)
        if camera_index < 0:
            raise ValueError("additional camera indices must be nonnegative")
        if camera_index in view_paths:
            raise ValueError(f"duplicate MotionCrafter camera {camera_index}")
        view_paths[camera_index] = Path(path)
    predictions = {
        camera: load_motioncrafter_prediction(path)
        for camera, path in sorted(view_paths.items())
    }
    prediction = predictions[config.camera_index]
    frame_indices = np.arange(0, frame_count, config.process_stride, dtype=np.int64)
    if len(frame_indices) < len(prediction.point_map):
        frame_indices = frame_indices[: len(prediction.point_map)]
    if len(frame_indices) != len(prediction.point_map):
        raise ValueError("process_stride does not explain MotionCrafter frame count")
    for camera, candidate in predictions.items():
        if candidate.point_map.shape != prediction.point_map.shape:
            raise ValueError(
                f"MotionCrafter camera {camera} shape does not match the primary view"
            )

    pcd_path = raw_path / "pcd" / "0.npz"
    with np.load(pcd_path) as pcd_archive:
        camera_points = np.asarray(pcd_archive["points"], dtype=float)
    if any(camera < 0 or camera >= len(camera_points) for camera in view_paths):
        raise ValueError("camera_index exceeds raw point-cloud cameras")
    with (raw_path / "mask" / "processed_masks.pkl").open("rb") as handle:
        processed_masks = pickle.load(handle)
    transforms: dict[int, dict[str, object]] = {}
    trajectories_by_camera: dict[int, DenseMotionTrajectories] = {}
    object_masks_by_camera: dict[int, np.ndarray] = {}
    for camera, camera_prediction in predictions.items():
        target_shape = camera_prediction.point_map.shape[1:3]
        object_masks = np.stack(
            [
                resample_cover_grid(
                    np.asarray(
                        processed_masks[int(frame)][camera]["object"]
                    ),
                    target_shape,
                ).astype(bool)
                for frame in frame_indices
            ]
        )
        initial_world = resample_cover_grid(
            camera_points[camera], target_shape
        ).astype(float)
        grid_y, grid_x = np.indices(target_shape)
        alignment_mask = (
            object_masks[0]
            & camera_prediction.valid_mask[0]
            & np.all(np.isfinite(camera_prediction.point_map[0]), axis=2)
            & np.all(np.isfinite(initial_world), axis=2)
            & (np.linalg.norm(initial_world, axis=2) > 1e-6)
            & (grid_y % config.alignment_stride_pixels == 0)
            & (grid_x % config.alignment_stride_pixels == 0)
        )
        transform = robust_similarity_transform(
            camera_prediction.point_map[0, alignment_mask],
            initial_world[alignment_mask],
            trim_fraction=config.alignment_trim_fraction,
            iterations=config.alignment_iterations,
        )
        aligned = align_motioncrafter_prediction(camera_prediction, transform)
        transforms[camera] = transform
        object_masks_by_camera[camera] = object_masks
        trajectories_by_camera[camera] = compose_dense_trajectories(
            aligned,
            object_masks,
            seed_stride_pixels=config.seed_stride_pixels,
            maximum_transport_error_m=config.maximum_transport_error_m,
            transport_candidate_count=config.transport_candidate_count,
        )
    trajectories, trajectory_camera_indices = concatenate_dense_trajectories(
        trajectories_by_camera
    )

    observed = np.asarray(data["object_points"], dtype=float)
    surface_points = np.asarray(data["surface_points"], dtype=float)
    interior_points = np.asarray(data["interior_points"], dtype=float)
    structure_points = np.concatenate(
        (observed[0], surface_points, interior_points), axis=0
    )
    if baseline.shape[1] != len(structure_points):
        raise ValueError("released graph and baseline trajectory disagree")
    surface_count = len(observed[0]) + len(surface_points)
    graph = build_phystwin_spring_graph(
        structure_points,
        None,
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    surface_springs = graph.springs[
        np.all(graph.springs < surface_count, axis=1)
    ]
    association_frame_count = int(np.sum(frame_indices < train_end))
    training_target = np.full(
        (association_frame_count, surface_count, 3), np.nan, dtype=float
    )
    training_target[0] = structure_points[:surface_count]
    training_target[:, : observed.shape[1]] = observed[
        frame_indices[:association_frame_count]
    ]
    association = infer_graph_association(
        structure_points[:surface_count],
        surface_springs,
        trajectories,
        candidate_count=config.candidate_count,
        position_scale_m=config.position_scale_m,
        motion_scale_m=config.motion_scale_m,
        motion_strength=config.motion_strength,
        graph_scale_m=config.graph_scale_m,
        graph_strength=config.graph_strength,
        collision_strength=config.collision_strength,
        mean_field_iterations=config.mean_field_iterations,
        minimum_trajectory_valid_fraction=(
            config.minimum_trajectory_valid_fraction
        ),
        association_frame_count=association_frame_count,
        graph_training_trajectory=training_target,
    )
    graph_observations, graph_valid, graph_reliability = apply_graph_association(
        trajectories,
        association,
        minimum_observation_mass=config.minimum_observation_mass,
    )
    backward_archive: dict[str, np.ndarray] = {}
    backward_summary: dict[str, object] | None = None
    reverse_alignment_summary: dict[str, object] | None = None
    if reverse_motioncrafter_npz_path is not None:
        reverse_prediction = load_motioncrafter_prediction(
            reverse_motioncrafter_npz_path
        )
        if reverse_prediction.point_map.shape != prediction.point_map.shape:
            raise ValueError("reverse MotionCrafter output shape does not match")
        camera = config.camera_index
        object_masks = object_masks_by_camera[camera]
        target_shape = reverse_prediction.point_map.shape[1:3]
        initial_world = resample_cover_grid(
            camera_points[camera], target_shape
        ).astype(float)
        grid_y, grid_x = np.indices(target_shape)
        initial_mask = (
            object_masks[0]
            & reverse_prediction.valid_mask[-1]
            & np.all(np.isfinite(reverse_prediction.point_map[-1]), axis=2)
            & np.all(np.isfinite(initial_world), axis=2)
            & (np.linalg.norm(initial_world, axis=2) > 1e-6)
            & (grid_y % config.alignment_stride_pixels == 0)
            & (grid_x % config.alignment_stride_pixels == 0)
        )
        reverse_initial_transform = robust_similarity_transform(
            reverse_prediction.point_map[-1, initial_mask],
            initial_world[initial_mask],
            trim_fraction=config.alignment_trim_fraction,
            iterations=config.alignment_iterations,
        )
        aligned_reverse = align_motioncrafter_prediction(
            reverse_prediction, reverse_initial_transform
        )
        bridge_frame = train_end - 1
        reverse_bridge_frame = frame_count - 1 - bridge_frame
        bridge_mask = (
            object_masks[bridge_frame]
            & aligned_reverse.valid_mask[reverse_bridge_frame]
            & np.all(
                np.isfinite(aligned_reverse.point_map[reverse_bridge_frame]),
                axis=2,
            )
            & (grid_y % config.alignment_stride_pixels == 0)
            & (grid_x % config.alignment_stride_pixels == 0)
        )
        bridge_icp = robust_icp_transform(
            aligned_reverse.point_map[reverse_bridge_frame, bridge_mask],
            observed[bridge_frame],
            mode="se3",
            trim_fraction=config.alignment_trim_fraction,
            iterations=config.alignment_iterations,
        )
        aligned_reverse = align_motioncrafter_prediction(
            aligned_reverse, bridge_icp
        )
        reverse_time_trajectories = compose_dense_trajectories(
            aligned_reverse,
            object_masks[::-1],
            seed_stride_pixels=config.seed_stride_pixels,
            maximum_transport_error_m=config.maximum_transport_error_m,
            transport_candidate_count=config.transport_candidate_count,
        )
        backward_trajectories = reverse_dense_trajectories(
            reverse_time_trajectories
        )
        backward_slice = DenseMotionTrajectories(
            positions=backward_trajectories.positions[bridge_frame:],
            valid=backward_trajectories.valid[bridge_frame:],
            step_error_m=backward_trajectories.step_error_m[bridge_frame:],
            pixel_indices=backward_trajectories.pixel_indices[bridge_frame:],
            seed_pixels_yx=backward_trajectories.seed_pixels_yx,
        )
        bridge_graph = baseline[bridge_frame, :surface_count].copy()
        bridge_graph[: observed.shape[1]] = observed[bridge_frame]
        backward_association = infer_graph_association(
            bridge_graph,
            surface_springs,
            backward_slice,
            candidate_count=config.candidate_count,
            position_scale_m=config.position_scale_m,
            motion_scale_m=config.motion_scale_m,
            motion_strength=0.0,
            graph_scale_m=config.graph_scale_m,
            graph_strength=config.graph_strength,
            collision_strength=config.collision_strength,
            mean_field_iterations=config.mean_field_iterations,
            minimum_trajectory_valid_fraction=1.0,
            association_frame_count=1,
        )
        backward_observations, backward_valid, backward_reliability = (
            apply_graph_association(
                backward_slice,
                backward_association,
                minimum_observation_mass=config.minimum_observation_mass,
            )
        )
        backward_future_offset = train_end - bridge_frame
        graph_observations = graph_observations.copy()
        graph_valid = graph_valid.copy()
        graph_reliability = graph_reliability.copy()
        graph_observations[train_end:] = backward_observations[
            backward_future_offset:
        ]
        graph_valid[train_end:] = backward_valid[backward_future_offset:]
        graph_reliability[train_end:] = backward_reliability[
            backward_future_offset:
        ]
        backward_archive = {
            "dense_backward_positions": backward_trajectories.positions,
            "dense_backward_valid": backward_trajectories.valid,
            "dense_backward_step_error_m": backward_trajectories.step_error_m,
            "dense_backward_pixel_indices": backward_trajectories.pixel_indices,
            "dense_backward_seed_pixels_yx": backward_trajectories.seed_pixels_yx,
            "backward_graph_observations": backward_observations,
            "backward_graph_valid": backward_valid,
            "backward_graph_reliability": backward_reliability,
            "backward_trajectory_indices": (
                backward_association.trajectory_indices
            ),
            "backward_association_weights": backward_association.weights,
        }
        backward_summary = {
            "bridge_frame": bridge_frame,
            "seed_count": int(backward_trajectories.positions.shape[1]),
            "valid_fraction_by_original_frame": np.mean(
                backward_trajectories.valid, axis=1
            ).tolist(),
            "graph_valid_fraction_from_bridge": np.mean(
                backward_valid, axis=1
            ).tolist(),
            "association_initial_error_m": _distribution(
                backward_association.initial_error_m
            ),
        }
        reverse_alignment_summary = {
            "frame_zero_inlier_rmse_m": float(
                reverse_initial_transform["inlier_rmse_m"]
            ),
            "bridge_icp_initial_residual_m": _distribution(
                np.asarray(bridge_icp["initial_nearest_residual_m"])
            ),
            "bridge_icp_final_residual_m": _distribution(
                np.asarray(bridge_icp["final_nearest_residual_m"])
            ),
            "bridge_icp_linear": np.asarray(bridge_icp["linear"]).tolist(),
            "bridge_icp_translation": np.asarray(
                bridge_icp["translation"]
            ).tolist(),
        }
    sampled_baseline = baseline[frame_indices, :surface_count]
    dense_error = dense_graph_error_by_frame(
        sampled_baseline,
        graph_observations,
        graph_valid,
        graph_reliability,
    )
    manual_audit = None
    if track_path.is_file():
        manual_audit = manual_track_association_audit(
            structure_points[:surface_count],
            graph_observations,
            graph_valid,
            np.asarray(_load_pickle(track_path), dtype=float),
            frame_indices,
        )
    train_selection = frame_indices < train_end
    future_selection = frame_indices >= train_end
    manual_error = (
        None
        if manual_audit is None
        else np.asarray(manual_audit["error_by_sampled_frame_m"], dtype=float)
    )

    output.mkdir(parents=True, exist_ok=True)
    association_path = output / "association.npz"
    np.savez_compressed(
        association_path,
        frame_indices=frame_indices.astype(np.int32),
        graph_observations=graph_observations,
        graph_valid=graph_valid,
        graph_reliability=graph_reliability,
        trajectory_indices=association.trajectory_indices,
        association_weights=association.weights,
        association_confidence=association.confidence,
        association_initial_error_m=association.initial_error_m,
        association_normalized_entropy=association.normalized_entropy,
        association_training_motion_error_m=(
            association.training_motion_error_m
        ),
        dense_positions=trajectories.positions,
        dense_valid=trajectories.valid,
        dense_step_error_m=trajectories.step_error_m,
        dense_seed_pixels_yx=trajectories.seed_pixels_yx,
        dense_pixel_indices=trajectories.pixel_indices,
        dense_camera_indices=trajectory_camera_indices,
        dense_graph_error_by_frame_m=dense_error,
        **backward_archive,
    )
    transform_summaries: dict[str, object] = {}
    for camera, transform in transforms.items():
        transform_summary = {
            key: value
            for key, value in transform.items()
            if key not in {"inlier_mask", "all_pair_residual_m"}
        }
        transform_summary.update(
            {
                "linear": np.asarray(transform["linear"]).tolist(),
                "translation": np.asarray(transform["translation"]).tolist(),
                "rotation": np.asarray(transform["rotation"]).tolist(),
                "all_pair_residual_m": _distribution(
                    np.asarray(transform["all_pair_residual_m"])
                ),
            }
        )
        transform_summaries[str(camera)] = transform_summary
    alignment_summary = {
        "view_count": len(transforms),
        "inlier_rmse_m": _distribution(
            np.asarray(
                [transform["inlier_rmse_m"] for transform in transforms.values()]
            )
        ),
        "by_camera": transform_summaries,
    }
    if reverse_alignment_summary is not None:
        alignment_summary["reverse"] = reverse_alignment_summary
    result: dict[str, object] = {
        "schema_version": 3,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case_path.name,
        "config": asdict(config),
        "contract": {
            "motioncrafter_representation": "world-coordinate dense point maps and forward 3D scene flow",
            "alignment": "per-view frame-zero same-pixel object-mask depth correspondences only",
            "identity_transport": "forward flow followed by gated nearest point-map transport and collision rejection",
            "graph_association": "frame-zero geometry plus automatic training-prefix track motion, persistence, spring-edge, and local-injectivity regularization",
            "future_use": (
                "offline reverse-video trajectories generate evaluation-only future observations and never enter PhysTwin fitting or prediction"
                if reverse_motioncrafter_npz_path is not None
                else "future MotionCrafter frames are evaluation observations only; association is frozen after the training prefix"
            ),
            "manual_tracks": "evaluation-only association audit; never used for alignment, transport, or graph assignment",
        },
        "software": {
            "motioncrafter_repository": MOTIONCRAFTER_REPOSITORY,
            "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        },
        "frame_indices": frame_indices.astype(int).tolist(),
        "train_end_frame": train_end,
        "association_frame_count": association_frame_count,
        "alignment": alignment_summary,
        "dense_transport": {
            "seed_count": int(trajectories.positions.shape[1]),
            "valid_fraction_by_sampled_frame": np.mean(
                trajectories.valid, axis=1
            ).tolist(),
            "step_error_m": _distribution(trajectories.step_error_m),
            "by_camera": {
                str(camera): {
                    "seed_count": int(camera_trajectories.positions.shape[1]),
                    "valid_fraction_by_sampled_frame": np.mean(
                        camera_trajectories.valid, axis=1
                    ).tolist(),
                    "step_error_m": _distribution(
                        camera_trajectories.step_error_m
                    ),
                }
                for camera, camera_trajectories in trajectories_by_camera.items()
            },
        },
        "backward_transport": backward_summary,
        "graph": {
            "surface_vertex_count": surface_count,
            "surface_spring_count": int(len(surface_springs)),
            "association_initial_error_m": _distribution(
                association.initial_error_m
            ),
            "association_confidence": _distribution(association.confidence),
            "normalized_entropy": _distribution(
                association.normalized_entropy
            ),
            "training_motion_error_m": _distribution(
                association.training_motion_error_m
            ),
            "valid_vertex_fraction_by_sampled_frame": np.mean(
                graph_valid, axis=1
            ).tolist(),
        },
        "released_dense_track_error": {
            "by_sampled_frame_m": dense_error.tolist(),
            "training_mean_m": _mean_on_frames(dense_error, train_selection),
            "future_mean_m": _mean_on_frames(dense_error, future_selection),
        },
        "manual_identity_audit": (
            {
                "available": False,
                "reason": "gt_track_3d.pkl is absent; automatic association is still complete",
            }
            if manual_audit is None
            else {
                "available": True,
                **manual_audit,
                "training_mean_m": _mean_on_frames(
                    manual_error, train_selection
                ),
                "future_mean_m": _mean_on_frames(
                    manual_error, future_selection
                ),
            }
        ),
        "inputs": {
            "final_data": {"path": str(final_path.resolve()), "sha256": _sha256(final_path)},
            "baseline": {"path": str(baseline_path.resolve()), "sha256": _sha256(baseline_path)},
            "optimal_params": {"path": str(optimal_path.resolve()), "sha256": _sha256(optimal_path)},
            "manual_tracks": (
                None
                if not track_path.is_file()
                else {
                    "path": str(track_path.resolve()),
                    "sha256": _sha256(track_path),
                }
            ),
            "split": {"path": str(split_path.resolve()), "sha256": _sha256(split_path)},
            "motioncrafter_views": {
                str(camera): {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
                for camera, path in sorted(view_paths.items())
            },
            "reverse_motioncrafter": (
                None
                if reverse_motioncrafter_npz_path is None
                else {
                    "path": str(Path(reverse_motioncrafter_npz_path).resolve()),
                    "sha256": _sha256(reverse_motioncrafter_npz_path),
                }
            ),
            "raw_case_dir": str(raw_path.resolve()),
        },
        "outputs": {
            "association_npz": str(association_path.resolve()),
            "association_npz_sha256": _sha256(association_path),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(summary_path.resolve())
    return result
