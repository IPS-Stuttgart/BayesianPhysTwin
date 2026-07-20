"""Exact batched accelerator for the frozen Deform360 camera selector.

The frozen selector scores every camera subset lexicographically.  Its direct
Python implementation is useful as a specification, but evaluating all
``C(32, 8)`` subsets also recomputes geometric scores that cannot affect the
result.  This module preserves the same subset order and score while batching
the integer prefix of the score in NumPy and evaluating ray angles only for
subsets tied on that prefix.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    _maximum_ray_angle_degrees,
)
from .phystwin_online_belief import deterministic_farthest_point_ids


FROZEN_RAW_CAMERA_BUILDER_SHA256 = (
    "2c24e587e9acd1dda589363240a81b268863fbe808ce05c58f8a5b70f12f76c3"
)
DEFAULT_COMBINATION_BATCH_SIZE = 65_536


def _combination_batches(
    item_count: int,
    selection_count: int,
    *,
    batch_size: int,
) -> Iterator[np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    iterator = itertools.combinations(range(item_count), selection_count)
    while True:
        rows = tuple(itertools.islice(iterator, batch_size))
        if not rows:
            return
        yield np.asarray(rows, dtype=np.int64)


def _packed_primary_scores(
    counts: np.ndarray,
    *,
    minimum_initial_view_count: int,
    selected_camera_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_count = counts.shape[1]
    supported = np.count_nonzero(
        counts >= minimum_initial_view_count,
        axis=1,
    )
    supported_by_three = np.count_nonzero(counts >= 3, axis=1)
    total = np.sum(counts, axis=1)
    packed = (supported * (center_count + 1) + supported_by_three) * (
        center_count * selected_camera_count + 1
    ) + total
    return packed, supported, supported_by_three, total


def _pair_angle_table(
    points: np.ndarray,
    center_ids: np.ndarray,
    camera_origins_m: np.ndarray,
) -> np.ndarray:
    camera_count = len(camera_origins_m)
    result = np.zeros(
        (len(center_ids), camera_count, camera_count),
        dtype=np.float64,
    )
    for center_index, point_id in enumerate(center_ids):
        for first, second in itertools.combinations(range(camera_count), 2):
            result[center_index, first, second] = _maximum_ray_angle_degrees(
                points[point_id],
                (first, second),
                camera_origins_m,
            )
    return result


def _candidate_angle_scores(
    subsets: np.ndarray,
    counts: np.ndarray,
    center_support: np.ndarray,
    pair_angles: np.ndarray,
) -> np.ndarray:
    if not len(subsets):
        return np.empty(0, dtype=np.float64)
    selected_camera_count = subsets.shape[1]
    position_pairs = np.asarray(
        tuple(itertools.combinations(range(selected_camera_count), 2)),
        dtype=np.int64,
    )
    left = subsets[:, position_pairs[:, 0]]
    right = subsets[:, position_pairs[:, 1]]
    center_angles = np.zeros(
        (len(subsets), len(center_support)),
        dtype=np.float64,
    )
    for center_index, support in enumerate(center_support):
        valid_pairs = support[left] & support[right]
        values = pair_angles[center_index, left, right]
        center_angles[:, center_index] = np.max(
            np.where(valid_pairs, values, 0.0),
            axis=1,
        )

    included = counts >= 2
    result = np.zeros(len(subsets), dtype=np.float64)
    has_angle = np.any(included, axis=1)
    if np.any(has_angle):
        masked = np.where(included[has_angle], center_angles[has_angle], np.nan)
        result[has_angle] = np.nanmedian(masked, axis=1)
    return result


def select_exact_camera_subset(
    frame_zero_points_m: np.ndarray,
    center_ids: np.ndarray,
    support: np.ndarray,
    camera_origins_m: np.ndarray,
    *,
    selected_camera_count: int,
    minimum_initial_view_count: int,
    batch_size: int = DEFAULT_COMBINATION_BATCH_SIZE,
) -> tuple[tuple[int, ...], tuple[int, int, int, float]]:
    """Return the exact frozen camera subset and lexicographic score."""

    points = np.asarray(frame_zero_points_m, dtype=float)
    centers = np.asarray(center_ids, dtype=np.int64)
    supported = np.asarray(support, dtype=bool)
    origins = np.asarray(camera_origins_m, dtype=float)
    camera_count = supported.shape[1]
    if supported.shape[0] != len(points):
        raise ValueError("support shape differs from points")
    if origins.shape != (camera_count, 3):
        raise ValueError("camera origins have invalid shape")
    if not 1 <= selected_camera_count <= camera_count:
        raise ValueError("selected camera count is invalid")
    if np.any(centers < 0) or np.any(centers >= len(points)):
        raise ValueError("center id is outside the point array")

    center_support = supported[centers]
    pair_angles = _pair_angle_table(points, centers, origins)
    best_primary = -1
    best_angle = -np.inf
    best_subset: tuple[int, ...] | None = None
    best_score: tuple[int, int, int, float] | None = None

    for subsets in _combination_batches(
        camera_count,
        selected_camera_count,
        batch_size=batch_size,
    ):
        counts = np.sum(center_support[:, subsets], axis=2).T
        packed, first, second, total = _packed_primary_scores(
            counts,
            minimum_initial_view_count=minimum_initial_view_count,
            selected_camera_count=selected_camera_count,
        )
        batch_primary = int(np.max(packed))
        if batch_primary < best_primary:
            continue
        if batch_primary > best_primary:
            best_primary = batch_primary
            best_angle = -np.inf
            best_subset = None
            best_score = None

        candidate_indices = np.flatnonzero(packed == best_primary)
        candidate_subsets = subsets[candidate_indices]
        candidate_counts = counts[candidate_indices]
        angle_scores = _candidate_angle_scores(
            candidate_subsets,
            candidate_counts,
            center_support,
            pair_angles,
        )
        local_index = int(np.argmax(angle_scores))
        local_angle = float(angle_scores[local_index])
        if best_subset is not None and local_angle <= best_angle:
            continue
        source_index = int(candidate_indices[local_index])
        best_angle = local_angle
        best_subset = tuple(int(value) for value in subsets[source_index])
        best_score = (
            int(first[source_index]),
            int(second[source_index]),
            int(total[source_index]),
            local_angle,
        )

    if best_subset is None or best_score is None:
        raise AssertionError("camera selection produced no subset")
    return best_subset, best_score


def select_frame_zero_observation_plan_exact_accelerated(
    frame_zero_points_m: np.ndarray,
    cameras: Sequence[str],
    support: np.ndarray,
    projected_pixels: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, Any],
    *,
    config: RawCameraObservationConfig,
) -> dict[str, Any]:
    """Execute the frozen observation plan with exact batched subset scoring."""

    points = np.asarray(frame_zero_points_m, dtype=float)
    camera_names = tuple(cameras)
    supported = np.asarray(support, dtype=bool)
    if supported.shape != (len(points), len(camera_names)):
        raise ValueError("support shape differs from points/cameras")
    if len(camera_names) < config.selected_camera_count:
        raise ValueError("fewer cameras than the fixed selected-camera count")
    origins = np.stack(
        [np.asarray(extrinsics[camera], dtype=float)[:3, 3] for camera in camera_names]
    )
    candidate_ids: list[int] = []
    for point_id in range(len(points)):
        views = np.flatnonzero(supported[point_id])
        if len(views) < config.minimum_initial_view_count:
            continue
        if (
            _maximum_ray_angle_degrees(points[point_id], views, origins)
            < config.minimum_ray_angle_degrees
        ):
            continue
        candidate_ids.append(point_id)
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    if len(candidates) < config.center_count:
        raise ValueError("too few multiview-visible frame-zero candidates")
    centers = deterministic_farthest_point_ids(
        points,
        candidates,
        config.center_count,
    )
    best_subset, best_score = select_exact_camera_subset(
        points,
        centers,
        supported,
        origins,
        selected_camera_count=config.selected_camera_count,
        minimum_initial_view_count=config.minimum_initial_view_count,
    )
    selected_cameras = tuple(camera_names[index] for index in best_subset)
    query_ids = {
        camera: centers[supported[centers, camera_names.index(camera)]].astype(np.int64)
        for camera in selected_cameras
    }
    query_pixels = {
        camera: np.asarray(projected_pixels[camera], dtype=float)[query_ids[camera]]
        for camera in selected_cameras
    }
    return {
        "candidate_ids": candidates,
        "center_ids": centers,
        "selected_cameras": selected_cameras,
        "selected_camera_indices": np.asarray(best_subset, dtype=np.int64),
        "selection_score": best_score,
        "query_ids": query_ids,
        "query_pixels": query_pixels,
        "support": supported,
        "camera_names": camera_names,
    }


def frozen_builder_path() -> Path:
    """Return the source file whose direct selector defines exact behavior."""

    from . import deform360_raw_camera_observation as raw_camera

    return Path(raw_camera.__file__).resolve()


__all__ = [
    "DEFAULT_COMBINATION_BATCH_SIZE",
    "FROZEN_RAW_CAMERA_BUILDER_SHA256",
    "frozen_builder_path",
    "select_exact_camera_subset",
    "select_frame_zero_observation_plan_exact_accelerated",
]
