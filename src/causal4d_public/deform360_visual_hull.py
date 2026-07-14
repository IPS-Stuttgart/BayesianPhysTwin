"""Adaptive multiview silhouette hulls for the thin Deform360 rope."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class AdaptiveRopeHullConfig:
    local_voxel_size_m: float = 0.004
    initial_margin_m: float = 0.05
    expansion_factor: float = 1.6
    maximum_expansion_attempts: int = 3
    maximum_grid_point_count: int = 1_500_000
    consensus_fraction_of_peak: float = 0.55
    minimum_consensus_votes: int = 8
    minimum_hull_point_count: int = 64

    def __post_init__(self) -> None:
        _require(self.local_voxel_size_m > 0.0, "voxel size must be positive")
        _require(self.initial_margin_m > 0.0, "hull margin must be positive")
        _require(self.expansion_factor > 1.0, "expansion factor must exceed one")
        _require(
            self.maximum_expansion_attempts >= 1,
            "at least one hull attempt is required",
        )
        _require(
            self.maximum_grid_point_count >= 1_000,
            "maximum grid size is too small",
        )
        _require(
            0.0 < self.consensus_fraction_of_peak <= 1.0,
            "invalid consensus fraction",
        )
        _require(self.minimum_consensus_votes >= 2, "consensus needs two views")
        _require(self.minimum_hull_point_count >= 8, "minimum hull size is too small")


def regular_grid_in_bounds(
    minimum_world_m: np.ndarray,
    maximum_world_m: np.ndarray,
    *,
    requested_voxel_size_m: float,
    maximum_point_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a bounded regular grid, coarsening only when the cap requires it."""

    minimum = np.asarray(minimum_world_m, dtype=np.float64)
    maximum = np.asarray(maximum_world_m, dtype=np.float64)
    _require(minimum.shape == maximum.shape == (3,), "grid bounds must have shape (3,)")
    _require(
        np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)),
        "grid bounds are non-finite",
    )
    _require(np.all(maximum > minimum), "grid maximum must exceed its minimum")
    _require(requested_voxel_size_m > 0.0, "voxel size must be positive")
    _require(maximum_point_count >= 8, "grid cap is too small")
    extent = maximum - minimum
    voxel_size = float(requested_voxel_size_m)
    shape = np.floor(extent / voxel_size).astype(np.int64) + 1
    point_count = int(np.prod(shape, dtype=np.int64))
    if point_count > maximum_point_count:
        scale = (point_count / maximum_point_count) ** (1.0 / 3.0)
        voxel_size *= scale * 1.001
        shape = np.floor(extent / voxel_size).astype(np.int64) + 1
        point_count = int(np.prod(shape, dtype=np.int64))
        while point_count > maximum_point_count:
            voxel_size *= 1.01
            shape = np.floor(extent / voxel_size).astype(np.int64) + 1
            point_count = int(np.prod(shape, dtype=np.int64))
    axes = [
        np.linspace(minimum[axis], maximum[axis], int(shape[axis])) for axis in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    actual_spacing = [
        float(axis[1] - axis[0]) if len(axis) > 1 else 0.0 for axis in axes
    ]
    return grid, {
        "bounds_minimum_world_m": minimum.tolist(),
        "bounds_maximum_world_m": maximum.tolist(),
        "requested_voxel_size_m": requested_voxel_size_m,
        "effective_axis_spacing_m": actual_spacing,
        "grid_shape": shape.tolist(),
        "grid_point_count": len(grid),
        "coarsened_for_grid_cap": voxel_size > requested_voxel_size_m,
    }


def carve_candidate_points(
    candidate_points_world_m: np.ndarray,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    consensus_fraction_of_peak: float,
    minimum_consensus_votes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep candidate points supported by a peak-relative multiview consensus."""

    points = np.asarray(candidate_points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
        "candidate points must have shape (N,3)",
    )
    _require(np.all(np.isfinite(points)), "candidate points are non-finite")
    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= 2, "at least two camera masks are required")
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "camera intrinsics are incomplete",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "camera extrinsics are incomplete",
    )
    _require(
        0.0 < consensus_fraction_of_peak <= 1.0,
        "invalid consensus fraction",
    )
    _require(minimum_consensus_votes >= 2, "consensus needs two views")
    hits = []
    in_bounds_counts = []
    for camera in cameras:
        mask = np.asarray(masks_by_camera[camera], dtype=bool)
        _require(mask.ndim == 2, f"mask for {camera} must be two-dimensional")
        height, width = mask.shape
        intrinsics = np.asarray(intrinsics_by_camera[camera], dtype=np.float64)
        camera_to_world = np.asarray(
            camera_to_world_by_camera[camera], dtype=np.float64
        )
        _require(intrinsics.shape == (3, 3), f"invalid intrinsics for {camera}")
        _require(camera_to_world.shape == (4, 4), f"invalid extrinsics for {camera}")
        world_to_camera = np.linalg.inv(camera_to_world)
        camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = camera_points[:, 2]
        in_front = depth > 1e-6
        safe_depth = np.where(in_front, depth, 1.0)
        u = camera_points[:, 0] / safe_depth * intrinsics[0, 0] + intrinsics[0, 2]
        v = camera_points[:, 1] / safe_depth * intrinsics[1, 1] + intrinsics[1, 2]
        in_bounds = in_front & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        columns = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
        rows = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
        hits.append(in_bounds & mask[rows, columns])
        in_bounds_counts.append(int(np.count_nonzero(in_bounds)))
    camera_hits = np.asarray(hits, dtype=bool)
    votes = camera_hits.sum(axis=0)
    peak = int(votes.max(initial=0))
    required = max(
        minimum_consensus_votes,
        int(math.ceil(consensus_fraction_of_peak * peak)),
    )
    accepted = (
        votes >= required if peak >= required else np.zeros(len(points), dtype=bool)
    )
    hull = points[accepted]
    return hull, {
        "candidate_point_count": len(points),
        "camera_count": len(cameras),
        "peak_vote_count": peak,
        "required_vote_count": required,
        "hull_point_count": len(hull),
        "accepted_candidate_fraction": float(np.mean(accepted)),
        "per_camera_in_bounds_count": {
            camera: count
            for camera, count in zip(cameras, in_bounds_counts, strict=True)
        },
    }


def adaptive_rope_visual_hull(
    prior_centerline_world_m: np.ndarray,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    config: AdaptiveRopeHullConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Carve a high-resolution local hull around the previous rope state."""

    cfg = config or AdaptiveRopeHullConfig()
    prior = np.asarray(prior_centerline_world_m, dtype=np.float64)
    _require(
        prior.ndim == 2 and prior.shape[1] == 3 and len(prior) >= 2,
        "prior centerline must have shape (N,3)",
    )
    _require(np.all(np.isfinite(prior)), "prior centerline is non-finite")
    attempts = []
    hull = np.empty((0, 3), dtype=np.float64)
    margin = cfg.initial_margin_m
    for attempt in range(cfg.maximum_expansion_attempts):
        grid, grid_diagnostics = regular_grid_in_bounds(
            np.min(prior, axis=0) - margin,
            np.max(prior, axis=0) + margin,
            requested_voxel_size_m=cfg.local_voxel_size_m,
            maximum_point_count=cfg.maximum_grid_point_count,
        )
        hull, carve_diagnostics = carve_candidate_points(
            grid,
            masks_by_camera,
            intrinsics_by_camera,
            camera_to_world_by_camera,
            consensus_fraction_of_peak=cfg.consensus_fraction_of_peak,
            minimum_consensus_votes=cfg.minimum_consensus_votes,
        )
        attempts.append(
            {
                "attempt": attempt,
                "margin_m": margin,
                "grid": grid_diagnostics,
                "carving": carve_diagnostics,
            }
        )
        if len(hull) >= cfg.minimum_hull_point_count:
            break
        margin *= cfg.expansion_factor
    _require(
        len(hull) >= cfg.minimum_hull_point_count,
        "adaptive visual hull did not reach the minimum point count",
    )
    quantiles = np.percentile(hull, [1.0, 50.0, 99.0], axis=0)
    return hull, {
        "parameters": asdict(cfg),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "final_hull_point_count": len(hull),
        "final_hull_world_m": {
            "q01": quantiles[0].tolist(),
            "median": quantiles[1].tolist(),
            "q99": quantiles[2].tolist(),
            "q01_to_q99_span": (quantiles[2] - quantiles[0]).tolist(),
        },
    }


__all__ = [
    "AdaptiveRopeHullConfig",
    "adaptive_rope_visual_hull",
    "carve_candidate_points",
    "regular_grid_in_bounds",
]
