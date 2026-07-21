"""Target-free frame-zero geometry fallback for Deform360 twins.

The frozen Splatfacto path remains authoritative whenever its exported material
point cloud passes the existing admission check.  Only a failed point-cloud
check activates the multiview silhouette fallback in this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class FrameZeroInitializerConfig:
    """Source-developed geometry and admission choices for frame zero."""

    minimum_original_point_count: int = 128
    minimum_camera_count: int = 8
    cube_half_extent_m: float = 0.5
    voxel_resolution: int = 120
    consensus_fraction_of_peak: float = 0.55
    minimum_consensus_votes: int = 8
    maximum_mask_dilation_pixels: int = 5
    minimum_fallback_point_count: int = 128
    maximum_output_point_count: int = 10_000

    def __post_init__(self) -> None:
        _require(
            self.minimum_original_point_count >= 3,
            "original point minimum is too small",
        )
        _require(self.minimum_camera_count >= 3, "three cameras are required")
        _require(self.cube_half_extent_m > 0.0, "cube extent must be positive")
        _require(self.voxel_resolution >= 16, "voxel grid is too coarse")
        _require(
            0.0 < self.consensus_fraction_of_peak <= 1.0,
            "invalid consensus fraction",
        )
        _require(
            self.minimum_consensus_votes >= 3,
            "strict consensus requires three views",
        )
        _require(
            self.maximum_mask_dilation_pixels >= 0,
            "mask dilation must be nonnegative",
        )
        _require(
            self.minimum_fallback_point_count >= 3,
            "fallback point minimum is too small",
        )
        _require(
            self.maximum_output_point_count >= self.minimum_fallback_point_count,
            "output cap is below the fallback minimum",
        )


@dataclass(frozen=True)
class FrameZeroPointCloud:
    """One selected material point cloud and its target-free provenance."""

    points_m: np.ndarray
    colors: np.ndarray
    method: str
    diagnostics: Mapping[str, Any]


def original_point_cloud_admissible(
    points_m: np.ndarray,
    *,
    minimum_point_count: int = 128,
) -> bool:
    """Apply the frozen point-only admission check without changing values."""

    points = np.asarray(points_m)
    return bool(
        points.ndim == 2
        and points.shape[1:] == (3,)
        and len(points) >= minimum_point_count
        and np.all(np.isfinite(points))
    )


def _mask_2d(mask: np.ndarray, camera: str) -> np.ndarray:
    value = np.asarray(mask)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    _require(value.ndim == 2, f"mask for {camera} must be two-dimensional")
    return value.astype(bool, copy=False)


def _dilate_mask(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    if radius_pixels == 0:
        return mask
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - integration dependency
        raise RuntimeError("SciPy is required for visual-hull dilation") from error
    return ndimage.maximum_filter(
        mask.astype(np.uint8),
        size=2 * radius_pixels + 1,
        mode="constant",
    ).astype(bool)


def _project_points(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    k = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    _require(k.shape == (3, 3), "camera intrinsics must have shape (3,3)")
    _require(c2w.shape == (4, 4), "camera extrinsics must have shape (4,4)")
    world_to_camera = np.linalg.inv(c2w)
    camera_points = (
        points_world_m @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    depth = camera_points[:, 2]
    in_front = depth > 1e-6
    safe_depth = np.where(in_front, depth, 1.0)
    u = camera_points[:, 0] / safe_depth * k[0, 0] + k[0, 2]
    v = camera_points[:, 1] / safe_depth * k[1, 1] + k[1, 2]
    in_bounds = (
        in_front
        & (u >= 0.0)
        & (u < width)
        & (v >= 0.0)
        & (v < height)
    )
    columns = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
    rows = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
    return rows, columns, in_bounds


def multiview_mask_votes(
    points_world_m: np.ndarray,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    mask_dilation_radius_pixels: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Count independent calibrated mask support for each world-frame point."""

    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
        "points must have shape (N,3)",
    )
    _require(np.all(np.isfinite(points)), "points are non-finite")
    cameras = tuple(sorted(masks_by_camera))
    _require(cameras, "camera masks are empty")
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "camera intrinsics are incomplete",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "camera extrinsics are incomplete",
    )
    votes = np.zeros(len(points), dtype=np.uint16)
    per_camera: dict[str, Any] = {}
    for camera in cameras:
        mask = _dilate_mask(
            _mask_2d(masks_by_camera[camera], camera),
            mask_dilation_radius_pixels,
        )
        rows, columns, in_bounds = _project_points(
            points,
            intrinsics_by_camera[camera],
            camera_to_world_by_camera[camera],
            mask.shape,
        )
        supported = in_bounds & mask[rows, columns]
        votes += supported.astype(np.uint16)
        per_camera[camera] = {
            "in_bounds_point_count": int(np.count_nonzero(in_bounds)),
            "supported_point_count": int(np.count_nonzero(supported)),
        }
    return votes, {
        "camera_count": len(cameras),
        "mask_dilation_radius_pixels": mask_dilation_radius_pixels,
        "peak_vote_count": int(votes.max(initial=0)),
        "per_camera": per_camera,
    }


def _largest_component_surface(
    occupied: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - integration dependency
        raise RuntimeError("SciPy is required for visual-hull components") from error
    structure_26 = np.ones((3, 3, 3), dtype=bool)
    labels, component_count = ndimage.label(occupied, structure=structure_26)
    _require(component_count > 0, "visual hull has no connected component")
    component_sizes = np.bincount(labels.ravel())[1:]
    largest_label = int(np.argmax(component_sizes)) + 1
    largest = labels == largest_label
    structure_6 = ndimage.generate_binary_structure(3, 1)
    interior = ndimage.binary_erosion(
        largest,
        structure=structure_6,
        border_value=0,
    )
    surface = largest & ~interior
    return surface, {
        "component_count": int(component_count),
        "largest_component_point_count": int(component_sizes[largest_label - 1]),
        "largest_component_fraction": float(
            component_sizes[largest_label - 1] / np.count_nonzero(occupied)
        ),
        "surface_point_count_before_subsampling": int(np.count_nonzero(surface)),
    }


def _morton_codes(indices: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(indices, dtype=np.uint64)
    codes = np.zeros(len(coordinates), dtype=np.uint64)
    for bit in range(21):
        shift = np.uint64(bit)
        for axis in range(3):
            values = (coordinates[:, axis] >> shift) & np.uint64(1)
            codes |= values << np.uint64(3 * bit + axis)
    return codes


def _spatial_subsample(
    points: np.ndarray,
    grid_indices: np.ndarray,
    maximum_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= maximum_count:
        return points, grid_indices
    order = np.argsort(_morton_codes(grid_indices), kind="stable")
    selected_in_order = np.linspace(
        0,
        len(order) - 1,
        maximum_count,
        dtype=np.int64,
    )
    selected = order[selected_in_order]
    return points[selected], grid_indices[selected]


def _colorize_points(
    points_world_m: np.ndarray,
    masks_by_camera: Mapping[str, np.ndarray],
    images_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    mask_dilation_radius_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    color_sum = np.zeros((len(points_world_m), 3), dtype=np.float64)
    color_count = np.zeros(len(points_world_m), dtype=np.int32)
    for camera in sorted(masks_by_camera):
        _require(camera in images_by_camera, f"image for {camera} is missing")
        mask = _dilate_mask(
            _mask_2d(masks_by_camera[camera], camera),
            mask_dilation_radius_pixels,
        )
        image = np.asarray(images_by_camera[camera])
        _require(
            image.ndim == 3 and image.shape[:2] == mask.shape and image.shape[2] >= 3,
            f"image for {camera} is incompatible with its mask",
        )
        rows, columns, in_bounds = _project_points(
            points_world_m,
            intrinsics_by_camera[camera],
            camera_to_world_by_camera[camera],
            mask.shape,
        )
        supported = in_bounds & mask[rows, columns]
        sampled = np.asarray(image[rows[supported], columns[supported], :3])
        if np.issubdtype(sampled.dtype, np.integer):
            sampled = sampled.astype(np.float64) / np.iinfo(sampled.dtype).max
        else:
            sampled = np.clip(sampled.astype(np.float64), 0.0, 1.0)
        color_sum[supported] += sampled
        color_count[supported] += 1
    _require(np.all(color_count > 0), "fallback points lack color support")
    colors = color_sum / color_count[:, None]
    return colors.astype(np.float32), {
        "minimum_color_view_count": int(color_count.min()),
        "median_color_view_count": float(np.median(color_count)),
        "maximum_color_view_count": int(color_count.max()),
    }


def build_strict_multiview_surface(
    masks_by_camera: Mapping[str, np.ndarray],
    images_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    config: FrameZeroInitializerConfig | None = None,
) -> FrameZeroPointCloud:
    """Build a strict, connected visual-hull surface from frame zero only."""

    cfg = config or FrameZeroInitializerConfig()
    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= cfg.minimum_camera_count, "too few frame-zero cameras")
    _require(
        all(camera in images_by_camera for camera in cameras),
        "camera images are incomplete",
    )
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "camera intrinsics are incomplete",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "camera extrinsics are incomplete",
    )
    centers = np.stack(
        [
            np.asarray(camera_to_world_by_camera[camera], dtype=np.float64)[:3, 3]
            for camera in cameras
        ]
    )
    grid_center = np.mean(centers, axis=0)
    axis = np.linspace(
        -cfg.cube_half_extent_m,
        cfg.cube_half_extent_m,
        cfg.voxel_resolution,
        dtype=np.float64,
    )
    grid = (
        np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        + grid_center
    )
    attempts: list[dict[str, Any]] = []
    selected_surface: np.ndarray | None = None
    selected_indices: np.ndarray | None = None
    selected_radius = -1
    for radius in range(cfg.maximum_mask_dilation_pixels + 1):
        votes, vote_diagnostics = multiview_mask_votes(
            grid,
            masks_by_camera,
            intrinsics_by_camera,
            camera_to_world_by_camera,
            mask_dilation_radius_pixels=radius,
        )
        peak = int(votes.max(initial=0))
        required = max(
            cfg.minimum_consensus_votes,
            int(math.ceil(cfg.consensus_fraction_of_peak * peak)),
        )
        accepted = votes >= required if peak >= required else np.zeros_like(votes, dtype=bool)
        attempt: dict[str, Any] = {
            "mask_dilation_radius_pixels": radius,
            "required_vote_count": required,
            "accepted_voxel_count": int(np.count_nonzero(accepted)),
            "voting": vote_diagnostics,
        }
        if np.any(accepted):
            surface_mask, component = _largest_component_surface(
                accepted.reshape((cfg.voxel_resolution,) * 3)
            )
            attempt["components"] = component
            surface_indices = np.argwhere(surface_mask)
            surface_points = grid[surface_mask.ravel()]
            if len(surface_points) >= cfg.minimum_fallback_point_count:
                selected_surface = surface_points
                selected_indices = surface_indices
                selected_radius = radius
                attempts.append(attempt)
                break
        attempts.append(attempt)
    _require(
        selected_surface is not None and selected_indices is not None,
        "strict multiview hull did not produce enough connected surface points",
    )
    points, retained_indices = _spatial_subsample(
        selected_surface,
        selected_indices,
        cfg.maximum_output_point_count,
    )
    colors, color_diagnostics = _colorize_points(
        points,
        masks_by_camera,
        images_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        mask_dilation_radius_pixels=selected_radius,
    )
    _require(
        original_point_cloud_admissible(
            points,
            minimum_point_count=cfg.minimum_fallback_point_count,
        ),
        "fallback point cloud failed admission",
    )
    _require(np.all(np.isfinite(colors)), "fallback colors are non-finite")
    spacing = float(axis[1] - axis[0])
    diagnostics: dict[str, Any] = {
        "config": asdict(cfg),
        "camera_count": len(cameras),
        "grid_center_world_m": grid_center.tolist(),
        "grid_spacing_m": spacing,
        "grid_point_count": len(grid),
        "attempts": attempts,
        "selected_mask_dilation_radius_pixels": selected_radius,
        "surface_point_count_before_subsampling": len(selected_surface),
        "output_point_count": len(points),
        "output_was_subsampled": len(points) < len(selected_surface),
        "retained_grid_index_minimum": retained_indices.min(axis=0).tolist(),
        "retained_grid_index_maximum": retained_indices.max(axis=0).tolist(),
        "color_support": color_diagnostics,
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    return FrameZeroPointCloud(
        points_m=points.astype(np.float32),
        colors=colors,
        method="strict-multiview-visual-hull-surface",
        diagnostics=diagnostics,
    )


def select_frame_zero_point_cloud(
    original_points_m: np.ndarray,
    original_colors: np.ndarray,
    fallback_builder: Callable[[], FrameZeroPointCloud],
    *,
    config: FrameZeroInitializerConfig | None = None,
) -> FrameZeroPointCloud:
    """Preserve an admitted original cloud exactly or lazily build a fallback."""

    cfg = config or FrameZeroInitializerConfig()
    points = np.asarray(original_points_m)
    colors = np.asarray(original_colors)
    if original_point_cloud_admissible(
        points,
        minimum_point_count=cfg.minimum_original_point_count,
    ):
        return FrameZeroPointCloud(
            points_m=points,
            colors=colors,
            method="original-splat",
            diagnostics={
                "original_point_count": len(points),
                "original_points_finite": True,
                "original_colors_shape_matches": colors.shape == points.shape,
                "original_colors_finite": bool(np.all(np.isfinite(colors))),
                "fallback_evaluated": False,
            },
        )
    fallback = fallback_builder()
    _require(
        fallback.method == "strict-multiview-visual-hull-surface",
        "unexpected fallback method",
    )
    _require(
        original_point_cloud_admissible(
            fallback.points_m,
            minimum_point_count=cfg.minimum_fallback_point_count,
        ),
        "fallback point cloud failed admission",
    )
    _require(
        fallback.colors.shape == fallback.points_m.shape
        and np.all(np.isfinite(fallback.colors)),
        "fallback colors failed admission",
    )
    return FrameZeroPointCloud(
        points_m=fallback.points_m,
        colors=fallback.colors,
        method=fallback.method,
        diagnostics={
            "original_point_count": len(points) if points.ndim >= 1 else 0,
            "original_points_finite": bool(np.all(np.isfinite(points))),
            "fallback_evaluated": True,
            "fallback": dict(fallback.diagnostics),
        },
    )


__all__ = [
    "FrameZeroInitializerConfig",
    "FrameZeroPointCloud",
    "build_strict_multiview_surface",
    "multiview_mask_votes",
    "original_point_cloud_admissible",
    "select_frame_zero_point_cloud",
]
