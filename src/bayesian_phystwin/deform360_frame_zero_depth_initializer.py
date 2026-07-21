"""Frame-zero depth support for the frozen Deform360 visual-hull fallback.

Rendered depth is used only after the original Splatfacto material cloud fails
the frozen point-count admission check.  It prunes the separately versioned
strict visual-hull surface; it is not treated as an independent modality or as
additional probabilistic evidence because it is rendered from the same Splat.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_frame_zero_initializer import (
    FrameZeroInitializerConfig,
    FrameZeroPointCloud,
    build_strict_multiview_surface,
    original_point_cloud_admissible,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class DepthSupportedFrameZeroInitializerConfig:
    """Source-developed choices for metric-depth pruning of the v1 hull."""

    visual_hull: FrameZeroInitializerConfig = field(
        default_factory=FrameZeroInitializerConfig
    )
    depth_tolerance_m: float = 0.05
    minimum_depth_camera_count: int = 8
    minimum_depth_support_views: int = 1
    minimum_depth_supported_point_count: int = 128

    def __post_init__(self) -> None:
        _require(self.depth_tolerance_m > 0.0, "depth tolerance must be positive")
        _require(
            self.minimum_depth_camera_count >= 1,
            "at least one depth camera is required",
        )
        _require(
            self.minimum_depth_support_views >= 1,
            "at least one supporting depth view is required",
        )
        _require(
            self.minimum_depth_support_views <= self.minimum_depth_camera_count,
            "support-view minimum exceeds the depth-camera minimum",
        )
        _require(
            self.minimum_depth_supported_point_count >= 3,
            "depth-supported point minimum is too small",
        )
        _require(
            self.minimum_depth_supported_point_count
            <= self.visual_hull.maximum_output_point_count,
            "depth-supported point minimum exceeds the visual-hull cap",
        )


def _depth_2d(depth_m: np.ndarray, camera: str) -> np.ndarray:
    value = np.asarray(depth_m)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    _require(value.ndim == 2, f"depth for {camera} must be two-dimensional")
    return value.astype(np.float64, copy=False)


def _project_points_with_depth(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return rows, columns, in_bounds, depth


def metric_depth_support_counts(
    points_world_m: np.ndarray,
    depth_by_camera_m: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    depth_tolerance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Count frame-zero rendered-depth support for each world-frame point."""

    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
        "points must have shape (N,3)",
    )
    _require(np.all(np.isfinite(points)), "points are non-finite")
    _require(depth_tolerance_m > 0.0, "depth tolerance must be positive")
    cameras = tuple(sorted(depth_by_camera_m))
    _require(cameras, "depth maps are empty")
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "camera intrinsics are incomplete",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "camera extrinsics are incomplete",
    )

    support_count = np.zeros(len(points), dtype=np.uint16)
    visible_depth_count = np.zeros(len(points), dtype=np.uint16)
    informative_camera_count = 0
    per_camera: dict[str, Any] = {}
    for camera in cameras:
        depth_map = _depth_2d(depth_by_camera_m[camera], camera)
        valid_pixel = np.isfinite(depth_map) & (depth_map > 0.0)
        informative = bool(np.any(valid_pixel))
        informative_camera_count += int(informative)
        rows, columns, in_bounds, projected_depth = _project_points_with_depth(
            points,
            intrinsics_by_camera[camera],
            camera_to_world_by_camera[camera],
            depth_map.shape,
        )
        sampled_depth = depth_map[rows, columns]
        has_depth = in_bounds & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        supported = has_depth & (
            np.abs(projected_depth - sampled_depth) <= depth_tolerance_m
        )
        visible_depth_count += has_depth.astype(np.uint16)
        support_count += supported.astype(np.uint16)
        residuals = np.abs(projected_depth[supported] - sampled_depth[supported])
        per_camera[camera] = {
            "valid_depth_pixel_count": int(np.count_nonzero(valid_pixel)),
            "in_bounds_point_count": int(np.count_nonzero(in_bounds)),
            "point_count_with_sampled_depth": int(np.count_nonzero(has_depth)),
            "supported_point_count": int(np.count_nonzero(supported)),
            "supported_residual_median_m": (
                float(np.median(residuals)) if len(residuals) else None
            ),
        }
    return support_count, {
        "camera_count": len(cameras),
        "informative_camera_count": informative_camera_count,
        "depth_tolerance_m": depth_tolerance_m,
        "point_count_with_any_sampled_depth": int(
            np.count_nonzero(visible_depth_count)
        ),
        "maximum_sampled_depth_view_count": int(
            visible_depth_count.max(initial=0)
        ),
        "maximum_support_view_count": int(support_count.max(initial=0)),
        "per_camera": per_camera,
    }


def build_depth_supported_multiview_surface(
    masks_by_camera: Mapping[str, np.ndarray],
    images_by_camera: Mapping[str, np.ndarray],
    depth_by_camera_m: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    config: DepthSupportedFrameZeroInitializerConfig | None = None,
) -> FrameZeroPointCloud:
    """Prune the frozen strict visual-hull surface with metric depth."""

    cfg = config or DepthSupportedFrameZeroInitializerConfig()
    hull = build_strict_multiview_surface(
        masks_by_camera,
        images_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        config=cfg.visual_hull,
    )
    return filter_surface_with_metric_depth(
        hull,
        depth_by_camera_m,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        visual_hull_cameras=masks_by_camera,
        config=cfg,
    )


def filter_surface_with_metric_depth(
    visual_hull: FrameZeroPointCloud,
    depth_by_camera_m: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    visual_hull_cameras: Mapping[str, Any],
    config: DepthSupportedFrameZeroInitializerConfig | None = None,
) -> FrameZeroPointCloud:
    """Apply the v2 depth gate to an already constructed frozen v1 surface."""

    cfg = config or DepthSupportedFrameZeroInitializerConfig()
    _require(
        visual_hull.method == "strict-multiview-visual-hull-surface",
        "depth pruning requires the frozen strict visual-hull surface",
    )
    depth_cameras = tuple(sorted(depth_by_camera_m))
    _require(
        len(depth_cameras) >= cfg.minimum_depth_camera_count,
        "too few frame-zero depth cameras",
    )
    _require(
        set(depth_cameras).issubset(visual_hull_cameras),
        "depth cameras are not a subset of the visual-hull cameras",
    )
    support, depth_diagnostics = metric_depth_support_counts(
        visual_hull.points_m,
        depth_by_camera_m,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        depth_tolerance_m=cfg.depth_tolerance_m,
    )
    _require(
        depth_diagnostics["informative_camera_count"]
        >= cfg.minimum_depth_camera_count,
        "too few informative frame-zero depth cameras",
    )
    retained = support >= cfg.minimum_depth_support_views
    _require(
        int(np.count_nonzero(retained)) >= cfg.minimum_depth_supported_point_count,
        "metric depth did not support enough visual-hull points",
    )
    points = visual_hull.points_m[retained]
    colors = visual_hull.colors[retained]
    diagnostics: dict[str, Any] = {
        "config": asdict(cfg),
        "visual_hull_method": visual_hull.method,
        "visual_hull_point_count": len(visual_hull.points_m),
        "visual_hull_diagnostics": dict(visual_hull.diagnostics),
        "depth_support": depth_diagnostics,
        "minimum_depth_support_views": cfg.minimum_depth_support_views,
        "output_point_count": len(points),
        "retained_fraction": float(len(points) / len(visual_hull.points_m)),
        "retained_minimum_support_views": int(support[retained].min()),
        "retained_median_support_views": float(np.median(support[retained])),
        "retained_maximum_support_views": int(support[retained].max()),
        "depth_is_independent_modality": False,
        "depth_origin": "frame-zero Splatfacto expected-depth render",
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    return FrameZeroPointCloud(
        points_m=points,
        colors=colors,
        method="depth-supported-strict-multiview-surface",
        diagnostics=diagnostics,
    )


def select_depth_supported_frame_zero_point_cloud(
    original_points_m: np.ndarray,
    original_colors: np.ndarray,
    fallback_builder: Callable[[], FrameZeroPointCloud],
    *,
    config: DepthSupportedFrameZeroInitializerConfig | None = None,
) -> FrameZeroPointCloud:
    """Preserve an admitted original cloud or lazily build the v2 fallback."""

    cfg = config or DepthSupportedFrameZeroInitializerConfig()
    points = np.asarray(original_points_m)
    colors = np.asarray(original_colors)
    if original_point_cloud_admissible(
        points,
        minimum_point_count=cfg.visual_hull.minimum_original_point_count,
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
        fallback.method == "depth-supported-strict-multiview-surface",
        "unexpected fallback method",
    )
    _require(
        original_point_cloud_admissible(
            fallback.points_m,
            minimum_point_count=cfg.minimum_depth_supported_point_count,
        ),
        "depth-supported fallback point cloud failed admission",
    )
    _require(
        fallback.colors.shape == fallback.points_m.shape
        and np.all(np.isfinite(fallback.colors)),
        "depth-supported fallback colors failed admission",
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
    "DepthSupportedFrameZeroInitializerConfig",
    "build_depth_supported_multiview_surface",
    "filter_surface_with_metric_depth",
    "metric_depth_support_counts",
    "select_depth_supported_frame_zero_point_cloud",
]
