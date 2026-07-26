"""Render-to-real AllTracker observations with conservative multiview geometry.

The observation begins at a material point projected from a PhysTwin render.
AllTracker supplies a direct two-frame image correspondence to the real prefix
frame. Confidence, cycle consistency, render support, mask support, and camera
geometry determine validity before any physical-state innovation is formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RenderedAllTrackerConfig:
    """Frozen residual-independent support and covariance settings."""

    minimum_quality: float = 0.5
    maximum_cycle_error_px: float = 5.0
    maximum_reprojection_error_px: float = 3.0
    minimum_camera_count: int = 2
    redundant_camera_count: int = 3
    pixel_noise_std: float = 2.0
    prior_std_m: float = 0.10
    shared_bias_std_m: float = 0.005
    two_view_extra_std_m: float = 0.010
    minimum_ray_angle_degrees: float = 0.5
    duplicate_center_tolerance_m: float = 1e-9
    duplicate_rotation_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        _require(
            0.0 <= self.minimum_quality <= 1.0,
            "minimum quality must lie in [0, 1]",
        )
        positive = (
            self.maximum_cycle_error_px,
            self.maximum_reprojection_error_px,
            self.pixel_noise_std,
            self.prior_std_m,
            self.shared_bias_std_m,
            self.two_view_extra_std_m,
            self.minimum_ray_angle_degrees,
            self.duplicate_center_tolerance_m,
            self.duplicate_rotation_tolerance,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "all geometric scales must be finite and positive",
        )
        _require(
            self.minimum_camera_count >= 2,
            "minimum camera count must be at least two",
        )
        _require(
            self.redundant_camera_count >= self.minimum_camera_count,
            "redundant camera count must cover the minimum",
        )


@dataclass(frozen=True)
class RenderedAllTrackerObservation:
    """Metric material-point observations and correlation-aware covariance."""

    points_world_m: np.ndarray
    covariance_m2: np.ndarray
    valid: np.ndarray
    raw_camera_count: np.ndarray
    effective_camera_count: np.ndarray
    reprojection_error_px: np.ndarray
    prior_reliability: np.ndarray
    two_view_fallback: np.ndarray

    def __post_init__(self) -> None:
        points = _readonly(self.points_world_m, dtype=np.float64)
        covariance = _readonly(self.covariance_m2, dtype=np.float64)
        valid = _readonly(self.valid, dtype=bool)
        raw_count = _readonly(self.raw_camera_count, dtype=np.int16)
        effective_count = _readonly(self.effective_camera_count, dtype=np.int16)
        reprojection = _readonly(self.reprojection_error_px, dtype=np.float64)
        reliability = _readonly(self.prior_reliability, dtype=np.float64)
        fallback = _readonly(self.two_view_fallback, dtype=bool)
        _require(
            points.ndim == 3 and points.shape[2] == 3,
            "points must have shape (frame, identity, 3)",
        )
        shape = points.shape[:2]
        _require(
            covariance.shape == (*shape, 3, 3),
            "covariance must have shape (frame, identity, 3, 3)",
        )
        for name, values in {
            "valid": valid,
            "raw camera count": raw_count,
            "effective camera count": effective_count,
            "reprojection": reprojection,
            "prior reliability": reliability,
            "two-view fallback": fallback,
        }.items():
            _require(values.shape == shape, f"{name} shape changed")
        _require(
            np.all(np.isfinite(covariance)),
            "covariance contains non-finite values",
        )
        _require(
            np.allclose(
                covariance,
                np.swapaxes(covariance, -1, -2),
                atol=1e-12,
                rtol=0.0,
            ),
            "covariance must be symmetric",
        )
        _require(
            np.all(np.linalg.eigvalsh(covariance) > 0.0),
            "covariance must be positive definite",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
        _require(
            np.all(raw_count >= effective_count) and np.all(effective_count >= 0),
            "camera counts are inconsistent",
        )
        _require(
            np.all(np.isfinite(points[valid])),
            "valid points contain non-finite values",
        )
        _require(
            np.all(np.isnan(points[~valid])),
            "invalid points must contain NaNs",
        )
        for name, values in {
            "points_world_m": points,
            "covariance_m2": covariance,
            "valid": valid,
            "raw_camera_count": raw_count,
            "effective_camera_count": effective_count,
            "reprojection_error_px": reprojection,
            "prior_reliability": reliability,
            "two_view_fallback": fallback,
        }.items():
            object.__setattr__(self, name, values)


def project_world_points(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into one calibrated camera."""

    points = np.asarray(points_world_m, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        "points must have shape (N, 3)",
    )
    _require(camera_matrix.shape == (3, 3), "intrinsics must be 3x3")
    _require(pose.shape == (4, 4), "camera_to_world must be 4x4")
    _require(
        np.all(np.isfinite(points))
        and np.all(np.isfinite(camera_matrix))
        and np.all(np.isfinite(pose)),
        "projection inputs must be finite",
    )
    world_to_camera = np.linalg.inv(pose)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    projected = (camera_matrix @ camera.T).T
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    positive = depth > 0.0
    pixels[positive] = projected[positive, :2] / projected[positive, 2:3]
    return pixels, depth


def _camera_pose_groups(
    camera_to_world: np.ndarray,
    *,
    center_tolerance_m: float,
    rotation_tolerance: float,
) -> np.ndarray:
    poses = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        poses.ndim == 3 and poses.shape[1:] == (4, 4),
        "camera poses must have shape (camera, 4, 4)",
    )
    groups = np.full(len(poses), -1, dtype=np.int64)
    group_count = 0
    for camera in range(len(poses)):
        for previous in range(camera):
            if (
                np.linalg.norm(poses[camera, :3, 3] - poses[previous, :3, 3])
                <= center_tolerance_m
                and np.max(
                    np.abs(
                        poses[camera, :3, :3] - poses[previous, :3, :3]
                    )
                )
                <= rotation_tolerance
            ):
                groups[camera] = groups[previous]
                break
        if groups[camera] < 0:
            groups[camera] = group_count
            group_count += 1
    return groups


def _representative_cameras(
    active: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    selected: list[int] = []
    for group in np.unique(groups[active]):
        candidates = active[groups[active] == group]
        best = np.max(quality[candidates])
        selected.append(int(np.min(candidates[quality[candidates] == best])))
    return np.asarray(selected, dtype=np.int64)


def _ray_geometry(
    tracks_xy: np.ndarray,
    cameras: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse_intrinsics = np.linalg.inv(intrinsics[cameras])
    homogeneous = np.column_stack(
        (tracks_xy[cameras], np.ones(len(cameras), dtype=np.float64))
    )
    rays_camera = np.einsum("ni,nji->nj", homogeneous, inverse_intrinsics)
    rotations = camera_to_world[cameras, :3, :3]
    rays_world = np.einsum("ni,nji->nj", rays_camera, rotations)
    rays_world /= np.maximum(
        np.linalg.norm(rays_world, axis=1, keepdims=True),
        1e-12,
    )
    centers = camera_to_world[cameras, :3, 3]
    projectors = np.eye(3)[None] - np.einsum(
        "ni,nj->nij",
        rays_world,
        rays_world,
    )
    return rays_world, centers, projectors


def _solve_rays(
    centers: np.ndarray,
    projectors: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(quality, dtype=np.float64)
    weights /= np.sum(weights)
    matrix = np.sum(weights[:, None, None] * projectors, axis=0)
    right = np.sum(
        weights[:, None] * np.einsum("nij,nj->ni", projectors, centers),
        axis=0,
    )
    return np.linalg.solve(matrix + 1e-12 * np.eye(3), right)


def _maximum_ray_angle_degrees(rays: np.ndarray) -> float:
    if len(rays) < 2:
        return 0.0
    cosine = np.clip(rays @ rays.T, -1.0, 1.0)
    return float(np.degrees(np.max(np.arccos(cosine))))


def build_rendered_alltracker_observation(
    target_tracks_xy: np.ndarray,
    quality_probability: np.ndarray,
    cycle_error_px: np.ndarray,
    source_supported: np.ndarray,
    target_supported: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    config: RenderedAllTrackerConfig | None = None,
) -> RenderedAllTrackerObservation:
    """Triangulate render-to-real correspondences without a state residual."""

    cfg = config or RenderedAllTrackerConfig()
    tracks = np.asarray(target_tracks_xy, dtype=np.float64)
    quality = np.asarray(quality_probability, dtype=np.float64)
    cycle = np.asarray(cycle_error_px, dtype=np.float64)
    source = np.asarray(source_supported, dtype=bool)
    target = np.asarray(target_supported, dtype=bool)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        tracks.ndim == 4 and tracks.shape[3] == 2,
        "tracks must have shape (frame, camera, identity, 2)",
    )
    shape = tracks.shape[:3]
    for name, values in {
        "quality": quality,
        "cycle": cycle,
        "source support": source,
        "target support": target,
    }.items():
        _require(values.shape == shape, f"{name} shape changed")
    _require(
        camera_matrix.shape == (tracks.shape[1], 3, 3),
        "intrinsics shape changed",
    )
    _require(
        poses.shape == (tracks.shape[1], 4, 4),
        "camera pose shape changed",
    )
    _require(
        np.all(np.isfinite(camera_matrix)) and np.all(np.isfinite(poses)),
        "camera calibration must be finite",
    )
    finite = np.all(np.isfinite(tracks), axis=3)
    view_valid = (
        finite
        & np.isfinite(quality)
        & np.isfinite(cycle)
        & (quality >= cfg.minimum_quality)
        & (cycle <= cfg.maximum_cycle_error_px)
        & source
        & target
    )
    raw_count = np.sum(view_valid, axis=1).astype(np.int16)
    output_shape = (tracks.shape[0], tracks.shape[2])
    points = np.full((*output_shape, 3), np.nan, dtype=np.float64)
    covariance = np.broadcast_to(
        cfg.prior_std_m**2 * np.eye(3),
        (*output_shape, 3, 3),
    ).copy()
    valid = np.zeros(output_shape, dtype=bool)
    effective_count = np.zeros(output_shape, dtype=np.int16)
    reprojection = np.full(output_shape, np.nan, dtype=np.float64)
    reliability = np.zeros(output_shape, dtype=np.float64)
    two_view = np.zeros(output_shape, dtype=bool)
    pose_groups = _camera_pose_groups(
        poses,
        center_tolerance_m=cfg.duplicate_center_tolerance_m,
        rotation_tolerance=cfg.duplicate_rotation_tolerance,
    )

    for frame in range(tracks.shape[0]):
        for identity in range(tracks.shape[2]):
            active = np.flatnonzero(view_valid[frame, :, identity])
            if len(active) < cfg.minimum_camera_count:
                continue
            cameras = _representative_cameras(
                active,
                quality[frame, :, identity],
                pose_groups,
            )
            effective_count[frame, identity] = len(cameras)
            if len(cameras) < cfg.minimum_camera_count:
                continue
            rays, centers, projectors = _ray_geometry(
                tracks[frame, :, identity],
                cameras,
                camera_matrix,
                poses,
            )
            ray_angle = _maximum_ray_angle_degrees(rays)
            if ray_angle < cfg.minimum_ray_angle_degrees:
                continue
            local_quality = quality[frame, cameras, identity]
            point = _solve_rays(centers, projectors, local_quality)
            errors: list[float] = []
            positive_depth = True
            for camera in cameras:
                projected, depth = project_world_points(
                    point[None],
                    camera_matrix[camera],
                    poses[camera],
                )
                errors.append(
                    float(
                        np.linalg.norm(
                            projected[0] - tracks[frame, camera, identity]
                        )
                    )
                )
                positive_depth &= bool(depth[0] > 0.0)
            rms_reprojection = float(
                np.sqrt(
                    np.average(
                        np.square(errors),
                        weights=local_quality,
                    )
                )
            )
            reprojection[frame, identity] = rms_reprojection
            if (
                not positive_depth
                or rms_reprojection > cfg.maximum_reprojection_error_px
            ):
                continue

            distances = np.linalg.norm(point[None] - centers, axis=1)
            focal = np.sqrt(
                camera_matrix[cameras, 0, 0]
                * camera_matrix[cameras, 1, 1]
            )
            metric_variance = np.square(
                cfg.pixel_noise_std * distances / focal
            ) / np.maximum(local_quality, 1e-6)
            ci_weight = local_quality / np.sum(local_quality)
            information = np.eye(3) / cfg.prior_std_m**2
            information += np.sum(
                ci_weight[:, None, None]
                * projectors
                / metric_variance[:, None, None],
                axis=0,
            )
            point_covariance = np.linalg.inv(information)
            point_covariance += cfg.shared_bias_std_m**2 * np.eye(3)
            if len(cameras) < cfg.redundant_camera_count:
                point_covariance += cfg.two_view_extra_std_m**2 * np.eye(3)
                two_view[frame, identity] = True

            cycle_score = np.exp(
                -0.5
                * np.square(
                    np.max(cycle[frame, cameras, identity])
                    / cfg.maximum_cycle_error_px
                )
            )
            reprojection_score = np.exp(
                -0.5
                * np.square(
                    rms_reprojection / cfg.maximum_reprojection_error_px
                )
            )
            redundancy_score = min(
                1.0,
                len(cameras) / cfg.redundant_camera_count,
            )
            points[frame, identity] = point
            covariance[frame, identity] = point_covariance
            valid[frame, identity] = True
            reliability[frame, identity] = float(
                np.clip(
                    np.min(local_quality)
                    * cycle_score
                    * reprojection_score
                    * redundancy_score,
                    0.0,
                    1.0,
                )
            )

    return RenderedAllTrackerObservation(
        points_world_m=points,
        covariance_m2=covariance,
        valid=valid,
        raw_camera_count=raw_count,
        effective_camera_count=effective_count,
        reprojection_error_px=reprojection,
        prior_reliability=reliability,
        two_view_fallback=two_view,
    )


__all__ = [
    "RenderedAllTrackerConfig",
    "RenderedAllTrackerObservation",
    "build_rendered_alltracker_observation",
    "project_world_points",
]
