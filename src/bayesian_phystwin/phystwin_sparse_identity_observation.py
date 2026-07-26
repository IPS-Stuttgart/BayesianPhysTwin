"""Covariance-bearing sparse material-identity observations for PhysTwin.

This module is deliberately observation-only. Source confidence, visibility,
cycle consistency, camera geometry, and multiview disagreement determine prior
reliability and metric covariance. A PhysTwin state is not an input, so its
innovation cannot be counted once as reliability and again in a likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .phystwin_bayesian_anchor import (
    RobustEndpointPosterior,
    robust_random_walk_endpoint,
)
from .phystwin_cotracker3_cues import project_world_points


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(value: np.ndarray, *, dtype: object) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SparseIdentityObservationConfig:
    """Frozen source-cue and covariance settings for one observation path."""

    minimum_view_quality: float = 0.5
    maximum_cycle_error_px: float = 5.0
    maximum_reprojection_error_px: float = 3.0
    minimum_camera_count: int = 2
    redundant_camera_count: int = 3
    pixel_noise_std: float = 2.0
    prior_std_m: float = 0.10
    shared_bias_std_m: float = 0.005
    two_view_extra_std_m: float = 0.010
    boundary_scale_px: float = 8.0
    two_view_reliability_multiplier: float = 0.5
    minimum_ray_angle_degrees: float = 0.5
    duplicate_center_tolerance_m: float = 1e-9
    duplicate_rotation_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        _require(
            0.0 <= self.minimum_view_quality <= 1.0,
            "minimum view quality must lie in [0, 1]",
        )
        positive = (
            self.maximum_cycle_error_px,
            self.maximum_reprojection_error_px,
            self.pixel_noise_std,
            self.prior_std_m,
            self.shared_bias_std_m,
            self.two_view_extra_std_m,
            self.boundary_scale_px,
            self.minimum_ray_angle_degrees,
            self.duplicate_center_tolerance_m,
            self.duplicate_rotation_tolerance,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "sparse identity scales must be finite and positive",
        )
        _require(
            self.minimum_camera_count >= 2,
            "minimum camera count must be at least two",
        )
        _require(
            self.redundant_camera_count >= self.minimum_camera_count,
            "redundant camera count must cover the minimum",
        )
        _require(
            0.0 < self.two_view_reliability_multiplier <= 1.0,
            "two-view reliability multiplier must lie in (0, 1]",
        )


@dataclass(frozen=True)
class SparseIdentityObservations:
    """Leakage-safe 3D material identities with calibrated-input diagnostics."""

    points_world_m: np.ndarray
    observation_covariance_m2: np.ndarray
    observation_variance_m2: np.ndarray
    prior_reliability: np.ndarray
    valid: np.ndarray
    raw_camera_count: np.ndarray
    effective_camera_count: np.ndarray
    reprojection_error_px: np.ndarray
    redundant_view_disagreement_m: np.ndarray
    two_view_fallback: np.ndarray

    def __post_init__(self) -> None:
        points = _readonly(self.points_world_m, dtype=np.float64)
        covariance = _readonly(self.observation_covariance_m2, dtype=np.float64)
        variance = _readonly(self.observation_variance_m2, dtype=np.float64)
        reliability = _readonly(self.prior_reliability, dtype=np.float64)
        valid = _readonly(self.valid, dtype=bool)
        raw_count = _readonly(self.raw_camera_count, dtype=np.int16)
        effective_count = _readonly(self.effective_camera_count, dtype=np.int16)
        reprojection = _readonly(self.reprojection_error_px, dtype=np.float64)
        disagreement = _readonly(
            self.redundant_view_disagreement_m,
            dtype=np.float64,
        )
        two_view = _readonly(self.two_view_fallback, dtype=bool)
        _require(
            points.ndim == 3 and points.shape[2] == 3,
            "points must have shape (T, N, 3)",
        )
        shape = points.shape[:2]
        _require(
            covariance.shape == (*shape, 3, 3),
            "covariance must have shape (T, N, 3, 3)",
        )
        for name, value in {
            "variance": variance,
            "reliability": reliability,
            "valid": valid,
            "raw count": raw_count,
            "effective count": effective_count,
            "reprojection": reprojection,
            "disagreement": disagreement,
            "two-view fallback": two_view,
        }.items():
            _require(value.shape == shape, f"{name} shape changed")
        _require(
            np.all(np.isfinite(covariance))
            and np.all(np.isfinite(variance))
            and np.all(variance > 0.0),
            "observation covariance must be finite and positive",
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
            "valid observations contain non-finite points",
        )
        _require(
            np.all(np.isnan(points[~valid])),
            "invalid observations must carry NaN points",
        )
        for name, value in {
            "points_world_m": points,
            "observation_covariance_m2": covariance,
            "observation_variance_m2": variance,
            "prior_reliability": reliability,
            "valid": valid,
            "raw_camera_count": raw_count,
            "effective_camera_count": effective_count,
            "reprojection_error_px": reprojection,
            "redundant_view_disagreement_m": disagreement,
            "two_view_fallback": two_view,
        }.items():
            object.__setattr__(self, name, value)


def _camera_pose_groups(
    camera_to_world: np.ndarray,
    *,
    center_tolerance_m: float,
    rotation_tolerance: float,
) -> np.ndarray:
    poses = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        poses.ndim == 3 and poses.shape[1:] == (4, 4),
        "camera poses must have shape (C, 4, 4)",
    )
    groups = np.full(len(poses), -1, dtype=np.int64)
    group_count = 0
    for camera in range(len(poses)):
        for previous in range(camera):
            if (
                np.linalg.norm(poses[camera, :3, 3] - poses[previous, :3, 3])
                <= center_tolerance_m
                and np.max(np.abs(poses[camera, :3, :3] - poses[previous, :3, :3]))
                <= rotation_tolerance
            ):
                groups[camera] = groups[previous]
                break
        if groups[camera] < 0:
            groups[camera] = group_count
            group_count += 1
    return groups


def _select_group_representatives(
    active: np.ndarray,
    quality: np.ndarray,
    pose_groups: np.ndarray,
) -> np.ndarray:
    selected: list[int] = []
    for group in np.unique(pose_groups[active]):
        candidates = active[pose_groups[active] == group]
        best_quality = np.max(quality[candidates])
        selected.append(int(np.min(candidates[quality[candidates] == best_quality])))
    return np.asarray(selected, dtype=np.int64)


def _ray_geometry(
    tracks_xy: np.ndarray,
    cameras: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse_intrinsics = np.linalg.inv(intrinsics[cameras])
    homogeneous = np.column_stack(
        [tracks_xy[cameras], np.ones(len(cameras), dtype=np.float64)]
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


def _leave_one_out_covariance(
    centers: np.ndarray,
    projectors: np.ndarray,
    quality: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, float]:
    if len(centers) < 3:
        return np.zeros((3, 3), dtype=np.float64), np.nan
    estimates = np.stack(
        [
            _solve_rays(
                np.delete(centers, index, axis=0),
                np.delete(projectors, index, axis=0),
                np.delete(quality, index, axis=0),
            )
            for index in range(len(centers))
        ]
    )
    delta = estimates - center
    covariance = delta.T @ delta / len(delta)
    disagreement = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
    return covariance, disagreement


def _raw_sparse_identity_observations(
    tracks: np.ndarray,
    selected_views: np.ndarray,
    quality: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    cycle_error: np.ndarray,
    boundary_distance: np.ndarray,
    source_available: np.ndarray,
    *,
    config: SparseIdentityObservationConfig,
) -> dict[str, np.ndarray]:
    camera_count, frame_count, track_count, _ = tracks.shape
    shape = (frame_count, track_count)
    points = np.full((*shape, 3), np.nan, dtype=np.float64)
    covariance = np.broadcast_to(
        config.prior_std_m**2 * np.eye(3),
        (*shape, 3, 3),
    ).copy()
    reliability = np.zeros(shape, dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    raw_count = np.sum(selected_views, axis=0).astype(np.int16)
    effective_count = np.zeros(shape, dtype=np.int16)
    reprojection = np.full(shape, np.nan, dtype=np.float64)
    disagreement = np.full(shape, np.nan, dtype=np.float64)
    two_view = np.zeros(shape, dtype=bool)
    pose_groups = _camera_pose_groups(
        camera_to_world,
        center_tolerance_m=config.duplicate_center_tolerance_m,
        rotation_tolerance=config.duplicate_rotation_tolerance,
    )

    for frame in range(frame_count):
        for track in range(track_count):
            active = np.flatnonzero(selected_views[:, frame, track])
            if len(active) < config.minimum_camera_count:
                continue
            cameras = _select_group_representatives(
                active,
                quality[:, frame, track],
                pose_groups,
            )
            effective_count[frame, track] = len(cameras)
            if len(cameras) < config.minimum_camera_count:
                continue
            rays, centers, projectors = _ray_geometry(
                tracks[:, frame, track],
                cameras,
                intrinsics,
                camera_to_world,
            )
            maximum_angle = _maximum_ray_angle_degrees(rays)
            if maximum_angle < config.minimum_ray_angle_degrees:
                continue
            view_quality = quality[cameras, frame, track]
            point = _solve_rays(centers, projectors, view_quality)
            projected_error: list[float] = []
            positive_depth = True
            for camera in cameras:
                projected, depth = project_world_points(
                    point[None],
                    intrinsics[camera],
                    camera_to_world[camera],
                )
                projected_error.append(
                    float(np.linalg.norm(projected[0] - tracks[camera, frame, track]))
                )
                positive_depth &= bool(depth[0] > 0.0)
            rms_reprojection = float(
                np.sqrt(
                    np.average(
                        np.square(projected_error),
                        weights=view_quality,
                    )
                )
            )
            reprojection[frame, track] = rms_reprojection
            if (
                not positive_depth
                or rms_reprojection > config.maximum_reprojection_error_px
                or not source_available[frame, track]
            ):
                continue

            distance = np.linalg.norm(point[None] - centers, axis=1)
            focal = np.sqrt(intrinsics[cameras, 0, 0] * intrinsics[cameras, 1, 1])
            metric_variance = np.square(
                config.pixel_noise_std * distance / focal
            ) / np.maximum(view_quality, 1e-6)
            ci_weight = view_quality / np.sum(view_quality)
            information = np.eye(3) / config.prior_std_m**2
            information += np.sum(
                ci_weight[:, None, None] * projectors / metric_variance[:, None, None],
                axis=0,
            )
            point_covariance = np.linalg.inv(information)
            loo_covariance, loo_disagreement = _leave_one_out_covariance(
                centers,
                projectors,
                view_quality,
                point,
            )
            point_covariance += loo_covariance
            point_covariance += config.shared_bias_std_m**2 * np.eye(3)
            if len(cameras) < config.redundant_camera_count:
                point_covariance += config.two_view_extra_std_m**2 * np.eye(3)
                two_view[frame, track] = True
            covariance[frame, track] = point_covariance
            disagreement[frame, track] = loo_disagreement

            cycle_score = np.exp(
                -0.5
                * np.square(cycle_error[frame, track] / config.maximum_cycle_error_px)
            )
            boundary_score = 1.0 - np.exp(
                -boundary_distance[frame, track] / config.boundary_scale_px
            )
            reprojection_score = np.exp(
                -0.5
                * np.square(rms_reprojection / config.maximum_reprojection_error_px)
            )
            geometry_score = min(
                1.0,
                np.sin(np.radians(maximum_angle))
                / max(
                    np.sin(np.radians(10.0)),
                    1e-12,
                ),
            )
            redundancy_score = (
                1.0
                if len(cameras) >= config.redundant_camera_count
                else config.two_view_reliability_multiplier
            )
            prior = (
                float(np.min(view_quality))
                * cycle_score
                * boundary_score
                * reprojection_score
                * geometry_score
                * redundancy_score
            )
            points[frame, track] = point
            reliability[frame, track] = float(np.clip(prior, 0.0, 1.0))
            valid[frame, track] = reliability[frame, track] > 0.0

    return {
        "points": points,
        "covariance": covariance,
        "reliability": reliability,
        "valid": valid,
        "raw_count": raw_count,
        "effective_count": effective_count,
        "reprojection": reprojection,
        "disagreement": disagreement,
        "two_view": two_view,
    }


def load_cotracker3_sparse_identity_observations(
    cues_path: str | Path,
    initial_world_points_m: np.ndarray,
    *,
    train_end_frame: int,
    config: SparseIdentityObservationConfig | None = None,
) -> SparseIdentityObservations:
    """Load causal CoTracker3 identities with conservative metric covariance."""

    cfg = config or SparseIdentityObservationConfig()
    _require(train_end_frame > 0, "train_end_frame must be positive")
    required = {
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
        "forward_backward_error_px",
        "forward_backward_valid",
        "boundary_distance",
        "cue_available",
    }
    with np.load(cues_path) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                "CoTracker3 archive lacks sparse-identity fields: "
                + ", ".join(sorted(missing))
            )
        tracks = np.asarray(
            archive["multiview_tracks_xy_prefix"][:, :train_end_frame],
            dtype=np.float64,
        )
        quality = np.asarray(
            archive["multiview_quality_probability_prefix"][:, :train_end_frame],
            dtype=np.float64,
        )
        view_valid = np.asarray(
            archive["multiview_view_valid_prefix"][:, :train_end_frame],
            dtype=bool,
        )
        intrinsics = np.asarray(archive["multiview_intrinsics"], dtype=np.float64)
        camera_to_world = np.asarray(
            archive["multiview_camera_to_world"],
            dtype=np.float64,
        )
        cycle_error = np.asarray(
            archive["forward_backward_error_px"][:train_end_frame],
            dtype=np.float64,
        )
        cycle_valid = np.asarray(
            archive["forward_backward_valid"][:train_end_frame],
            dtype=bool,
        )
        boundary = np.asarray(
            archive["boundary_distance"][:train_end_frame],
            dtype=np.float64,
        )
        available = np.asarray(
            archive["cue_available"][:train_end_frame],
            dtype=bool,
        )

    _require(
        tracks.ndim == 4 and tracks.shape[3] == 2,
        "multiview tracks must have shape (C, T, N, 2)",
    )
    _require(
        quality.shape == tracks.shape[:3] and view_valid.shape == quality.shape,
        "multiview quality or validity shape changed",
    )
    _require(
        tracks.shape[1] == train_end_frame,
        "archive is shorter than train_end_frame",
    )
    camera_count, _, track_count, _ = tracks.shape
    _require(
        intrinsics.shape == (camera_count, 3, 3),
        "intrinsics do not match camera count",
    )
    _require(
        camera_to_world.shape == (camera_count, 4, 4),
        "camera poses do not match camera count",
    )
    _require(
        cfg.minimum_camera_count <= camera_count,
        "minimum camera count exceeds archive",
    )
    initial = np.asarray(initial_world_points_m, dtype=np.float64)
    _require(initial.shape == (track_count, 3), "initial points must have shape (N, 3)")
    shape = (train_end_frame, track_count)
    for name, value in {
        "cycle error": cycle_error,
        "cycle validity": cycle_valid,
        "boundary distance": boundary,
        "cue availability": available,
    }.items():
        _require(value.shape == shape, f"{name} shape changed")

    source_available = (
        cycle_valid
        & np.isfinite(cycle_error)
        & (cycle_error <= cfg.maximum_cycle_error_px)
        & np.isfinite(boundary)
        & (boundary > 0.0)
        & available
    )
    selected_views = (
        view_valid
        & np.all(np.isfinite(tracks), axis=3)
        & np.isfinite(quality)
        & (quality >= cfg.minimum_view_quality)
        & source_available[None]
    )
    raw = _raw_sparse_identity_observations(
        tracks,
        selected_views,
        quality,
        intrinsics,
        camera_to_world,
        cycle_error,
        boundary,
        source_available,
        config=cfg,
    )

    points = raw["points"]
    covariance = raw["covariance"]
    reliability = raw["reliability"]
    valid = raw["valid"]
    anchored = np.full_like(points, np.nan)
    anchored_covariance = np.broadcast_to(
        cfg.prior_std_m**2 * np.eye(3),
        covariance.shape,
    ).copy()
    anchored_reliability = np.zeros_like(reliability)
    anchored_valid = np.zeros_like(valid)
    for track in range(track_count):
        if not valid[0, track]:
            continue
        track_valid = valid[:, track]
        anchored[track_valid, track] = (
            initial[track] + points[track_valid, track] - points[0, track]
        )
        anchored_covariance[0, track] = covariance[0, track]
        later = np.flatnonzero(track_valid & (np.arange(train_end_frame) > 0))
        if len(later):
            anchored_covariance[later, track] = 2.0 * (
                covariance[later, track] + covariance[0, track]
            )
        anchored_reliability[track_valid, track] = np.minimum(
            reliability[track_valid, track],
            reliability[0, track],
        )
        anchored_valid[track_valid, track] = True
    anchored[~anchored_valid] = np.nan
    isotropic_variance = (
        np.trace(
            anchored_covariance,
            axis1=2,
            axis2=3,
        )
        / 3.0
    )
    return SparseIdentityObservations(
        points_world_m=anchored,
        observation_covariance_m2=anchored_covariance,
        observation_variance_m2=isotropic_variance,
        prior_reliability=anchored_reliability,
        valid=anchored_valid,
        raw_camera_count=raw["raw_count"],
        effective_camera_count=raw["effective_count"],
        reprojection_error_px=raw["reprojection"],
        redundant_view_disagreement_m=raw["disagreement"],
        two_view_fallback=raw["two_view"],
    )


def sparse_identity_endpoint(
    observations: SparseIdentityObservations,
    baseline_points_world_m: np.ndarray,
    *,
    end_frame: int,
    process_variance: float,
    initial_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
) -> RobustEndpointPosterior:
    """Apply the state innovation once through the robust endpoint likelihood."""

    baseline = np.asarray(baseline_points_world_m, dtype=np.float64)
    _require(
        baseline.shape == observations.points_world_m.shape,
        "baseline must match sparse identity observations",
    )
    residual = np.zeros_like(baseline)
    residual[observations.valid] = (
        observations.points_world_m[observations.valid] - baseline[observations.valid]
    )
    return robust_random_walk_endpoint(
        residual,
        observations.valid,
        end_frame=end_frame,
        process_variance=process_variance,
        observation_variance=observations.observation_variance_m2,
        initial_variance=initial_variance,
        inlier_prior=inlier_prior,
        outlier_variance_multiplier=outlier_variance_multiplier,
        prior_reliability=observations.prior_reliability,
    )


__all__ = [
    "SparseIdentityObservationConfig",
    "SparseIdentityObservations",
    "load_cotracker3_sparse_identity_observations",
    "sparse_identity_endpoint",
]
