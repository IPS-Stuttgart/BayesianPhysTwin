"""Conservative multiview lifting for causal TAPNext++ observations.

The tracker supplies per-camera image trajectories and visibility
probabilities.  This module lifts them to metric world coordinates without
using a PhysTwin innovation to manufacture perception confidence.  Candidate
geometry determines cross-view association, while tracker visibility, object
mask support, depth agreement, and reprojection consistency determine
observation reliability.

Two-view estimates remain available because real deformable-object points are
often occluded in one of three cameras.  They receive an explicit covariance
inflation, and every estimate retains a shared metric bias floor so duplicating
correlated camera evidence cannot drive uncertainty to zero.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TAPNextPPMultiviewConfig:
    """Frozen geometric and uncertainty choices for multiview lifting."""

    visibility_threshold: float = 0.5
    maximum_reprojection_error_px: float = 3.0
    maximum_depth_residual_m: float = 0.03
    minimum_view_count: int = 2
    mask_patch_radius_px: int = 2
    minimum_object_mask_fraction: float = 0.20
    pixel_standard_deviation_px: float = 1.5
    shared_bias_standard_deviation_m: float = 0.005
    two_view_covariance_inflation: float = 4.0

    def __post_init__(self) -> None:
        _require(
            0.0 < self.visibility_threshold < 1.0,
            "visibility threshold must lie in (0, 1)",
        )
        _require(
            self.maximum_reprojection_error_px > 0.0,
            "reprojection threshold must be positive",
        )
        _require(
            self.maximum_depth_residual_m > 0.0,
            "depth threshold must be positive",
        )
        _require(
            self.minimum_view_count >= 2,
            "multiview lifting requires at least two views",
        )
        _require(
            self.mask_patch_radius_px >= 0,
            "mask patch radius must be nonnegative",
        )
        _require(
            0.0 <= self.minimum_object_mask_fraction <= 1.0,
            "mask fraction must lie in [0, 1]",
        )
        _require(
            self.pixel_standard_deviation_px > 0.0,
            "pixel standard deviation must be positive",
        )
        _require(
            self.shared_bias_standard_deviation_m > 0.0,
            "shared bias floor must be positive",
        )
        _require(
            self.two_view_covariance_inflation >= 1.0,
            "two-view covariance inflation must be at least one",
        )


def camera_projection_matrix(
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Return a world-to-pixel projection matrix."""

    matrix = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    _require(matrix.shape == (3, 3), "intrinsic must have shape (3, 3)")
    _require(pose.shape == (4, 4), "camera pose must have shape (4, 4)")
    return matrix @ np.linalg.inv(pose)[:3]


def project_world_point(
    point_world_m: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project one world point and return image position and projective depth."""

    point = np.asarray(point_world_m, dtype=np.float64)
    matrix = np.asarray(projection, dtype=np.float64)
    _require(point.shape == (3,), "world point must have shape (3,)")
    _require(matrix.shape == (3, 4), "projection must have shape (3, 4)")
    homogeneous = matrix @ np.append(point, 1.0)
    if not np.all(np.isfinite(homogeneous)) or homogeneous[2] <= 1e-9:
        return np.full(2, np.nan), float("nan")
    return homogeneous[:2] / homogeneous[2], float(homogeneous[2])


def triangulate_dlt(
    image_points_xy: np.ndarray,
    projection_matrices: np.ndarray,
) -> np.ndarray:
    """Triangulate one point from two or more calibrated image observations."""

    points = np.asarray(image_points_xy, dtype=np.float64)
    projections = np.asarray(projection_matrices, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 2,
        "image points must have shape (V, 2)",
    )
    _require(
        projections.shape == (len(points), 3, 4),
        "projection matrices must have shape (V, 3, 4)",
    )
    _require(len(points) >= 2, "triangulation requires two views")
    rows = []
    for (x, y), projection in zip(points, projections, strict=True):
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
    _, _, right = np.linalg.svd(np.asarray(rows), full_matrices=False)
    homogeneous = right[-1]
    if (
        not np.all(np.isfinite(homogeneous))
        or abs(float(homogeneous[3])) <= 1e-12
    ):
        return np.full(3, np.nan)
    return homogeneous[:3] / homogeneous[3]


def _sample_patch(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    point_xy: np.ndarray,
    radius_px: int,
) -> tuple[float, float]:
    height, width = depth_m.shape
    x, y = np.asarray(point_xy, dtype=np.float64)
    if not np.isfinite(x) or not np.isfinite(y):
        return float("nan"), 0.0
    center_x = int(np.rint(x))
    center_y = int(np.rint(y))
    if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
        return float("nan"), 0.0
    x0 = max(0, center_x - radius_px)
    x1 = min(width, center_x + radius_px + 1)
    y0 = max(0, center_y - radius_px)
    y1 = min(height, center_y + radius_px + 1)
    mask_patch = np.asarray(object_mask[y0:y1, x0:x1], dtype=bool)
    depth_patch = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64)
    mask_fraction = float(np.mean(mask_patch)) if mask_patch.size else 0.0
    valid = (
        mask_patch
        & np.isfinite(depth_patch)
        & (depth_patch > 0.0)
    )
    depth = float(np.median(depth_patch[valid])) if np.any(valid) else float("nan")
    return depth, mask_fraction


def _camera_depth(
    point_world_m: np.ndarray,
    camera_to_world: np.ndarray,
) -> float:
    point = np.append(np.asarray(point_world_m, dtype=np.float64), 1.0)
    camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64)) @ point
    return float(camera[2])


def _projection_jacobian(
    point_world_m: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    world_to_camera = np.linalg.inv(
        np.asarray(camera_to_world, dtype=np.float64)
    )
    rotation = world_to_camera[:3, :3]
    camera = world_to_camera[:3] @ np.append(point_world_m, 1.0)
    x, y, z = camera
    _require(z > 1e-9, "point is behind a selected camera")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    camera_jacobian = np.asarray(
        [
            [fx / z, 0.0, -fx * x / (z**2)],
            [0.0, fy / z, -fy * y / (z**2)],
        ],
        dtype=np.float64,
    )
    return camera_jacobian @ rotation


def conservative_triangulation_covariance_m2(
    point_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    visibility_probability: np.ndarray,
    *,
    mixture_spread_m2: np.ndarray | None = None,
    config: TAPNextPPMultiviewConfig | None = None,
) -> np.ndarray:
    """Return metric covariance with a non-vanishing shared-bias floor."""

    cfg = config or TAPNextPPMultiviewConfig()
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    visibility = np.asarray(visibility_probability, dtype=np.float64)
    _require(
        matrices.shape == (len(visibility), 3, 3),
        "intrinsics must have shape (V, 3, 3)",
    )
    _require(
        poses.shape == (len(visibility), 4, 4),
        "camera poses must have shape (V, 4, 4)",
    )
    _require(
        len(visibility) >= cfg.minimum_view_count,
        "covariance requires the minimum view count",
    )
    _require(
        np.all(np.isfinite(visibility))
        and np.all((visibility >= 0.0) & (visibility <= 1.0)),
        "visibility probabilities must lie in [0, 1]",
    )
    reliability = float(np.clip(np.mean(visibility), 0.05, 1.0))
    pixel_variance = (cfg.pixel_standard_deviation_px**2) / reliability
    information = np.zeros((3, 3), dtype=np.float64)
    for intrinsic, pose in zip(matrices, poses, strict=True):
        jacobian = _projection_jacobian(point_world_m, intrinsic, pose)
        information += (jacobian.T @ jacobian) / pixel_variance
    geometry = np.linalg.pinv(information, hermitian=True)
    if len(visibility) == 2:
        geometry *= cfg.two_view_covariance_inflation
    shared = np.eye(3) * (cfg.shared_bias_standard_deviation_m**2)
    spread = (
        np.zeros((3, 3), dtype=np.float64)
        if mixture_spread_m2 is None
        else np.asarray(mixture_spread_m2, dtype=np.float64)
    )
    _require(spread.shape == (3, 3), "mixture spread must have shape (3, 3)")
    covariance = (geometry + shared + spread)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(
        eigenvalues,
        cfg.shared_bias_standard_deviation_m**2,
    )
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _candidate_diagnostics(
    point_world_m: np.ndarray,
    tracks_xy: np.ndarray,
    visibility_probability: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    projections: np.ndarray,
    camera_to_world: np.ndarray,
    config: TAPNextPPMultiviewConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_count = len(tracks_xy)
    reprojection = np.full(camera_count, np.inf, dtype=np.float64)
    depth_residual = np.full(camera_count, np.inf, dtype=np.float64)
    mask_fraction = np.zeros(camera_count, dtype=np.float64)
    inlier = np.zeros(camera_count, dtype=bool)
    for camera in range(camera_count):
        if visibility_probability[camera] < config.visibility_threshold:
            continue
        projected, _ = project_world_point(
            point_world_m,
            projections[camera],
        )
        if not np.all(np.isfinite(projected)):
            continue
        reprojection[camera] = float(
            np.linalg.norm(projected - tracks_xy[camera])
        )
        depth, mask_fraction[camera] = _sample_patch(
            depths_m[camera],
            object_masks[camera],
            tracks_xy[camera],
            config.mask_patch_radius_px,
        )
        if np.isfinite(depth):
            depth_residual[camera] = abs(
                depth
                - _camera_depth(point_world_m, camera_to_world[camera])
            )
        inlier[camera] = bool(
            reprojection[camera]
            <= config.maximum_reprojection_error_px
            and depth_residual[camera] <= config.maximum_depth_residual_m
            and mask_fraction[camera]
            >= config.minimum_object_mask_fraction
        )
    return inlier, reprojection, depth_residual, mask_fraction


def _mixture_spread(
    candidates_world_m: list[np.ndarray],
) -> np.ndarray:
    if len(candidates_world_m) < 2:
        return np.zeros((3, 3), dtype=np.float64)
    candidates = np.asarray(candidates_world_m, dtype=np.float64)
    centered = candidates - np.mean(candidates, axis=0, keepdims=True)
    return (centered.T @ centered) / len(candidates)


def fuse_causal_multiview_tracks(
    tracks_xy: np.ndarray,
    visibility_probability: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    query_points_world_m: np.ndarray,
    *,
    config: TAPNextPPMultiviewConfig | None = None,
) -> dict[str, np.ndarray]:
    """Lift causal per-camera tracks into conservative metric observations."""

    cfg = config or TAPNextPPMultiviewConfig()
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    visibility = np.asarray(visibility_probability, dtype=np.float64)
    depth = np.asarray(depths_m, dtype=np.float64)
    masks = np.asarray(object_masks, dtype=bool)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    query = np.asarray(query_points_world_m, dtype=np.float64)
    _require(
        tracks.ndim == 4 and tracks.shape[-1] == 2,
        "tracks must have shape (C, T, N, 2)",
    )
    camera_count, frame_count, point_count, _ = tracks.shape
    _require(
        visibility.shape == (camera_count, frame_count, point_count),
        "visibility shape differs from tracks",
    )
    _require(
        depth.ndim == 4 and depth.shape[:2] == (camera_count, frame_count),
        "depth must have shape (C, T, H, W)",
    )
    _require(masks.shape == depth.shape, "object-mask shape differs from depth")
    _require(
        matrices.shape == (camera_count, 3, 3),
        "intrinsics must have shape (C, 3, 3)",
    )
    _require(
        poses.shape == (camera_count, 4, 4),
        "camera poses must have shape (C, 4, 4)",
    )
    _require(query.shape == (point_count, 3), "query shape differs from tracks")
    _require(
        camera_count >= cfg.minimum_view_count,
        "not enough cameras for the configured lift",
    )
    projections = np.stack(
        [
            camera_projection_matrix(matrices[camera], poses[camera])
            for camera in range(camera_count)
        ]
    )
    output = np.repeat(query[None], frame_count, axis=0)
    accepted = np.zeros((frame_count, point_count), dtype=bool)
    reliability = np.zeros((frame_count, point_count), dtype=np.float64)
    support_count = np.zeros((frame_count, point_count), dtype=np.int16)
    reprojection_rmse = np.full(
        (frame_count, point_count),
        np.nan,
        dtype=np.float64,
    )
    depth_rmse = np.full_like(reprojection_rmse, np.nan)
    covariance = np.repeat(
        np.eye(3, dtype=np.float64)[None, None],
        frame_count * point_count,
        axis=0,
    ).reshape(frame_count, point_count, 3, 3)
    covariance *= cfg.shared_bias_standard_deviation_m**2
    mask_support = np.zeros(
        (camera_count, frame_count, point_count),
        dtype=np.float64,
    )
    depth_sample = np.full_like(mask_support, np.nan)

    for frame in range(frame_count):
        for point_index in range(point_count):
            frame_tracks = tracks[:, frame, point_index]
            frame_visibility = visibility[:, frame, point_index]
            for camera in range(camera_count):
                sampled_depth, sampled_mask = _sample_patch(
                    depth[camera, frame],
                    masks[camera, frame],
                    frame_tracks[camera],
                    cfg.mask_patch_radius_px,
                )
                depth_sample[camera, frame, point_index] = sampled_depth
                mask_support[camera, frame, point_index] = sampled_mask
            eligible = [
                camera
                for camera in range(camera_count)
                if frame_visibility[camera] >= cfg.visibility_threshold
                and np.all(np.isfinite(frame_tracks[camera]))
            ]
            if len(eligible) < cfg.minimum_view_count:
                continue

            ranked: list[
                tuple[
                    tuple[int, float, float],
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ]
            ] = []
            for subset_size in range(
                cfg.minimum_view_count,
                len(eligible) + 1,
            ):
                for subset in itertools.combinations(eligible, subset_size):
                    subset_array = np.asarray(subset, dtype=np.int64)
                    candidate = triangulate_dlt(
                        frame_tracks[subset_array],
                        projections[subset_array],
                    )
                    if not np.all(np.isfinite(candidate)):
                        continue
                    diagnostics = _candidate_diagnostics(
                        candidate,
                        frame_tracks,
                        frame_visibility,
                        depth[:, frame],
                        masks[:, frame],
                        projections,
                        poses,
                        cfg,
                    )
                    inlier, reprojection, depth_residual, mask_fraction = (
                        diagnostics
                    )
                    count = int(np.sum(inlier))
                    if count < cfg.minimum_view_count:
                        continue
                    score = (
                        count,
                        -float(np.median(reprojection[inlier])),
                        -float(np.median(depth_residual[inlier])),
                    )
                    ranked.append(
                        (
                            score,
                            candidate,
                            inlier,
                            reprojection,
                            depth_residual,
                            mask_fraction,
                        )
                    )
            if not ranked:
                continue
            ranked.sort(key=lambda item: item[0], reverse=True)
            best = ranked[0]
            inlier_cameras = np.flatnonzero(best[2])
            refined = triangulate_dlt(
                frame_tracks[inlier_cameras],
                projections[inlier_cameras],
            )
            diagnostics = _candidate_diagnostics(
                refined,
                frame_tracks,
                frame_visibility,
                depth[:, frame],
                masks[:, frame],
                projections,
                poses,
                cfg,
            )
            inlier, reprojection, depth_residual, mask_fraction = diagnostics
            inlier_cameras = np.flatnonzero(inlier)
            if len(inlier_cameras) < cfg.minimum_view_count:
                continue
            refined = triangulate_dlt(
                frame_tracks[inlier_cameras],
                projections[inlier_cameras],
            )
            if not np.all(np.isfinite(refined)):
                continue

            maximum_support = max(item[0][0] for item in ranked)
            alternatives = [
                item[1]
                for item in ranked
                if item[0][0] == maximum_support
                and np.all(np.isfinite(item[1]))
            ]
            spread = _mixture_spread(alternatives)
            covariance[frame, point_index] = (
                conservative_triangulation_covariance_m2(
                    refined,
                    matrices[inlier_cameras],
                    poses[inlier_cameras],
                    frame_visibility[inlier_cameras],
                    mixture_spread_m2=spread,
                    config=cfg,
                )
            )
            selected_reprojection = reprojection[inlier_cameras]
            selected_depth = depth_residual[inlier_cameras]
            selected_mask = mask_fraction[inlier_cameras]
            selected_visibility = frame_visibility[inlier_cameras]
            reprojection_value = float(
                np.sqrt(np.mean(np.square(selected_reprojection)))
            )
            depth_value = float(np.sqrt(np.mean(np.square(selected_depth))))
            visibility_score = float(
                np.exp(np.mean(np.log(np.clip(selected_visibility, 1e-6, 1.0))))
            )
            reprojection_score = float(
                np.exp(
                    -0.5
                    * (
                        reprojection_value
                        / cfg.maximum_reprojection_error_px
                    )
                    ** 2
                )
            )
            depth_score = float(
                np.exp(
                    -0.5
                    * (depth_value / cfg.maximum_depth_residual_m) ** 2
                )
            )
            mask_score = float(np.mean(selected_mask))
            output[frame, point_index] = refined
            accepted[frame, point_index] = True
            support_count[frame, point_index] = len(inlier_cameras)
            reprojection_rmse[frame, point_index] = reprojection_value
            depth_rmse[frame, point_index] = depth_value
            reliability[frame, point_index] = float(
                np.clip(
                    visibility_score
                    * reprojection_score
                    * depth_score
                    * mask_score,
                    0.0,
                    1.0,
                )
            )

    return {
        "trajectory_world_m": output.astype(np.float32),
        "accepted_support": accepted,
        "observation_reliability": reliability.astype(np.float32),
        "observation_covariance_m2": covariance.astype(np.float32),
        "support_view_count": support_count,
        "reprojection_rmse_px": reprojection_rmse.astype(np.float32),
        "depth_residual_rmse_m": depth_rmse.astype(np.float32),
        "per_camera_mask_support": mask_support.astype(np.float32),
        "per_camera_depth_sample_m": depth_sample.astype(np.float32),
    }


__all__ = [
    "TAPNextPPMultiviewConfig",
    "camera_projection_matrix",
    "conservative_triangulation_covariance_m2",
    "fuse_causal_multiview_tracks",
    "project_world_point",
    "triangulate_dlt",
]
