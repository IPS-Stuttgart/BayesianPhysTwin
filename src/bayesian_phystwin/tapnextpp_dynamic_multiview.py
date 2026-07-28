"""Correlation-aware multiview lifting for dynamic TAPNext++ queries.

The provider is deliberately conservative. Cameras with the same calibrated
pose form one information cluster, a claim-bearing observation needs at least
three such clusters, and local metric covariance uses an equal-weight
covariance-intersection approximation. Coherent camera bias is exported as a
separate low-rank factor rather than copied into every local covariance block.

No PhysTwin innovation enters this module. Physical geometry may propose the
query association, but reliability is formed only from tracker visibility,
mask/depth support, multiview consistency, and redundancy. The legacy path
also uses assignment entropy; the set-valued path represents that ambiguity
once through metric covariance instead.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .observation_belief import ObservationBeliefV1, array_sha256

PROTOCOL_ID = "deform360-dynamic-tapnextpp-provider-v1"
TAPNEXT_REPOSITORY = "google-deepmind/tapnet"
TAPNEXT_REVISION = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNEXT_CHECKPOINT_SHA256 = (
    "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f"
)
_PROVIDER_FINAL_WEIGHT_SEMANTICS = "final-per-row-effective-sample-cap-v1"
LEGACY_ENTROPY_RELIABILITY = "legacy-entropy-reliability-v1"
COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY = (
    "covariance-only-assignment-uncertainty-v1"
)
_ASSIGNMENT_UNCERTAINTY_MODES = frozenset(
    {
        LEGACY_ENTROPY_RELIABILITY,
        COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY,
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _positive_definite(values: np.ndarray, *, floor: float) -> np.ndarray:
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


@dataclass(frozen=True)
class DynamicMultiviewConfig:
    """Frozen geometric and uncertainty choices from the v1 protocol."""

    visibility_threshold: float = 0.5
    maximum_reprojection_error_px: float = 3.0
    maximum_depth_residual_m: float = 0.03
    minimum_proposal_view_count: int = 2
    minimum_claim_view_count: int = 3
    mask_patch_radius_px: int = 2
    minimum_object_mask_fraction: float = 0.20
    pixel_standard_deviation_px: float = 1.5
    shared_bias_standard_deviation_m: float = 0.005
    maximum_effective_samples_per_group: float = 3.0
    camera_cluster_translation_tolerance_m: float = 1e-7
    camera_cluster_rotation_tolerance: float = 1e-7
    covariance_eigenvalue_floor_m2: float = 1e-12
    two_view_covariance_inflation: float = 2.0
    assignment_uncertainty_mode: str = LEGACY_ENTROPY_RELIABILITY

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
            self.minimum_proposal_view_count >= 2,
            "proposal lifting requires two views",
        )
        _require(
            self.minimum_claim_view_count >= 2,
            "claim-bearing lifting requires at least two views",
        )
        _require(
            self.minimum_claim_view_count >= self.minimum_proposal_view_count,
            "claim-bearing support is weaker than proposal support",
        )
        _require(
            self.mask_patch_radius_px >= 0,
            "mask patch radius must be nonnegative",
        )
        _require(
            0.0 <= self.minimum_object_mask_fraction <= 1.0,
            "mask fraction must lie in [0, 1]",
        )
        for name in (
            "pixel_standard_deviation_px",
            "shared_bias_standard_deviation_m",
            "maximum_effective_samples_per_group",
            "camera_cluster_translation_tolerance_m",
            "camera_cluster_rotation_tolerance",
            "covariance_eigenvalue_floor_m2",
            "two_view_covariance_inflation",
        ):
            _require(
                np.isfinite(getattr(self, name)) and getattr(self, name) > 0.0,
                f"{name} must be finite and positive",
            )
        _require(
            self.two_view_covariance_inflation >= 2.0,
            "two-view covariance inflation must preserve covariance intersection",
        )
        _require(
            self.minimum_claim_view_count >= 3
            or self.two_view_covariance_inflation >= 4.0,
            "two-view claim support requires at least fourfold inflation",
        )
        _require(
            self.assignment_uncertainty_mode
            in _ASSIGNMENT_UNCERTAINTY_MODES,
            "unsupported assignment uncertainty mode",
        )


@dataclass(frozen=True)
class DynamicMultiviewResult:
    """Metric observations and dependence diagnostics before state inference."""

    trajectory_world_m: np.ndarray
    proposal_available: np.ndarray
    accepted_support: np.ndarray
    prior_reliability: np.ndarray
    association_probability: np.ndarray
    local_covariance_m2: np.ndarray
    naive_independent_covariance_m2: np.ndarray
    assignment_mixture_spread_m2: np.ndarray
    independent_support_count: np.ndarray
    raw_support_count: np.ndarray
    reprojection_rmse_px: np.ndarray
    depth_residual_rmse_m: np.ndarray
    inlier_camera_mask: np.ndarray
    camera_cluster_ids: np.ndarray
    shared_bias_standard_deviation_m: float
    config: DynamicMultiviewConfig

    def __post_init__(self) -> None:
        trajectory = _readonly(self.trajectory_world_m, dtype=np.float64)
        _require(
            trajectory.ndim == 3 and trajectory.shape[2] == 3,
            "trajectory_world_m must have shape (T, N, 3)",
        )
        frame_count, entity_count, _ = trajectory.shape
        scalar_shape = (frame_count, entity_count)
        arrays = {
            "proposal_available": _readonly(
                self.proposal_available, dtype=bool
            ),
            "accepted_support": _readonly(self.accepted_support, dtype=bool),
            "prior_reliability": _readonly(
                self.prior_reliability, dtype=np.float64
            ),
            "association_probability": _readonly(
                self.association_probability, dtype=np.float64
            ),
            "independent_support_count": _readonly(
                self.independent_support_count, dtype=np.int64
            ),
            "raw_support_count": _readonly(
                self.raw_support_count, dtype=np.int64
            ),
            "reprojection_rmse_px": _readonly(
                self.reprojection_rmse_px, dtype=np.float64
            ),
            "depth_residual_rmse_m": _readonly(
                self.depth_residual_rmse_m, dtype=np.float64
            ),
        }
        for name, values in arrays.items():
            _require(values.shape == scalar_shape, f"{name} shape changed")
        local = _readonly(self.local_covariance_m2, dtype=np.float64)
        naive = _readonly(
            self.naive_independent_covariance_m2, dtype=np.float64
        )
        spread = _readonly(
            self.assignment_mixture_spread_m2, dtype=np.float64
        )
        for name, values in (
            ("local_covariance_m2", local),
            ("naive_independent_covariance_m2", naive),
            ("assignment_mixture_spread_m2", spread),
        ):
            _require(
                values.shape == (*scalar_shape, 3, 3),
                f"{name} shape changed",
            )
            _require(np.all(np.isfinite(values)), f"{name} is not finite")
            _require(
                np.allclose(
                    values,
                    np.swapaxes(values, -1, -2),
                    atol=1e-12,
                    rtol=1e-10,
                ),
                f"{name} is not symmetric",
            )
        inlier = _readonly(self.inlier_camera_mask, dtype=bool)
        _require(
            inlier.ndim == 3 and inlier.shape[1:] == scalar_shape,
            "inlier_camera_mask must have shape (C, T, N)",
        )
        cluster_ids = _readonly(self.camera_cluster_ids, dtype=np.int64)
        _require(
            cluster_ids.shape == (len(inlier),),
            "camera_cluster_ids differ from camera count",
        )
        _require(
            np.array_equal(np.unique(cluster_ids), np.arange(cluster_ids.max() + 1)),
            "camera cluster IDs must be contiguous",
        )
        _require(
            np.all(np.isfinite(trajectory)),
            "trajectory_world_m contains non-finite values",
        )
        for name in ("prior_reliability", "association_probability"):
            values = arrays[name]
            _require(
                np.all(np.isfinite(values))
                and np.all((values >= 0.0) & (values <= 1.0)),
                f"{name} must lie in [0, 1]",
            )
        _require(
            np.all(
                arrays["independent_support_count"]
                <= arrays["raw_support_count"]
            ),
            "independent support exceeds raw support",
        )
        _require(
            np.all(
                arrays["accepted_support"]
                <= arrays["proposal_available"]
            ),
            "claim-bearing support lacks a proposal",
        )
        accepted = arrays["accepted_support"]
        if np.any(accepted):
            minimum_eigenvalue = np.min(
                np.linalg.eigvalsh(local[accepted]), axis=1
            )
            _require(
                np.all(minimum_eigenvalue > 0.0),
                "accepted local covariance must be positive definite",
            )
        _require(
            np.isfinite(self.shared_bias_standard_deviation_m)
            and self.shared_bias_standard_deviation_m > 0.0,
            "shared bias scale must be positive",
        )
        _require(
            isinstance(self.config, DynamicMultiviewConfig),
            "multiview configuration is invalid",
        )
        _require(
            self.shared_bias_standard_deviation_m
            == self.config.shared_bias_standard_deviation_m,
            "shared bias scale differs from the frozen configuration",
        )
        object.__setattr__(self, "trajectory_world_m", trajectory)
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "local_covariance_m2", local)
        object.__setattr__(
            self, "naive_independent_covariance_m2", naive
        )
        object.__setattr__(
            self, "assignment_mixture_spread_m2", spread
        )
        object.__setattr__(self, "inlier_camera_mask", inlier)
        object.__setattr__(self, "camera_cluster_ids", cluster_ids)


def camera_projection_matrix(
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Return a calibrated world-to-pixel projection matrix."""

    matrix = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    _require(matrix.shape == (3, 3), "intrinsic must have shape (3, 3)")
    _require(pose.shape == (4, 4), "camera pose must have shape (4, 4)")
    return matrix @ np.linalg.inv(pose)[:3]


def project_world_point(
    point_world_m: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project a world point and return pixels and projective depth."""

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
    """Triangulate a point from two or more calibrated observations."""

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


def independent_camera_clusters(
    camera_to_world: np.ndarray,
    *,
    config: DynamicMultiviewConfig | None = None,
) -> np.ndarray:
    """Group cameras that do not provide independent calibrated viewpoints."""

    cfg = config or DynamicMultiviewConfig()
    poses = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        poses.ndim == 3 and poses.shape[1:] == (4, 4),
        "camera poses must have shape (C, 4, 4)",
    )
    _require(np.all(np.isfinite(poses)), "camera poses are not finite")
    clusters = np.full(len(poses), -1, dtype=np.int64)
    representatives: list[int] = []
    for camera, pose in enumerate(poses):
        assigned = False
        for cluster_id, representative in enumerate(representatives):
            reference = poses[representative]
            translation_close = (
                np.linalg.norm(pose[:3, 3] - reference[:3, 3])
                <= cfg.camera_cluster_translation_tolerance_m
            )
            rotation_close = (
                np.linalg.norm(pose[:3, :3] - reference[:3, :3])
                <= cfg.camera_cluster_rotation_tolerance
            )
            if translation_close and rotation_close:
                clusters[camera] = cluster_id
                assigned = True
                break
        if not assigned:
            clusters[camera] = len(representatives)
            representatives.append(camera)
    return clusters


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
    valid = mask_patch & np.isfinite(depth_patch) & (depth_patch > 0.0)
    depth = float(np.median(depth_patch[valid])) if np.any(valid) else float("nan")
    return depth, mask_fraction


def _camera_depth_m(
    point_world_m: np.ndarray,
    camera_to_world: np.ndarray,
) -> float:
    camera = np.linalg.inv(camera_to_world) @ np.append(point_world_m, 1.0)
    return float(camera[2])


def _projection_jacobian(
    point_world_m: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    world_to_camera = np.linalg.inv(camera_to_world)
    rotation = world_to_camera[:3, :3]
    camera = world_to_camera[:3] @ np.append(point_world_m, 1.0)
    x, y, z = camera
    _require(z > 1e-9, "point is behind a selected camera")
    camera_jacobian = np.asarray(
        [
            [intrinsic[0, 0] / z, 0.0, -intrinsic[0, 0] * x / z**2],
            [0.0, intrinsic[1, 1] / z, -intrinsic[1, 1] * y / z**2],
        ]
    )
    return camera_jacobian @ rotation


def _camera_diagnostics(
    point_world_m: np.ndarray,
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    sampled_depth_m: np.ndarray,
    mask_fraction: np.ndarray,
    projections: np.ndarray,
    camera_to_world: np.ndarray,
    config: DynamicMultiviewConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_count = len(tracks_xy)
    reprojection = np.full(camera_count, np.inf, dtype=np.float64)
    depth_residual = np.full(camera_count, np.inf, dtype=np.float64)
    inlier = np.zeros(camera_count, dtype=bool)
    for camera in range(camera_count):
        if (
            visibility[camera] < config.visibility_threshold
            or not np.all(np.isfinite(tracks_xy[camera]))
            or not np.isfinite(sampled_depth_m[camera])
        ):
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
        depth_residual[camera] = abs(
            sampled_depth_m[camera]
            - _camera_depth_m(point_world_m, camera_to_world[camera])
        )
        inlier[camera] = bool(
            reprojection[camera]
            <= config.maximum_reprojection_error_px
            and depth_residual[camera] <= config.maximum_depth_residual_m
            and mask_fraction[camera]
            >= config.minimum_object_mask_fraction
        )
    return inlier, reprojection, depth_residual


def _cluster_representatives(
    eligible: np.ndarray,
    cluster_ids: np.ndarray,
    preliminary_score: np.ndarray,
) -> np.ndarray:
    representatives: list[int] = []
    for cluster_id in np.unique(cluster_ids[eligible]):
        members = eligible[cluster_ids[eligible] == cluster_id]
        order = np.lexsort((members, -preliminary_score[members]))
        representatives.append(int(members[order[0]]))
    return np.asarray(representatives, dtype=np.int64)


def _candidate_spread_m2(candidates: list[np.ndarray]) -> np.ndarray:
    if len(candidates) < 2:
        return np.zeros((3, 3), dtype=np.float64)
    values = np.asarray(candidates, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    return (centered.T @ centered) / len(values)


def _geometric_mean_probability(values: np.ndarray) -> float:
    probabilities = np.asarray(values, dtype=np.float64)
    if np.any(probabilities <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(probabilities))))


def conservative_triangulation_covariance_m2(
    point_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    visibility_probability: np.ndarray,
    assignment_pixel_covariance_px2: np.ndarray,
    *,
    candidate_spread_m2: np.ndarray | None = None,
    config: DynamicMultiviewConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative, naive, and assignment-spread covariance.

    The naive covariance sums independent per-view information. The returned
    local covariance averages that information, an equal-weight
    covariance-intersection approximation for unknown cross-view correlation.
    Shared metric camera bias is intentionally absent from both matrices.
    """

    cfg = config or DynamicMultiviewConfig()
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    visibility = np.asarray(visibility_probability, dtype=np.float64)
    assignment = np.asarray(
        assignment_pixel_covariance_px2, dtype=np.float64
    )
    count = len(visibility)
    _require(
        matrices.shape == (count, 3, 3),
        "intrinsics must have shape (V, 3, 3)",
    )
    _require(
        poses.shape == (count, 4, 4),
        "camera poses must have shape (V, 4, 4)",
    )
    _require(
        assignment.shape == (count, 2, 2),
        "assignment covariance must have shape (V, 2, 2)",
    )
    _require(
        count >= cfg.minimum_proposal_view_count,
        "covariance requires proposal support",
    )
    information = np.zeros((3, 3), dtype=np.float64)
    assignment_sandwich = np.zeros((3, 3), dtype=np.float64)
    for intrinsic, pose, probability, assignment_covariance in zip(
        matrices,
        poses,
        visibility,
        assignment,
        strict=True,
    ):
        _require(
            np.isfinite(probability) and 0.0 <= probability <= 1.0,
            "visibility probability must lie in [0, 1]",
        )
        assignment_covariance = 0.5 * (
            assignment_covariance + assignment_covariance.T
        )
        _require(
            np.min(np.linalg.eigvalsh(assignment_covariance)) >= -1e-10,
            "assignment covariance must be positive semidefinite",
        )
        jacobian = _projection_jacobian(
            point_world_m,
            intrinsic,
            pose,
        )
        pixel_variance = (
            cfg.pixel_standard_deviation_px**2
            / max(float(probability), 0.05)
        )
        pixel_precision = np.eye(2) / pixel_variance
        information += jacobian.T @ pixel_precision @ jacobian
        assignment_sandwich += (
            jacobian.T
            @ pixel_precision
            @ assignment_covariance
            @ pixel_precision
            @ jacobian
        )
    geometry = np.linalg.pinv(information, hermitian=True)
    assignment_metric = (
        geometry @ assignment_sandwich @ geometry
    )
    assignment_metric = 0.5 * (assignment_metric + assignment_metric.T)
    candidate_spread = (
        np.zeros((3, 3), dtype=np.float64)
        if candidate_spread_m2 is None
        else np.asarray(candidate_spread_m2, dtype=np.float64)
    )
    _require(
        candidate_spread.shape == (3, 3),
        "candidate spread must have shape (3, 3)",
    )
    candidate_spread = 0.5 * (candidate_spread + candidate_spread.T)
    _require(
        np.min(np.linalg.eigvalsh(candidate_spread)) >= -1e-10,
        "candidate spread must be positive semidefinite",
    )
    naive = geometry + assignment_metric + candidate_spread
    correlation_multiplier = float(count)
    if count == 2:
        correlation_multiplier = max(
            correlation_multiplier,
            cfg.two_view_covariance_inflation,
        )
    spread = correlation_multiplier * assignment_metric + candidate_spread
    conservative = correlation_multiplier * geometry + spread
    naive = _positive_definite(
        naive,
        floor=cfg.covariance_eigenvalue_floor_m2,
    )
    conservative = _positive_definite(
        conservative,
        floor=cfg.covariance_eigenvalue_floor_m2,
    )
    return conservative, naive, spread


def fuse_dynamic_tapnextpp_multiview(
    tracks_xy: np.ndarray,
    visibility_probability: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    query_points_world_m: np.ndarray,
    *,
    association_valid: np.ndarray | None = None,
    association_probability: np.ndarray | None = None,
    association_entropy: np.ndarray | None = None,
    assignment_pixel_covariance_px2: np.ndarray | None = None,
    config: DynamicMultiviewConfig | None = None,
) -> DynamicMultiviewResult:
    """Lift causal tracks without using the downstream physical innovation."""

    cfg = config or DynamicMultiviewConfig()
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    visibility = np.asarray(visibility_probability, dtype=np.float64)
    # A real eight-camera Deform360 prefix is several gigabytes in float64.
    # Sampling promotes the small per-track patches, so retaining the source
    # precision here avoids an unnecessary full-volume copy.
    depth = np.asarray(depths_m)
    masks = np.asarray(object_masks, dtype=bool)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    query = np.asarray(query_points_world_m, dtype=np.float64)
    _require(
        tracks.ndim == 4 and tracks.shape[-1] == 2,
        "tracks must have shape (C, T, N, 2)",
    )
    camera_count, frame_count, entity_count, _ = tracks.shape
    _require(
        visibility.shape == (camera_count, frame_count, entity_count),
        "visibility shape differs from tracks",
    )
    _require(
        depth.ndim == 4
        and depth.shape[:2] == (camera_count, frame_count),
        "depth must have shape (C, T, H, W)",
    )
    _require(masks.shape == depth.shape, "object masks differ from depth")
    _require(
        matrices.shape == (camera_count, 3, 3),
        "intrinsics must have shape (C, 3, 3)",
    )
    _require(
        poses.shape == (camera_count, 4, 4),
        "camera poses must have shape (C, 4, 4)",
    )
    _require(
        query.shape == (entity_count, 3),
        "query points differ from tracks",
    )
    _require(
        np.all(np.isfinite(visibility))
        and np.all((visibility >= 0.0) & (visibility <= 1.0)),
        "visibility must lie in [0, 1]",
    )
    valid = (
        np.ones((camera_count, entity_count), dtype=bool)
        if association_valid is None
        else np.asarray(association_valid, dtype=bool)
    )
    probability = (
        np.ones((camera_count, entity_count), dtype=np.float64)
        if association_probability is None
        else np.asarray(association_probability, dtype=np.float64)
    )
    entropy = (
        np.zeros((camera_count, entity_count), dtype=np.float64)
        if association_entropy is None
        else np.asarray(association_entropy, dtype=np.float64)
    )
    assignment_covariance = (
        np.zeros((camera_count, entity_count, 2, 2), dtype=np.float64)
        if assignment_pixel_covariance_px2 is None
        else np.asarray(
            assignment_pixel_covariance_px2,
            dtype=np.float64,
        )
    )
    _require(
        valid.shape == (camera_count, entity_count),
        "association validity shape changed",
    )
    for name, values in (
        ("association probability", probability),
        ("association entropy", entropy),
    ):
        _require(
            values.shape == (camera_count, entity_count)
            and np.all(np.isfinite(values))
            and np.all((values >= 0.0) & (values <= 1.0)),
            f"{name} must have shape (C, N) in [0, 1]",
        )
    _require(
        assignment_covariance.shape
        == (camera_count, entity_count, 2, 2),
        "assignment covariance must have shape (C, N, 2, 2)",
    )
    _require(
        camera_count >= cfg.minimum_proposal_view_count,
        "not enough cameras for proposal lifting",
    )

    projections = np.stack(
        [
            camera_projection_matrix(matrices[camera], poses[camera])
            for camera in range(camera_count)
        ]
    )
    cluster_ids = independent_camera_clusters(poses, config=cfg)
    trajectory = np.repeat(query[None], frame_count, axis=0)
    proposal = np.zeros((frame_count, entity_count), dtype=bool)
    accepted = np.zeros_like(proposal)
    reliability = np.zeros((frame_count, entity_count), dtype=np.float64)
    fused_association = np.zeros_like(reliability)
    local_covariance = np.repeat(
        np.eye(3)[None, None],
        frame_count * entity_count,
        axis=0,
    ).reshape(frame_count, entity_count, 3, 3)
    naive_covariance = local_covariance.copy()
    mixture_spread = np.zeros_like(local_covariance)
    independent_count = np.zeros(
        (frame_count, entity_count), dtype=np.int64
    )
    raw_count = np.zeros_like(independent_count)
    reprojection_rmse = np.full_like(reliability, np.nan)
    depth_rmse = np.full_like(reliability, np.nan)
    inlier_camera_mask = np.zeros(
        (camera_count, frame_count, entity_count),
        dtype=bool,
    )

    for frame in range(frame_count):
        for entity in range(entity_count):
            frame_tracks = tracks[:, frame, entity]
            frame_visibility = visibility[:, frame, entity]
            sampled_depth = np.full(camera_count, np.nan)
            mask_fraction = np.zeros(camera_count)
            for camera in range(camera_count):
                sampled_depth[camera], mask_fraction[camera] = _sample_patch(
                    depth[camera, frame],
                    masks[camera, frame],
                    frame_tracks[camera],
                    cfg.mask_patch_radius_px,
                )
            eligible = np.flatnonzero(
                valid[:, entity]
                & (frame_visibility >= cfg.visibility_threshold)
                & np.all(np.isfinite(frame_tracks), axis=1)
                & np.isfinite(sampled_depth)
                & (mask_fraction >= cfg.minimum_object_mask_fraction)
            )
            if len(eligible) < cfg.minimum_proposal_view_count:
                continue
            preliminary = (
                frame_visibility
                * mask_fraction
                * np.clip(1.0 - entropy[:, entity], 0.0, 1.0)
            )
            representatives = _cluster_representatives(
                eligible,
                cluster_ids,
                preliminary,
            )
            if len(representatives) < cfg.minimum_proposal_view_count:
                continue

            candidates: list[
                tuple[
                    tuple[int, float, float, tuple[int, int]],
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ]
            ] = []
            for first, second in itertools.combinations(representatives, 2):
                pair = np.asarray([first, second], dtype=np.int64)
                candidate = triangulate_dlt(
                    frame_tracks[pair],
                    projections[pair],
                )
                if not np.all(np.isfinite(candidate)):
                    continue
                inlier, reprojection, depth_residual = _camera_diagnostics(
                    candidate,
                    frame_tracks[representatives],
                    frame_visibility[representatives],
                    sampled_depth[representatives],
                    mask_fraction[representatives],
                    projections[representatives],
                    poses[representatives],
                    cfg,
                )
                support = int(np.sum(inlier))
                if support < cfg.minimum_proposal_view_count:
                    continue
                score = (
                    support,
                    -float(np.median(reprojection[inlier])),
                    -float(np.median(depth_residual[inlier])),
                    (-int(first), -int(second)),
                )
                candidates.append(
                    (
                        score,
                        candidate,
                        inlier,
                        reprojection,
                        depth_residual,
                    )
                )
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            best = candidates[0]
            selected = representatives[best[2]]
            refined = triangulate_dlt(
                frame_tracks[selected],
                projections[selected],
            )
            if not np.all(np.isfinite(refined)):
                continue
            refined_inlier, _, _ = _camera_diagnostics(
                refined,
                frame_tracks[representatives],
                frame_visibility[representatives],
                sampled_depth[representatives],
                mask_fraction[representatives],
                projections[representatives],
                poses[representatives],
                cfg,
            )
            selected = representatives[refined_inlier]
            if len(selected) < cfg.minimum_proposal_view_count:
                continue
            refined = triangulate_dlt(
                frame_tracks[selected],
                projections[selected],
            )
            raw_inlier, raw_reprojection, raw_depth_residual = (
                _camera_diagnostics(
                    refined,
                    frame_tracks,
                    frame_visibility,
                    sampled_depth,
                    mask_fraction,
                    projections,
                    poses,
                    cfg,
                )
            )
            independent_inlier, reprojection, depth_residual = (
                _camera_diagnostics(
                    refined,
                    frame_tracks[representatives],
                    frame_visibility[representatives],
                    sampled_depth[representatives],
                    mask_fraction[representatives],
                    projections[representatives],
                    poses[representatives],
                    cfg,
                )
            )
            selected = representatives[independent_inlier]
            if len(selected) < cfg.minimum_proposal_view_count:
                continue

            maximum_support = best[0][0]
            alternatives = [
                item[1]
                for item in candidates
                if item[0][0] == maximum_support
            ]
            candidate_spread = _candidate_spread_m2(alternatives)
            conservative, naive, assignment_spread = (
                conservative_triangulation_covariance_m2(
                    refined,
                    matrices[selected],
                    poses[selected],
                    frame_visibility[selected],
                    assignment_covariance[selected, entity],
                    candidate_spread_m2=candidate_spread,
                    config=cfg,
                )
            )
            selected_reprojection = reprojection[independent_inlier]
            selected_depth = depth_residual[independent_inlier]
            selected_mask = mask_fraction[selected]
            selected_visibility = frame_visibility[selected]
            selected_entropy = entropy[selected, entity]
            selected_association = probability[selected, entity]
            reprojection_value = float(
                np.sqrt(np.mean(np.square(selected_reprojection)))
            )
            depth_value = float(
                np.sqrt(np.mean(np.square(selected_depth)))
            )
            visibility_score = _geometric_mean_probability(
                selected_visibility
            )
            entropy_score = _geometric_mean_probability(
                1.0 - selected_entropy
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
            redundancy_score = min(
                1.0,
                len(selected) / max(3, cfg.minimum_claim_view_count),
            )
            trajectory[frame, entity] = refined
            proposal[frame, entity] = True
            accepted[frame, entity] = (
                len(selected) >= cfg.minimum_claim_view_count
            )
            independent_count[frame, entity] = len(selected)
            raw_count[frame, entity] = int(np.sum(raw_inlier))
            inlier_camera_mask[:, frame, entity] = raw_inlier
            reprojection_rmse[frame, entity] = reprojection_value
            depth_rmse[frame, entity] = depth_value
            local_covariance[frame, entity] = conservative
            naive_covariance[frame, entity] = naive
            mixture_spread[frame, entity] = assignment_spread
            fused_association[frame, entity] = (
                _geometric_mean_probability(selected_association)
            )
            assignment_reliability = (
                entropy_score
                if cfg.assignment_uncertainty_mode
                == LEGACY_ENTROPY_RELIABILITY
                else 1.0
            )
            reliability[frame, entity] = float(
                np.clip(
                    visibility_score
                    * float(np.mean(selected_mask))
                    * reprojection_score
                    * depth_score
                    * redundancy_score
                    * assignment_reliability,
                    0.0,
                    1.0,
                )
            )

    return DynamicMultiviewResult(
        trajectory_world_m=trajectory,
        proposal_available=proposal,
        accepted_support=accepted,
        prior_reliability=reliability,
        association_probability=fused_association,
        local_covariance_m2=local_covariance,
        naive_independent_covariance_m2=naive_covariance,
        assignment_mixture_spread_m2=mixture_spread,
        independent_support_count=independent_count,
        raw_support_count=raw_count,
        reprojection_rmse_px=reprojection_rmse,
        depth_residual_rmse_m=depth_rmse,
        inlier_camera_mask=inlier_camera_mask,
        camera_cluster_ids=cluster_ids,
        shared_bias_standard_deviation_m=(
            cfg.shared_bias_standard_deviation_m
        ),
        config=cfg,
    )


def dynamic_multiview_result_sha256(
    result: DynamicMultiviewResult,
) -> str:
    """Content-address the complete fused provider result."""

    digest = hashlib.sha256()
    descriptor = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "shared_bias_standard_deviation_m": (
            result.shared_bias_standard_deviation_m
        ),
        "configuration": asdict(result.config),
    }
    digest.update(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    arrays = {
        name: getattr(result, name)
        for name in (
            "trajectory_world_m",
            "proposal_available",
            "accepted_support",
            "prior_reliability",
            "association_probability",
            "local_covariance_m2",
            "naive_independent_covariance_m2",
            "assignment_mixture_spread_m2",
            "independent_support_count",
            "raw_support_count",
            "reprojection_rmse_px",
            "depth_residual_rmse_m",
            "inlier_camera_mask",
            "camera_cluster_ids",
        )
    }
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def build_dynamic_tapnextpp_observation_belief(
    result: DynamicMultiviewResult,
    *,
    case_id: str,
    frame_ids: np.ndarray,
    entity_ids: np.ndarray,
    entity_birth_frames: np.ndarray,
    entity_update_frames: np.ndarray,
    camera_names: tuple[str, ...],
    query_schedule_sha256: str,
    tracker_revision: str = TAPNEXT_REVISION,
    tracker_checkpoint_sha256: str = TAPNEXT_CHECKPOINT_SHA256,
    maximum_effective_samples_per_group: float | None = None,
) -> ObservationBeliefV1:
    """Export claim-bearing rows with explicit covariance dependence."""

    frames = np.asarray(frame_ids, dtype=np.int64)
    entities = np.asarray(entity_ids, dtype=np.int64)
    births = np.asarray(entity_birth_frames, dtype=np.int64)
    updates = np.asarray(entity_update_frames, dtype=np.int64)
    frame_count, entity_count, _ = result.trajectory_world_m.shape
    _require(
        frames.shape == (frame_count,)
        and np.all(frames >= 0)
        and np.all(np.diff(frames) > 0),
        "frame_ids must be strictly increasing and match the result",
    )
    for name, values in (
        ("entity_ids", entities),
        ("entity_birth_frames", births),
        ("entity_update_frames", updates),
    ):
        _require(
            values.shape == (entity_count,),
            f"{name} must match the result entity count",
        )
    _require(len(set(map(int, entities))) == entity_count, "entity IDs repeat")
    _require(np.all(births <= updates), "entity birth follows update")
    _require(
        len(camera_names) == len(result.camera_cluster_ids),
        "camera names differ from provider camera count",
    )
    for name, digest in (
        ("query_schedule_sha256", query_schedule_sha256),
        ("tracker_checkpoint_sha256", tracker_checkpoint_sha256),
    ):
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"{name} is not a SHA-256 digest",
        )
    _require(
        tracker_revision == TAPNEXT_REVISION,
        "tracker revision differs from the frozen protocol",
    )
    _require(
        tracker_checkpoint_sha256 == TAPNEXT_CHECKPOINT_SHA256,
        "tracker checkpoint differs from the frozen protocol",
    )
    effective_sample_cap = (
        result.config.maximum_effective_samples_per_group
        if maximum_effective_samples_per_group is None
        else float(maximum_effective_samples_per_group)
    )
    _require(
        np.isfinite(effective_sample_cap)
        and effective_sample_cap > 0.0,
        "effective-sample cap must be positive",
    )

    active = (
        result.accepted_support
        & (frames[:, None] >= births[None])
        & (frames[:, None] <= updates[None])
    )
    row_frame, row_entity_position = np.nonzero(active)
    provider_sha256 = dynamic_multiview_result_sha256(result)
    source_digest = hashlib.sha256(
        (
            f"{provider_sha256}\0{query_schedule_sha256}\0"
            f"{tracker_revision}\0{tracker_checkpoint_sha256}"
        ).encode("ascii")
    ).hexdigest()
    if not len(row_frame):
        return ObservationBeliefV1(
            case_id=case_id,
            stream_id="tapnextpp:dynamic-causal-multiview",
            causal_frame_stop=int(np.max(updates)) + 1,
            view_names=("fused-multiview",),
            window_names=("prior-only",),
            factor_names=(
                "shared_camera_bias_x",
                "shared_camera_bias_y",
                "shared_camera_bias_z",
            ),
            source_repository=TAPNEXT_REPOSITORY,
            source_revision=tracker_revision,
            source_artifact_sha256=source_digest,
            declared_frame_ids=np.unique(
                np.concatenate((births, updates))
            ),
            mean_xyz_m=np.empty((0, 3), dtype=np.float64),
            frame_ids=np.empty(0, dtype=np.int64),
            entity_ids=np.empty(0, dtype=np.int64),
            view_indices=np.empty(0, dtype=np.int64),
            window_indices=np.empty(0, dtype=np.int64),
            correlation_group_ids=np.empty(0, dtype=np.int64),
            factor_group_ids=np.empty(0, dtype=np.int64),
            prior_reliability=np.empty(0, dtype=np.float64),
            association_probability=np.empty(0, dtype=np.float64),
            local_covariance_m2=np.empty((0, 3, 3), dtype=np.float64),
            low_rank_factor_m=np.empty((0, 3, 3), dtype=np.float64),
            group_ids=np.empty(0, dtype=np.int64),
            group_prior_nominal_probability=np.empty(
                0,
                dtype=np.float64,
            ),
            group_composite_weight=np.empty(0, dtype=np.float64),
            metadata={
                "protocol_id": PROTOCOL_ID,
                "provider": (
                    "physics-guided-dynamic-tapnextpp-multiview-v1"
                ),
                "provider_result_sha256": provider_sha256,
                "query_schedule_sha256": query_schedule_sha256,
                "tracker_checkpoint_sha256": tracker_checkpoint_sha256,
                "tracker_revision": tracker_revision,
                "prior_only_fallback": True,
                "fallback_reason": "no-claim-bearing-observation-row",
                "future_prediction_payloads_opened": 0,
                "physical_innovation_used_as_prior_reliability": False,
                "association_probability_used_as_prior_reliability": False,
                "raw_camera_count": len(camera_names),
                "camera_names": list(camera_names),
                "configuration": asdict(result.config),
            },
        )
    row_frames = frames[row_frame]
    row_entities = entities[row_entity_position]
    row_births = births[row_entity_position]
    row_updates = updates[row_entity_position]

    window_pairs = sorted(
        set(zip(map(int, row_updates), map(int, row_births), strict=True))
    )
    window_position = {
        pair: index for index, pair in enumerate(window_pairs)
    }
    window_indices = np.asarray(
        [
            window_position[(int(update), int(birth))]
            for update, birth in zip(row_updates, row_births, strict=True)
        ],
        dtype=np.int64,
    )
    update_values = sorted(set(map(int, row_updates)))
    factor_position = {
        update: index for index, update in enumerate(update_values)
    }
    factor_groups = np.asarray(
        [factor_position[int(update)] for update in row_updates],
        dtype=np.int64,
    )
    group_keys = sorted(
        set(zip(map(int, row_births), map(int, row_entities), strict=True))
    )
    group_position = {key: index for index, key in enumerate(group_keys)}
    correlation_groups = np.asarray(
        [
            group_position[(int(birth), int(entity))]
            for birth, entity in zip(row_births, row_entities, strict=True)
        ],
        dtype=np.int64,
    )
    group_ids = np.arange(len(group_keys), dtype=np.int64)
    group_sizes = np.asarray(
        [
            int(np.sum(correlation_groups == group_id))
            for group_id in group_ids
        ],
        dtype=np.float64,
    )
    group_weights = np.minimum(
        1.0,
        effective_sample_cap / group_sizes,
    )
    factor = (
        result.shared_bias_standard_deviation_m
        * np.repeat(np.eye(3)[None], len(row_frame), axis=0)
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "provider": "physics-guided-dynamic-tapnextpp-multiview-v1",
        "provider_result_sha256": provider_sha256,
        "query_schedule_sha256": query_schedule_sha256,
        "tracker_checkpoint_sha256": tracker_checkpoint_sha256,
        "tracker_revision": tracker_revision,
        "causal_frame_stop_convention": "exclusive",
        "maximum_observed_frame_inclusive": int(np.max(row_frames)),
        "future_prediction_payloads_opened": 0,
        "physical_innovation_used_as_prior_reliability": False,
        "association_probability_used_as_prior_reliability": False,
        "prior_reliability_definition": (
            "tracker visibility, mask/depth support, reprojection consistency, "
            "and independent-view redundancy; assignment entropy is included "
            "only in the legacy mode"
        ),
        "innovation_processing": (
            "formed once downstream and processed by the robust likelihood"
        ),
        "local_covariance_definition": (
            "equal-weight covariance-intersection geometry plus assignment "
            "mixture spread; excludes coherent camera bias"
        ),
        "assignment_uncertainty_semantics": (
            result.config.assignment_uncertainty_mode
        ),
        "shared_bias_definition": (
            "one coherent 3-D low-rank factor per update interval"
        ),
        "correlation_group_definition": (
            "one material-identity trajectory within one birth wave"
        ),
        "maximum_effective_samples_per_group": float(
            effective_sample_cap
        ),
        "group_composite_weight_semantics": (
            _PROVIDER_FINAL_WEIGHT_SEMANTICS
        ),
        "independent_camera_cluster_count": int(
            len(np.unique(result.camera_cluster_ids))
        ),
        "raw_camera_count": len(camera_names),
        "camera_names": list(camera_names),
        "configuration": asdict(result.config),
    }
    return ObservationBeliefV1(
        case_id=case_id,
        stream_id="tapnextpp:dynamic-causal-multiview",
        causal_frame_stop=int(np.max(updates)) + 1,
        view_names=("fused-multiview",),
        window_names=tuple(
            f"update-{update:03d}-birth-{birth:03d}"
            for update, birth in window_pairs
        ),
        factor_names=(
            "shared_camera_bias_x",
            "shared_camera_bias_y",
            "shared_camera_bias_z",
        ),
        source_repository=TAPNEXT_REPOSITORY,
        source_revision=tracker_revision,
        source_artifact_sha256=source_digest,
        declared_frame_ids=np.unique(row_frames),
        mean_xyz_m=result.trajectory_world_m[
            row_frame,
            row_entity_position,
        ],
        frame_ids=row_frames,
        entity_ids=row_entities,
        view_indices=np.zeros(len(row_frame), dtype=np.int64),
        window_indices=window_indices,
        correlation_group_ids=correlation_groups,
        factor_group_ids=factor_groups,
        prior_reliability=result.prior_reliability[
            row_frame,
            row_entity_position,
        ],
        association_probability=result.association_probability[
            row_frame,
            row_entity_position,
        ],
        local_covariance_m2=result.local_covariance_m2[
            row_frame,
            row_entity_position,
        ],
        low_rank_factor_m=factor,
        group_ids=group_ids,
        group_prior_nominal_probability=np.ones(len(group_ids)),
        group_composite_weight=group_weights,
        metadata=metadata,
    )


__all__ = [
    "COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY",
    "DynamicMultiviewConfig",
    "DynamicMultiviewResult",
    "LEGACY_ENTROPY_RELIABILITY",
    "PROTOCOL_ID",
    "TAPNEXT_CHECKPOINT_SHA256",
    "TAPNEXT_REPOSITORY",
    "TAPNEXT_REVISION",
    "build_dynamic_tapnextpp_observation_belief",
    "camera_projection_matrix",
    "conservative_triangulation_covariance_m2",
    "dynamic_multiview_result_sha256",
    "fuse_dynamic_tapnextpp_multiview",
    "independent_camera_clusters",
    "project_world_point",
    "triangulate_dlt",
]
