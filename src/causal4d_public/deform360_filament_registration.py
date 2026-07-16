"""Topology-aware multiview registration for Deform360 filaments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class FilamentRegistrationConfig:
    """Development configuration for fixed-topology filament registration."""

    samples_per_edge: int = 5
    maximum_mask_sample_count: int = 192
    maximum_point_sample_count: int = 512
    silhouette_distance_scale_px: float = 6.0
    maximum_silhouette_distance_px: float = 30.0
    point_distance_scale_m: float = 0.015
    maximum_point_distance_m: float = 0.075
    containment_weight: float = 1.0
    coverage_weight: float = 0.7
    point_cloud_weight: float = 0.12
    equal_edge_weight: float = 8.0
    curvature_weight: float = 0.08
    initialization_weight: float = 0.015
    maximum_node_displacement_m: float = 0.20
    maximum_function_evaluations: int = 180
    robust_scale: float = 1.0
    length_projection_iterations: int = 32

    def __post_init__(self) -> None:
        _require(self.samples_per_edge >= 2, "each filament edge needs two samples")
        _require(
            self.maximum_mask_sample_count >= 16,
            "mask sampling budget is too small",
        )
        _require(
            self.maximum_point_sample_count >= 16,
            "point sampling budget is too small",
        )
        _require(
            self.silhouette_distance_scale_px > 0.0
            and self.maximum_silhouette_distance_px
            >= self.silhouette_distance_scale_px,
            "invalid silhouette distance scales",
        )
        _require(
            self.point_distance_scale_m > 0.0
            and self.maximum_point_distance_m >= self.point_distance_scale_m,
            "invalid point-cloud distance scales",
        )
        for name, value in (
            ("containment_weight", self.containment_weight),
            ("coverage_weight", self.coverage_weight),
            ("point_cloud_weight", self.point_cloud_weight),
            ("equal_edge_weight", self.equal_edge_weight),
            ("curvature_weight", self.curvature_weight),
            ("initialization_weight", self.initialization_weight),
        ):
            _require(value >= 0.0, f"{name} must be nonnegative")
        _require(
            self.maximum_node_displacement_m > 0.0,
            "node displacement bound must be positive",
        )
        _require(
            self.maximum_function_evaluations >= 1,
            "optimizer evaluation budget must be positive",
        )
        _require(self.robust_scale > 0.0, "robust scale must be positive")
        _require(
            self.length_projection_iterations >= 1,
            "length projection needs at least one iteration",
        )


@dataclass(frozen=True)
class FilamentRegistrationQAConfig:
    """Topology-aware development gates, to be frozen before new-object use."""

    local_neighbor_count: int = 16
    maximum_local_radial_p95_m: float = 0.015
    minimum_median_mask_support: float = 0.70
    minimum_lower_quartile_mask_support: float = 0.45
    maximum_median_mask_coverage_p95_px: float = 12.0
    minimum_effective_camera_count: float = 3.0
    maximum_edge_length_coefficient_of_variation: float = 0.08
    maximum_relative_length_error: float = 0.04

    def __post_init__(self) -> None:
        _require(self.local_neighbor_count >= 4, "local geometry needs four neighbors")
        _require(
            self.maximum_local_radial_p95_m > 0.0,
            "local thickness threshold must be positive",
        )
        for name, value in (
            ("minimum_median_mask_support", self.minimum_median_mask_support),
            (
                "minimum_lower_quartile_mask_support",
                self.minimum_lower_quartile_mask_support,
            ),
        ):
            _require(0.0 <= value <= 1.0, f"{name} must be a probability")
        _require(
            self.maximum_median_mask_coverage_p95_px >= 0.0,
            "mask coverage threshold must be nonnegative",
        )
        _require(
            self.minimum_effective_camera_count >= 2.0,
            "effective camera threshold must be at least two",
        )
        _require(
            self.maximum_edge_length_coefficient_of_variation >= 0.0,
            "edge variation threshold must be nonnegative",
        )
        _require(
            self.maximum_relative_length_error >= 0.0,
            "length threshold must be nonnegative",
        )


def _validate_centerline(centerline_world_m: np.ndarray) -> np.ndarray:
    centerline = np.asarray(centerline_world_m, dtype=np.float64)
    _require(
        centerline.ndim == 2 and centerline.shape[1] == 3 and len(centerline) >= 4,
        "filament centerline must have shape (N,3) with at least four nodes",
    )
    _require(np.all(np.isfinite(centerline)), "filament centerline is non-finite")
    _require(
        np.all(np.linalg.norm(np.diff(centerline, axis=0), axis=1) > 1e-8),
        "filament centerline contains a collapsed edge",
    )
    return centerline


def sample_filament_centerline(
    centerline_world_m: np.ndarray, *, samples_per_edge: int
) -> np.ndarray:
    """Sample an ordered open chain without repeating shared edge endpoints."""

    centerline = _validate_centerline(centerline_world_m)
    _require(samples_per_edge >= 2, "each filament edge needs two samples")
    alpha = np.arange(samples_per_edge, dtype=np.float64) / samples_per_edge
    samples = [
        centerline[index] + alpha[:, None] * (centerline[index + 1] - centerline[index])
        for index in range(len(centerline) - 1)
    ]
    return np.concatenate((*samples, centerline[-1:]), axis=0)


def _project_world_points(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world_m, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    _require(intrinsic.shape == (3, 3), "camera intrinsics must have shape (3,3)")
    _require(pose.shape == (4, 4), "camera pose must have shape (4,4)")
    world_to_camera = np.linalg.inv(pose)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    safe_depth = np.where(depth > 1e-8, depth, 1.0)
    pixels = np.column_stack(
        (
            camera[:, 0] / safe_depth * intrinsic[0, 0] + intrinsic[0, 2],
            camera[:, 1] / safe_depth * intrinsic[1, 1] + intrinsic[1, 2],
        )
    )
    height, width = image_shape
    visible = (
        (depth > 1e-8)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    return pixels, visible


def _pixel_indices(
    pixels: np.ndarray, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    columns = np.clip(np.rint(pixels[:, 0]), 0, width - 1).astype(np.int64)
    rows = np.clip(np.rint(pixels[:, 1]), 0, height - 1).astype(np.int64)
    return rows, columns


def _subsample_rows(values: np.ndarray, maximum_count: int) -> np.ndarray:
    if len(values) <= maximum_count:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, maximum_count)).astype(np.int64)
    return values[indices]


def _mask_pixels(mask: np.ndarray, maximum_count: int) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    _require(len(rows) > 0, "filament object mask is empty")
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    return _subsample_rows(pixels, maximum_count)


def _validate_views(
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    gripper_masks_by_camera: Mapping[str, np.ndarray] | None,
) -> tuple[str, ...]:
    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= 2, "filament registration needs at least two views")
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "filament registration is missing camera intrinsics",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "filament registration is missing camera poses",
    )
    if gripper_masks_by_camera is not None:
        _require(
            all(camera in gripper_masks_by_camera for camera in cameras),
            "filament registration is missing gripper masks",
        )
    for camera in cameras:
        mask = np.asarray(masks_by_camera[camera])
        _require(mask.ndim == 2, f"object mask for {camera} is not two-dimensional")
        _require(np.any(mask), f"object mask for {camera} is empty")
        if gripper_masks_by_camera is not None:
            gripper = np.asarray(gripper_masks_by_camera[camera])
            _require(
                gripper.shape == mask.shape,
                f"gripper/object mask shapes differ for {camera}",
            )
    return cameras


def _validate_camera_reliability(
    cameras: tuple[str, ...],
    camera_reliability_by_camera: Mapping[str, float] | None,
) -> np.ndarray:
    if camera_reliability_by_camera is None:
        return np.ones(len(cameras), dtype=np.float64)
    _require(
        all(camera in camera_reliability_by_camera for camera in cameras),
        "filament registration is missing camera reliability",
    )
    values = np.asarray(
        [camera_reliability_by_camera[camera] for camera in cameras],
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(values)), "camera reliability is non-finite")
    _require(
        np.all((values > 0.0) & (values <= 1.0)),
        "camera reliability must lie in (0,1]",
    )
    return values


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    _require(0.0 <= quantile <= 1.0, "weighted quantile is invalid")
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def filament_local_geometry_diagnostics(
    points_world_m: np.ndarray,
    *,
    neighbor_count: int = 16,
    maximum_center_count: int = 256,
) -> dict[str, Any]:
    """Measure local radial thickness without confusing curvature for width."""

    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) >= 8,
        "filament points must have shape (N,3) with at least eight points",
    )
    _require(np.all(np.isfinite(points)), "filament points are non-finite")
    _require(neighbor_count >= 4, "local geometry needs four neighbors")
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is a graph dependency
        raise RuntimeError("SciPy is required for filament diagnostics") from error

    k = min(neighbor_count, len(points))
    centers = np.rint(
        np.linspace(0, len(points) - 1, min(maximum_center_count, len(points)))
    ).astype(np.int64)
    _, neighbors = cKDTree(points).query(points[centers], k=k)
    radial = []
    for indices in np.atleast_2d(neighbors):
        neighborhood = points[np.asarray(indices)]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        axial = centered @ axes[0]
        orthogonal = centered - axial[:, None] * axes[0]
        radial.extend(np.linalg.norm(orthogonal, axis=1).tolist())
    radial_values = np.asarray(radial, dtype=np.float64)

    global_centered = points - np.median(points, axis=0)
    _, _, global_axes = np.linalg.svd(global_centered, full_matrices=False)
    global_coordinates = global_centered @ global_axes.T
    global_spans = np.sort(
        np.percentile(global_coordinates, 99.0, axis=0)
        - np.percentile(global_coordinates, 1.0, axis=0)
    )[::-1]
    return {
        "point_count": len(points),
        "neighbor_count": k,
        "evaluated_center_count": len(centers),
        "local_radial_distance_m": {
            "median": float(np.median(radial_values)),
            "p95": float(np.quantile(radial_values, 0.95)),
            "maximum": float(np.max(radial_values)),
        },
        "global_pca_q01_to_q99_spans_m_descending": global_spans.tolist(),
    }


def filament_multiview_support_diagnostics(
    centerline_world_m: np.ndarray,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    gripper_masks_by_camera: Mapping[str, np.ndarray] | None = None,
    observed_centerline_pixels_by_camera: Mapping[str, np.ndarray] | None = None,
    camera_reliability_by_camera: Mapping[str, float] | None = None,
    samples_per_edge: int = 5,
    maximum_mask_sample_count: int = 512,
) -> dict[str, Any]:
    """Score a chain against silhouettes while treating gripper cover as missing."""

    cameras = _validate_views(
        masks_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        gripper_masks_by_camera,
    )
    if observed_centerline_pixels_by_camera is not None:
        _require(
            all(camera in observed_centerline_pixels_by_camera for camera in cameras),
            "filament registration is missing observed medial paths",
        )
    reliability = _validate_camera_reliability(cameras, camera_reliability_by_camera)
    samples = sample_filament_centerline(
        centerline_world_m, samples_per_edge=samples_per_edge
    )
    per_camera = []
    for camera, camera_reliability in zip(cameras, reliability, strict=True):
        mask = np.asarray(masks_by_camera[camera], dtype=bool)
        gripper = (
            np.asarray(gripper_masks_by_camera[camera], dtype=bool)
            if gripper_masks_by_camera is not None
            else np.zeros_like(mask)
        )
        pixels, visible = _project_world_points(
            samples,
            intrinsics_by_camera[camera],
            camera_to_world_by_camera[camera],
            mask.shape,
        )
        rows, columns = _pixel_indices(pixels, mask.shape)
        occluded = visible & gripper[rows, columns]
        evaluable = visible & ~occluded
        inside = evaluable & mask[rows, columns]
        support = (
            float(np.count_nonzero(inside) / np.count_nonzero(evaluable))
            if np.any(evaluable)
            else 0.0
        )
        eligible_pixels = pixels[evaluable]
        observed = (
            np.asarray(observed_centerline_pixels_by_camera[camera], dtype=np.float64)
            if observed_centerline_pixels_by_camera is not None
            else _mask_pixels(mask, maximum_mask_sample_count)
        )
        _require(
            observed.ndim == 2 and observed.shape[1] == 2 and len(observed) >= 2,
            f"observed centerline for {camera} must have shape (N,2)",
        )
        observed = _subsample_rows(observed, maximum_mask_sample_count)
        if len(eligible_pixels):
            distances = np.linalg.norm(
                observed[:, None, :] - eligible_pixels[None, :, :], axis=2
            ).min(axis=1)
        else:
            distances = np.full(len(observed), float(np.hypot(*mask.shape)))
        per_camera.append(
            {
                "camera": camera,
                "prior_camera_reliability": float(camera_reliability),
                "projected_sample_count": len(samples),
                "visible_sample_count": int(np.count_nonzero(visible)),
                "occluded_sample_count": int(np.count_nonzero(occluded)),
                "evaluable_sample_count": int(np.count_nonzero(evaluable)),
                "inside_object_mask_count": int(np.count_nonzero(inside)),
                "visibility_aware_mask_support": support,
                "mask_to_centerline_distance_px": {
                    "median": float(np.median(distances)),
                    "p95": float(np.quantile(distances, 0.95)),
                    "maximum": float(np.max(distances)),
                },
            }
        )
    support_values = np.asarray(
        [item["visibility_aware_mask_support"] for item in per_camera]
    )
    coverage_p95 = np.asarray(
        [item["mask_to_centerline_distance_px"]["p95"] for item in per_camera]
    )
    effective_camera_count = float(np.sum(reliability) ** 2 / np.sum(reliability**2))
    return {
        "camera_count": len(cameras),
        "centerline_sample_count": len(samples),
        "per_camera": per_camera,
        "visibility_aware_mask_support": {
            "minimum": float(np.min(support_values)),
            "lower_quartile": float(np.quantile(support_values, 0.25)),
            "median": float(np.median(support_values)),
            "maximum": float(np.max(support_values)),
        },
        "mask_coverage_p95_px": {
            "minimum": float(np.min(coverage_p95)),
            "median": float(np.median(coverage_p95)),
            "maximum": float(np.max(coverage_p95)),
        },
        "camera_reliability": {
            "minimum": float(np.min(reliability)),
            "median": float(np.median(reliability)),
            "maximum": float(np.max(reliability)),
            "sum": float(np.sum(reliability)),
            "effective_camera_count": effective_camera_count,
        },
        "reliability_weighted_visibility_aware_mask_support": {
            "lower_quartile": _weighted_quantile(support_values, reliability, 0.25),
            "median": _weighted_quantile(support_values, reliability, 0.5),
        },
        "reliability_weighted_mask_coverage_p95_px": {
            "median": _weighted_quantile(coverage_p95, reliability, 0.5),
        },
    }


def _registration_residuals(
    parameters: np.ndarray,
    *,
    initial: np.ndarray,
    target_length_m: float,
    views: tuple[dict[str, Any], ...],
    points: np.ndarray | None,
    config: FilamentRegistrationConfig,
) -> np.ndarray:
    def weighted(values: np.ndarray, weight: float) -> np.ndarray:
        # Each cue is one evidence block. Normalization prevents hundreds of
        # correlated silhouette pixels from overwhelming topology constraints.
        return np.sqrt(weight / max(len(values), 1)) * values

    def robust_observation(values: np.ndarray) -> np.ndarray:
        # This residual has the same squared objective as soft-L1, but only for
        # observation cues. Structural constraints remain genuinely quadratic.
        scaled = values / config.robust_scale
        return config.robust_scale * np.sqrt(2.0 * (np.sqrt(1.0 + scaled**2) - 1.0))

    centerline, raw_direction_norm = _decode_filament_parameters(
        parameters,
        node_count=len(initial),
        target_length_m=target_length_m,
    )
    samples = sample_filament_centerline(
        centerline, samples_per_edge=config.samples_per_edge
    )
    residuals = []
    maximum_silhouette_scaled = (
        config.maximum_silhouette_distance_px / config.silhouette_distance_scale_px
    )
    for view in views:
        camera_weight = float(view["fit_camera_weight"])
        pixels, visible = _project_world_points(
            samples,
            view["intrinsics"],
            view["camera_to_world"],
            view["mask"].shape,
        )
        rows, columns = _pixel_indices(pixels, view["mask"].shape)
        occluded = visible & view["gripper_mask"][rows, columns]
        evaluable = visible & ~occluded
        containment = np.full(len(samples), maximum_silhouette_scaled)
        containment[occluded] = 0.0
        if np.any(evaluable):
            distance = view["distance_to_mask_px"][rows[evaluable], columns[evaluable]]
            containment[evaluable] = np.minimum(
                distance / config.silhouette_distance_scale_px,
                maximum_silhouette_scaled,
            )
        residuals.append(
            weighted(
                robust_observation(containment),
                config.containment_weight * camera_weight,
            )
        )

        eligible_pixels = pixels[evaluable]
        if len(eligible_pixels):
            coverage = np.linalg.norm(
                view["mask_pixels"][:, None, :] - eligible_pixels[None, :, :],
                axis=2,
            ).min(axis=1)
        else:
            coverage = np.full(
                len(view["mask_pixels"]), config.maximum_silhouette_distance_px
            )
        residuals.append(
            weighted(
                robust_observation(
                    np.minimum(
                        coverage / config.silhouette_distance_scale_px,
                        maximum_silhouette_scaled,
                    )
                ),
                config.coverage_weight * camera_weight,
            )
        )

    if points is not None and config.point_cloud_weight > 0.0:
        distance = np.linalg.norm(points[:, None, :] - samples[None, :, :], axis=2).min(
            axis=1
        )
        residuals.append(
            weighted(
                robust_observation(
                    np.minimum(
                        distance / config.point_distance_scale_m,
                        config.maximum_point_distance_m / config.point_distance_scale_m,
                    )
                ),
                config.point_cloud_weight,
            )
        )

    target_edge = target_length_m / (len(centerline) - 1)
    residuals.append(weighted(raw_direction_norm - 1.0, config.equal_edge_weight))
    residuals.append(
        weighted(
            np.diff(centerline, n=2, axis=0).reshape(-1) / target_edge,
            config.curvature_weight,
        )
    )
    residuals.append(
        weighted(
            (centerline - initial).reshape(-1) / target_edge,
            config.initialization_weight,
        )
    )
    return np.concatenate(residuals)


def _encode_filament_parameters(centerline_world_m: np.ndarray) -> np.ndarray:
    centerline = _validate_centerline(centerline_world_m)
    direction = np.diff(centerline, axis=0)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return np.concatenate((centerline[0], direction.reshape(-1)))


def _project_centerline_equal_edge_lengths(
    centerline_world_m: np.ndarray,
    target_length_m: float,
    *,
    iterations: int,
) -> np.ndarray:
    """Project a chain symmetrically before exact direction parameterization."""

    values = _validate_centerline(centerline_world_m).copy()
    _require(target_length_m > 0.0, "target filament length must be positive")
    _require(iterations >= 1, "length projection needs at least one iteration")
    target_edge = target_length_m / (len(values) - 1)
    for _ in range(iterations):
        for edge_index in range(len(values) - 1):
            difference = values[edge_index + 1] - values[edge_index]
            length = float(np.linalg.norm(difference))
            if length <= 1e-12:
                continue
            correction = 0.5 * (length - target_edge) * difference / length
            values[edge_index] += correction
            values[edge_index + 1] -= correction
        for edge_index in range(len(values) - 2, -1, -1):
            difference = values[edge_index + 1] - values[edge_index]
            length = float(np.linalg.norm(difference))
            if length <= 1e-12:
                continue
            correction = 0.5 * (length - target_edge) * difference / length
            values[edge_index] += correction
            values[edge_index + 1] -= correction
    length = float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())
    _require(length > 1e-12, "length projection collapsed the filament")
    centroid = np.mean(values, axis=0)
    return centroid + (values - centroid) * (target_length_m / length)


def _decode_filament_parameters(
    parameters: np.ndarray,
    *,
    node_count: int,
    target_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(parameters, dtype=np.float64)
    _require(
        values.shape == (3 + 3 * (node_count - 1),),
        "filament parameter count differs from its topology",
    )
    raw_direction = values[3:].reshape(node_count - 1, 3)
    raw_norm = np.linalg.norm(raw_direction, axis=1)
    safe_norm = np.maximum(raw_norm, 1e-8)
    direction = raw_direction / safe_norm[:, None]
    edge = direction * (target_length_m / (node_count - 1))
    centerline = np.concatenate(
        (values[:3][None, :], values[:3][None, :] + np.cumsum(edge, axis=0)),
        axis=0,
    )
    return centerline, raw_norm


def fit_multiview_filament_centerline(
    initial_centerline_world_m: np.ndarray,
    target_length_m: float,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    gripper_masks_by_camera: Mapping[str, np.ndarray] | None = None,
    observed_centerline_pixels_by_camera: Mapping[str, np.ndarray] | None = None,
    camera_reliability_by_camera: Mapping[str, float] | None = None,
    seed_points_world_m: np.ndarray | None = None,
    config: FilamentRegistrationConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a fixed-topology chain to multiview silhouettes and weak 3D seeds."""

    cfg = config or FilamentRegistrationConfig()
    initial = _validate_centerline(initial_centerline_world_m).copy()
    _require(target_length_m > 0.0, "target filament length must be positive")
    cameras = _validate_views(
        masks_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        gripper_masks_by_camera,
    )
    if observed_centerline_pixels_by_camera is not None:
        _require(
            all(camera in observed_centerline_pixels_by_camera for camera in cameras),
            "filament registration is missing observed medial paths",
        )
    reliability = _validate_camera_reliability(cameras, camera_reliability_by_camera)
    fit_camera_weights = reliability / np.mean(reliability)
    try:
        from scipy.ndimage import distance_transform_edt
        from scipy.optimize import least_squares
    except ImportError as error:  # pragma: no cover - scipy is a graph dependency
        raise RuntimeError("SciPy is required for filament registration") from error

    views = []
    for camera, raw_reliability, fit_camera_weight in zip(
        cameras, reliability, fit_camera_weights, strict=True
    ):
        mask = np.asarray(masks_by_camera[camera], dtype=bool)
        gripper = (
            np.asarray(gripper_masks_by_camera[camera], dtype=bool)
            if gripper_masks_by_camera is not None
            else np.zeros_like(mask)
        )
        views.append(
            {
                "camera": camera,
                "prior_camera_reliability": float(raw_reliability),
                "fit_camera_weight": float(fit_camera_weight),
                "mask": mask,
                "gripper_mask": gripper,
                "intrinsics": np.asarray(
                    intrinsics_by_camera[camera], dtype=np.float64
                ),
                "camera_to_world": np.asarray(
                    camera_to_world_by_camera[camera], dtype=np.float64
                ),
                "distance_to_mask_px": distance_transform_edt(~mask),
                "mask_pixels": (
                    _subsample_rows(
                        np.asarray(
                            observed_centerline_pixels_by_camera[camera],
                            dtype=np.float64,
                        ),
                        cfg.maximum_mask_sample_count,
                    )
                    if observed_centerline_pixels_by_camera is not None
                    else _mask_pixels(mask, cfg.maximum_mask_sample_count)
                ),
            }
        )

    points = None
    if seed_points_world_m is not None:
        points = np.asarray(seed_points_world_m, dtype=np.float64)
        _require(
            points.ndim == 2 and points.shape[1] == 3 and len(points) >= 8,
            "filament seed points must have shape (N,3)",
        )
        _require(np.all(np.isfinite(points)), "filament seed points are non-finite")
        points = _subsample_rows(points, cfg.maximum_point_sample_count)

    parameterized_initial = _project_centerline_equal_edge_lengths(
        initial,
        target_length_m,
        iterations=cfg.length_projection_iterations,
    )
    initial_parameters = _encode_filament_parameters(parameterized_initial)
    keyword = {
        "initial": parameterized_initial,
        "target_length_m": float(target_length_m),
        "views": tuple(views),
        "points": points,
        "config": cfg,
    }
    initial_residual = _registration_residuals(initial_parameters, **keyword)
    lower = np.full_like(initial_parameters, -np.inf)
    upper = np.full_like(initial_parameters, np.inf)
    lower[:3] = parameterized_initial[0] - cfg.maximum_node_displacement_m
    upper[:3] = parameterized_initial[0] + cfg.maximum_node_displacement_m
    result = least_squares(
        _registration_residuals,
        initial_parameters,
        kwargs=keyword,
        bounds=(lower, upper),
        loss="linear",
        max_nfev=cfg.maximum_function_evaluations,
    )
    centerline, _ = _decode_filament_parameters(
        result.x,
        node_count=len(initial),
        target_length_m=target_length_m,
    )
    final_residual = _registration_residuals(result.x, **keyword)
    initial_support = filament_multiview_support_diagnostics(
        parameterized_initial,
        masks_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        gripper_masks_by_camera=gripper_masks_by_camera,
        observed_centerline_pixels_by_camera=(observed_centerline_pixels_by_camera),
        camera_reliability_by_camera=camera_reliability_by_camera,
        samples_per_edge=cfg.samples_per_edge,
        maximum_mask_sample_count=cfg.maximum_mask_sample_count,
    )
    final_support = filament_multiview_support_diagnostics(
        centerline,
        masks_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        gripper_masks_by_camera=gripper_masks_by_camera,
        observed_centerline_pixels_by_camera=(observed_centerline_pixels_by_camera),
        camera_reliability_by_camera=camera_reliability_by_camera,
        samples_per_edge=cfg.samples_per_edge,
        maximum_mask_sample_count=cfg.maximum_mask_sample_count,
    )
    edge = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    projection_displacement = np.linalg.norm(parameterized_initial - initial, axis=1)
    projected_initial_edge = np.linalg.norm(
        np.diff(parameterized_initial, axis=0), axis=1
    )
    return centerline, {
        "parameters": asdict(cfg),
        "camera_count": len(cameras),
        "camera_ids": list(cameras),
        "camera_reliability_by_camera": {
            camera: float(value)
            for camera, value in zip(cameras, reliability, strict=True)
        },
        "effective_camera_count": float(
            np.sum(reliability) ** 2 / np.sum(reliability**2)
        ),
        "node_count": len(centerline),
        "target_length_m": float(target_length_m),
        "fitted_length_m": float(np.sum(edge)),
        "relative_length_error": float(
            abs(np.sum(edge) - target_length_m) / target_length_m
        ),
        "edge_length_coefficient_of_variation": float(np.std(edge) / np.mean(edge)),
        "seed_point_count": 0 if points is None else len(points),
        "initial_equal_edge_projection": {
            "displacement_m": {
                "median": float(np.median(projection_displacement)),
                "p95": float(np.quantile(projection_displacement, 0.95)),
                "maximum": float(np.max(projection_displacement)),
            },
            "edge_length_coefficient_of_variation": float(
                np.std(projected_initial_edge) / np.mean(projected_initial_edge)
            ),
        },
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "initial_mean_squared_residual": float(np.mean(initial_residual**2)),
            "final_mean_squared_residual": float(np.mean(final_residual**2)),
        },
        "initial_multiview_support": initial_support,
        "fitted_multiview_support": final_support,
    }


def audit_filament_registration(
    centerline_world_m: np.ndarray,
    target_length_m: float,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    reconstructed_points_world_m: np.ndarray | None = None,
    gripper_masks_by_camera: Mapping[str, np.ndarray] | None = None,
    observed_centerline_pixels_by_camera: Mapping[str, np.ndarray] | None = None,
    camera_reliability_by_camera: Mapping[str, float] | None = None,
    config: FilamentRegistrationQAConfig | None = None,
) -> dict[str, Any]:
    """Apply topology-aware QA without using a volumetric visual-hull target."""

    cfg = config or FilamentRegistrationQAConfig()
    centerline = _validate_centerline(centerline_world_m)
    _require(target_length_m > 0.0, "target filament length must be positive")
    support = filament_multiview_support_diagnostics(
        centerline,
        masks_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        gripper_masks_by_camera=gripper_masks_by_camera,
        observed_centerline_pixels_by_camera=(observed_centerline_pixels_by_camera),
        camera_reliability_by_camera=camera_reliability_by_camera,
    )
    edge = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    length = float(np.sum(edge))
    edge_cv = float(np.std(edge) / np.mean(edge))
    relative_length_error = abs(length - target_length_m) / target_length_m
    local_geometry = None
    local_thickness_passed = True
    if reconstructed_points_world_m is not None:
        local_geometry = filament_local_geometry_diagnostics(
            reconstructed_points_world_m,
            neighbor_count=cfg.local_neighbor_count,
        )
        local_thickness_passed = (
            local_geometry["local_radial_distance_m"]["p95"]
            <= cfg.maximum_local_radial_p95_m
        )
    use_weighted_support = camera_reliability_by_camera is not None
    support_statistics = (
        support["reliability_weighted_visibility_aware_mask_support"]
        if use_weighted_support
        else support["visibility_aware_mask_support"]
    )
    coverage_statistics = (
        support["reliability_weighted_mask_coverage_p95_px"]
        if use_weighted_support
        else support["mask_coverage_p95_px"]
    )
    gates = {
        "local_filament_thickness": bool(local_thickness_passed),
        "median_visibility_aware_mask_support": bool(
            support_statistics["median"] >= cfg.minimum_median_mask_support
        ),
        "lower_quartile_visibility_aware_mask_support": bool(
            support_statistics["lower_quartile"]
            >= cfg.minimum_lower_quartile_mask_support
        ),
        "mask_coverage": bool(
            coverage_statistics["median"] <= cfg.maximum_median_mask_coverage_p95_px
        ),
        "effective_camera_count": bool(
            support["camera_reliability"]["effective_camera_count"]
            >= cfg.minimum_effective_camera_count
        ),
        "equal_edge_spacing": bool(
            edge_cv <= cfg.maximum_edge_length_coefficient_of_variation
        ),
        "shared_length_consistency": bool(
            relative_length_error <= cfg.maximum_relative_length_error
        ),
    }
    return {
        "parameters": asdict(cfg),
        "centerline_node_count": len(centerline),
        "centerline_length_m": length,
        "target_length_m": float(target_length_m),
        "relative_length_error": float(relative_length_error),
        "edge_length_coefficient_of_variation": edge_cv,
        "local_geometry": local_geometry,
        "multiview_support": support,
        "support_gate_statistics": (
            "source-reliability-weighted" if use_weighted_support else "all-view"
        ),
        "acceptance_gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": (
            "source-geometry development QA; thresholds must be frozen before "
            "evaluation on a new object"
        ),
    }


__all__ = [
    "FilamentRegistrationConfig",
    "FilamentRegistrationQAConfig",
    "audit_filament_registration",
    "filament_local_geometry_diagnostics",
    "filament_multiview_support_diagnostics",
    "fit_multiview_filament_centerline",
    "sample_filament_centerline",
]
