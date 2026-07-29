"""Direct causal depth endpoint observations for sentinel-debiased updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .deform360_dynamic_query import projection_matrices
from .deform360_dynamic_tapnextpp_assimilation import (
    BirthAnchoredMeasurements,
)
from .deform360_sentinel_query_schedule import (
    Deform360SentinelQuerySchedule,
)
from .tapnextpp_birth_association import (
    SET_VALUED_COVARIANCE_ASSOCIATION,
    BirthAssociationConfig,
    propose_birth_query_pixels,
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _solve_spd(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    cholesky = np.linalg.cholesky(matrix)
    return np.linalg.solve(
        cholesky.T,
        np.linalg.solve(cholesky, right_hand_side),
    )


@dataclass(frozen=True)
class DirectDepthEndpointConfig:
    """Frozen uncertainty and support settings for the V8 source arm."""

    minimum_camera_support: int = 3
    search_radius_px: int = 12
    depth_scale_m: float = 0.03
    minimum_candidate_count: int = 1
    depth_standard_deviation_m: float = 0.005
    pixel_quantization_variance_px2: float = 0.25
    covariance_floor_m2: float = 1e-10
    temporal_covariance_multiplier: float = 2.0

    def __post_init__(self) -> None:
        _require(
            self.minimum_camera_support >= 3,
            "direct depth requires at least three cameras",
        )
        _require(self.search_radius_px >= 1, "search radius must be positive")
        for name, value in (
            ("depth_scale_m", self.depth_scale_m),
            (
                "depth_standard_deviation_m",
                self.depth_standard_deviation_m,
            ),
            ("covariance_floor_m2", self.covariance_floor_m2),
            (
                "temporal_covariance_multiplier",
                self.temporal_covariance_multiplier,
            ),
        ):
            _require(
                np.isfinite(value) and value > 0.0,
                f"{name} must be positive",
            )
        _require(
            self.minimum_candidate_count >= 1,
            "minimum candidate count must be positive",
        )
        _require(
            np.isfinite(self.pixel_quantization_variance_px2)
            and self.pixel_quantization_variance_px2 >= 0.0,
            "pixel quantization variance must be nonnegative",
        )

    def association_config(self) -> BirthAssociationConfig:
        return BirthAssociationConfig(
            search_radius_px=self.search_radius_px,
            depth_scale_m=self.depth_scale_m,
            minimum_candidate_count=self.minimum_candidate_count,
            association_mode=SET_VALUED_COVARIANCE_ASSOCIATION,
        )


@dataclass(frozen=True)
class DirectDepthEndpointObservations:
    """Two endpoint point beliefs with conservative view fusion."""

    endpoint_frames: np.ndarray
    entity_ids: np.ndarray
    point_world_m: np.ndarray
    covariance_m2: np.ndarray
    accepted_support: np.ndarray
    association_probability: np.ndarray
    support_count: np.ndarray
    maximum_view_scatter_m: np.ndarray
    config: DirectDepthEndpointConfig

    def __post_init__(self) -> None:
        frames = _readonly(self.endpoint_frames, dtype=np.int64)
        entities = _readonly(self.entity_ids, dtype=np.int64)
        points = _readonly(self.point_world_m, dtype=np.float64)
        covariance = _readonly(self.covariance_m2, dtype=np.float64)
        accepted = _readonly(self.accepted_support, dtype=bool)
        association = _readonly(
            self.association_probability,
            dtype=np.float64,
        )
        support = _readonly(self.support_count, dtype=np.int64)
        scatter = _readonly(
            self.maximum_view_scatter_m,
            dtype=np.float64,
        )
        endpoint_count = len(frames)
        entity_count = len(entities)
        _require(
            frames.shape == (2,) and int(frames[0]) < int(frames[1]),
            "direct-depth endpoint frames are invalid",
        )
        _require(
            entities.shape == (entity_count,)
            and entity_count > 0
            and len(np.unique(entities)) == entity_count,
            "direct-depth entity IDs are invalid",
        )
        _require(
            points.shape == (endpoint_count, entity_count, 3)
            and covariance.shape
            == (
                endpoint_count,
                entity_count,
                3,
                3,
            )
            and accepted.shape
            == association.shape
            == support.shape
            == scatter.shape
            == (endpoint_count, entity_count),
            "direct-depth endpoint arrays changed shape",
        )
        _require(
            np.all(np.isfinite(points[accepted]))
            and np.all(np.isnan(points[~accepted]))
            and np.all(np.isfinite(covariance[accepted]))
            and np.all(np.isnan(covariance[~accepted])),
            "direct-depth finiteness differs from support",
        )
        _require(
            np.all(np.isfinite(association))
            and np.all((association >= 0.0) & (association <= 1.0))
            and np.all(support >= 0)
            and np.all(np.isfinite(scatter))
            and np.all(scatter >= 0.0),
            "direct-depth metadata is invalid",
        )
        _require(
            np.array_equal(
                accepted,
                support >= self.config.minimum_camera_support,
            ),
            "direct-depth support gate changed",
        )
        for matrix in covariance[accepted]:
            _require(
                np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
                and np.min(np.linalg.eigvalsh(matrix)) > 0.0,
                "direct-depth covariance is not positive definite",
            )
        object.__setattr__(self, "endpoint_frames", frames)
        object.__setattr__(self, "entity_ids", entities)
        object.__setattr__(self, "point_world_m", points)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "accepted_support", accepted)
        object.__setattr__(self, "association_probability", association)
        object.__setattr__(self, "support_count", support)
        object.__setattr__(self, "maximum_view_scatter_m", scatter)


def _backproject_with_covariance(
    pixel_xy: np.ndarray,
    depth_m: float,
    assignment_covariance_xyd: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    config: DirectDepthEndpointConfig,
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(pixel_xy, dtype=np.float64)
    assignment = np.asarray(assignment_covariance_xyd, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        pixel.shape == (2,)
        and np.all(np.isfinite(pixel))
        and np.isfinite(depth_m)
        and depth_m > 0.0,
        "direct-depth pixel or depth is invalid",
    )
    _require(
        assignment.shape == (3, 3)
        and np.all(np.isfinite(assignment))
        and np.allclose(assignment, assignment.T, rtol=0.0, atol=1e-12),
        "assignment covariance is invalid",
    )
    inverse = np.linalg.inv(matrix)
    ray = inverse @ np.asarray([pixel[0], pixel[1], 1.0])
    camera_point = depth_m * ray
    rotation = pose[:3, :3]
    world = rotation @ camera_point + pose[:3, 3]
    jacobian = np.column_stack(
        (
            rotation @ (depth_m * inverse[:, 0]),
            rotation @ (depth_m * inverse[:, 1]),
            rotation @ ray,
        )
    )
    input_covariance = assignment.copy()
    input_covariance[:2, :2] += config.pixel_quantization_variance_px2 * np.eye(2)
    input_covariance[2, 2] += config.depth_standard_deviation_m**2
    covariance = (
        jacobian @ input_covariance @ jacobian.T
        + config.covariance_floor_m2 * np.eye(3)
    )
    return world, covariance


def _fuse_unknown_correlation(
    means: np.ndarray,
    covariances: np.ndarray,
    covariance_floor_m2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    count = len(means)
    _require(count > 0, "cannot fuse an empty direct-depth view set")
    weight = 1.0 / count
    precision: np.ndarray = np.zeros((3, 3), dtype=np.float64)
    information: np.ndarray = np.zeros(3, dtype=np.float64)
    for mean, covariance in zip(means, covariances, strict=True):
        local_precision = _solve_spd(covariance, np.eye(3))
        precision += weight * local_precision
        information += weight * (local_precision @ mean)
    fused_covariance = _solve_spd(precision, np.eye(3))
    fused_mean = fused_covariance @ information
    scatter_squared = np.sum(np.square(means - fused_mean), axis=1)
    maximum_scatter_squared = float(np.max(scatter_squared))
    fused_covariance = fused_covariance + (
        maximum_scatter_squared + covariance_floor_m2
    ) * np.eye(3)
    return (
        fused_mean,
        fused_covariance,
        float(np.sqrt(maximum_scatter_squared)),
    )


def build_direct_depth_endpoint_observations(
    physical_positions_m: np.ndarray,
    schedule: Deform360SentinelQuerySchedule,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    *,
    config: DirectDepthEndpointConfig | None = None,
) -> DirectDepthEndpointObservations:
    """Associate and fuse sparse depth endpoints without an RGB carrier."""

    return build_direct_depth_observations_for_entities(
        physical_positions_m,
        schedule.entity_ids,
        np.asarray(
            [
                schedule.config.query_birth_frame,
                schedule.config.query_update_frame,
            ],
            dtype=np.int64,
        ),
        intrinsics,
        camera_to_world,
        depths_m,
        object_masks,
        config=config,
    )


def build_direct_depth_observations_for_entities(
    physical_positions_m: np.ndarray,
    entity_ids: np.ndarray,
    endpoint_frames: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    *,
    config: DirectDepthEndpointConfig | None = None,
) -> DirectDepthEndpointObservations:
    """Associate specified graph identities at two causal depth endpoints."""

    cfg = config or DirectDepthEndpointConfig()
    physical = np.asarray(physical_positions_m, dtype=np.float64)
    entities = np.asarray(entity_ids, dtype=np.int64)
    frames = np.asarray(endpoint_frames, dtype=np.int64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    depths = np.asarray(depths_m)
    masks = np.asarray(object_masks, dtype=bool)
    camera_count = len(matrices)
    _require(
        physical.ndim == 3
        and physical.shape[2] == 3
        and len(physical) > int(frames[1])
        and np.all(np.isfinite(physical[frames][:, entities])),
        "direct-depth physical endpoints are invalid",
    )
    _require(
        matrices.shape == (camera_count, 3, 3)
        and poses.shape == (camera_count, 4, 4)
        and depths.ndim == 4
        and depths.shape[0] == camera_count
        and depths.shape[1] > int(frames[1])
        and masks.shape == depths.shape,
        "direct-depth camera inputs are invalid",
    )
    _require(
        camera_count >= cfg.minimum_camera_support,
        "direct-depth camera count is insufficient",
    )
    projections = projection_matrices(matrices, poses)
    endpoint_count = len(frames)
    entity_count = len(entities)
    points = np.full((endpoint_count, entity_count, 3), np.nan)
    covariance = np.full(
        (endpoint_count, entity_count, 3, 3),
        np.nan,
    )
    accepted: np.ndarray = np.zeros(
        (endpoint_count, entity_count),
        dtype=bool,
    )
    association = np.zeros((endpoint_count, entity_count))
    support: np.ndarray = np.zeros(
        (endpoint_count, entity_count),
        dtype=np.int64,
    )
    scatter = np.zeros((endpoint_count, entity_count))
    association_cfg = cfg.association_config()
    for endpoint_index, frame in enumerate(frames):
        proposal = propose_birth_query_pixels(
            physical[frame, entities],
            projections,
            poses,
            depths[:, frame],
            masks[:, frame],
            config=association_cfg,
        )
        valid = np.asarray(proposal["valid"], dtype=bool)
        for entity_index in range(entity_count):
            views = np.flatnonzero(valid[:, entity_index])
            support[endpoint_index, entity_index] = len(views)
            if len(views) < cfg.minimum_camera_support:
                continue
            view_means: list[np.ndarray] = []
            view_covariances: list[np.ndarray] = []
            view_probabilities: list[float] = []
            for camera in views:
                pixel = proposal["candidate_mean_xy"][
                    camera,
                    entity_index,
                ]
                depth = float(
                    proposal["candidate_mean_depth_m"][
                        camera,
                        entity_index,
                    ]
                )
                mean, local_covariance = _backproject_with_covariance(
                    pixel,
                    depth,
                    proposal["candidate_xyd_covariance"][
                        camera,
                        entity_index,
                    ],
                    matrices[camera],
                    poses[camera],
                    cfg,
                )
                view_means.append(mean)
                view_covariances.append(local_covariance)
                view_probabilities.append(
                    float(
                        proposal["association_probability"][
                            camera,
                            entity_index,
                        ]
                    )
                )
            fused_mean, fused_covariance, maximum_scatter = _fuse_unknown_correlation(
                np.asarray(view_means),
                np.asarray(view_covariances),
                cfg.covariance_floor_m2,
            )
            points[endpoint_index, entity_index] = fused_mean
            covariance[endpoint_index, entity_index] = fused_covariance
            accepted[endpoint_index, entity_index] = True
            association[endpoint_index, entity_index] = min(view_probabilities)
            scatter[endpoint_index, entity_index] = maximum_scatter
    return DirectDepthEndpointObservations(
        endpoint_frames=frames,
        entity_ids=entities,
        point_world_m=points,
        covariance_m2=covariance,
        accepted_support=accepted,
        association_probability=association,
        support_count=support,
        maximum_view_scatter_m=scatter,
        config=cfg,
    )


def build_direct_depth_birth_anchored_measurements(
    observations: DirectDepthEndpointObservations,
    physical_prediction_m: np.ndarray,
) -> BirthAnchoredMeasurements:
    """Convert endpoint point beliefs to conservative displacement evidence."""

    physical = np.asarray(physical_prediction_m, dtype=np.float64)
    entities = observations.entity_ids
    birth, update = map(int, observations.endpoint_frames)
    _require(
        physical.ndim == 3
        and physical.shape[2] == 3
        and len(physical) > update
        and np.all((entities >= 0) & (entities < physical.shape[1]))
        and np.all(np.isfinite(physical)),
        "direct-depth physical prediction is invalid",
    )
    measurement = np.full(physical.shape, np.nan, dtype=np.float64)
    covariance = np.full((*physical.shape[:2], 3, 3), np.nan)
    reliability = np.zeros(physical.shape[:2])
    association = np.zeros(physical.shape[:2])
    available = np.zeros(physical.shape[:2], dtype=bool)
    endpoint_supported = np.all(observations.accepted_support, axis=0)
    selected = entities[endpoint_supported]
    if len(selected):
        local = np.flatnonzero(endpoint_supported)
        observed_displacement = (
            observations.point_world_m[1, local] - observations.point_world_m[0, local]
        )
        measurement[update, selected] = (
            physical[birth, selected] + observed_displacement
        )
        covariance[update, selected] = (
            observations.config.temporal_covariance_multiplier
            * (
                observations.covariance_m2[0, local]
                + observations.covariance_m2[1, local]
            )
        )
        reliability[update, selected] = 1.0
        association[update, selected] = np.sqrt(
            observations.association_probability[0, local]
            * observations.association_probability[1, local]
        )
        available[update, selected] = True
    return BirthAnchoredMeasurements(
        measurement_m=measurement,
        covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        available=available,
        entity_ids=entities,
    )


__all__ = [
    "DirectDepthEndpointConfig",
    "DirectDepthEndpointObservations",
    "build_direct_depth_birth_anchored_measurements",
    "build_direct_depth_endpoint_observations",
    "build_direct_depth_observations_for_entities",
]
