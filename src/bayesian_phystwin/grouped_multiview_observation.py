"""Disjoint-camera 3-D evidence for action-response admission."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class GroupedMultiviewConfig:
    """Target-free grouping and metric-covariance settings."""

    group_count: int = 3
    minimum_cameras_per_group: int = 2
    pixel_standard_deviation: float = 1.5
    covariance_inflation: float = 4.0
    covariance_floor_m2: float = 1e-8
    maximum_condition_number: float = 1e10

    def __post_init__(self) -> None:
        _require(self.group_count >= 2, "group_count must be at least two")
        _require(
            self.minimum_cameras_per_group >= 2,
            "minimum_cameras_per_group must be at least two",
        )
        positive = (
            self.pixel_standard_deviation,
            self.covariance_inflation,
            self.covariance_floor_m2,
            self.maximum_condition_number,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "grouped multiview scales must be finite and positive",
        )


@dataclass(frozen=True)
class GroupedMultiviewObservation:
    """Independent disjoint-camera triangulations at one prefix frame."""

    group_ids: tuple[str, ...]
    camera_groups: tuple[tuple[str, ...], ...]
    points_m: np.ndarray
    valid: np.ndarray
    covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    association_probability: np.ndarray
    inlier_view_count: np.ndarray
    median_reprojection_px: np.ndarray
    maximum_ray_angle_degrees: np.ndarray

    def __post_init__(self) -> None:
        group_count = len(self.group_ids)
        _require(
            len(self.camera_groups) == group_count,
            "camera group count changed",
        )
        _require(
            len(set(self.group_ids)) == group_count,
            "group IDs must be unique",
        )
        flattened = [
            camera for group in self.camera_groups for camera in group
        ]
        _require(
            len(flattened) == len(set(flattened)),
            "camera groups must be disjoint",
        )
        arrays = {
            "points_m": np.asarray(self.points_m, dtype=np.float64).copy(),
            "valid": np.asarray(self.valid, dtype=bool).copy(),
            "covariance_m2": np.asarray(
                self.covariance_m2, dtype=np.float64
            ).copy(),
            "prior_reliability": np.asarray(
                self.prior_reliability, dtype=np.float64
            ).copy(),
            "association_probability": np.asarray(
                self.association_probability, dtype=np.float64
            ).copy(),
            "inlier_view_count": np.asarray(
                self.inlier_view_count, dtype=np.int64
            ).copy(),
            "median_reprojection_px": np.asarray(
                self.median_reprojection_px, dtype=np.float64
            ).copy(),
            "maximum_ray_angle_degrees": np.asarray(
                self.maximum_ray_angle_degrees, dtype=np.float64
            ).copy(),
        }
        points = arrays["points_m"]
        _require(
            points.ndim == 3
            and points.shape[0] == group_count
            and points.shape[2] == 3,
            "points_m must have shape (G, N, 3)",
        )
        expected = points.shape[:2]
        for name in (
            "valid",
            "prior_reliability",
            "association_probability",
            "inlier_view_count",
            "median_reprojection_px",
            "maximum_ray_angle_degrees",
        ):
            _require(arrays[name].shape == expected, f"{name} shape changed")
        _require(
            arrays["covariance_m2"].shape == (*expected, 3, 3),
            "covariance_m2 shape changed",
        )
        valid = arrays["valid"]
        _require(
            np.all(np.isfinite(points[valid])),
            "valid grouped point is not finite",
        )
        _require(
            np.all(np.isnan(points[~valid])),
            "invalid grouped points must be NaN",
        )
        covariance = arrays["covariance_m2"]
        _require(
            np.all(np.isfinite(covariance[valid])),
            "valid grouped covariance is not finite",
        )
        for matrix in covariance[valid]:
            _require(
                np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12),
                "grouped covariance is not symmetric",
            )
            _require(
                np.min(np.linalg.eigvalsh(matrix)) >= -1e-12,
                "grouped covariance is not positive semidefinite",
            )
        for name in ("prior_reliability", "association_probability"):
            value = arrays[name]
            _require(
                np.all(np.isfinite(value))
                and np.all((value >= 0.0) & (value <= 1.0)),
                f"{name} must lie in [0, 1]",
            )
            _require(
                np.all(value[~valid] == 0.0),
                f"invalid {name} must be zero",
            )
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def partition_disjoint_camera_groups(
    camera_names: Sequence[str],
    camera_origins_m: Mapping[str, np.ndarray],
    reference_points_m: np.ndarray,
    *,
    config: GroupedMultiviewConfig | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Partition cameras by azimuth into spatially spread disjoint groups."""

    cfg = config or GroupedMultiviewConfig()
    names = tuple(str(name) for name in camera_names)
    _require(len(names) == len(set(names)), "camera names must be unique")
    _require(
        len(names) >= cfg.group_count * cfg.minimum_cameras_per_group,
        "too few cameras for the requested disjoint groups",
    )
    points = np.asarray(reference_points_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) >= 1
        and np.all(np.isfinite(points)),
        "reference_points_m must contain finite 3-D points",
    )
    centroid = np.mean(points, axis=0)
    ranked: list[tuple[float, str]] = []
    for name in names:
        _require(name in camera_origins_m, f"missing camera origin for {name}")
        origin = np.asarray(camera_origins_m[name], dtype=np.float64)
        _require(
            origin.shape == (3,) and np.all(np.isfinite(origin)),
            f"invalid camera origin for {name}",
        )
        delta = origin - centroid
        ranked.append((float(np.arctan2(delta[1], delta[0])), name))
    ordered = tuple(name for _, name in sorted(ranked))
    groups = tuple(
        tuple(ordered[index:: cfg.group_count])
        for index in range(cfg.group_count)
    )
    _require(
        all(len(group) >= cfg.minimum_cameras_per_group for group in groups),
        "camera partition produced an undersized group",
    )
    return groups


def triangulation_covariance_m2(
    point_m: np.ndarray,
    inlier_cameras: Sequence[str],
    projection_matrices: Mapping[str, np.ndarray],
    *,
    config: GroupedMultiviewConfig | None = None,
) -> np.ndarray:
    """Linearized metric covariance from pixel reprojection information."""

    cfg = config or GroupedMultiviewConfig()
    point = np.asarray(point_m, dtype=np.float64)
    _require(
        point.shape == (3,) and np.all(np.isfinite(point)),
        "point_m must be a finite 3-D point",
    )
    cameras = tuple(str(camera) for camera in inlier_cameras)
    _require(
        len(cameras) >= cfg.minimum_cameras_per_group,
        "too few inlier cameras for metric covariance",
    )
    jacobians: list[np.ndarray] = []
    homogeneous_point = np.append(point, 1.0)
    for camera in cameras:
        _require(camera in projection_matrices, f"missing projection for {camera}")
        projection = np.asarray(projection_matrices[camera], dtype=np.float64)
        _require(projection.shape == (3, 4), "projection matrix shape changed")
        projected = projection @ homogeneous_point
        depth = float(projected[2])
        _require(depth > 1e-12, "triangulated point lies behind a camera")
        numerator = (
            projection[:2, :3] * depth
            - projected[:2, None] * projection[2, :3]
        )
        jacobians.append(numerator / (depth * depth))
    jacobian = np.concatenate(jacobians, axis=0)
    information = (
        jacobian.T @ jacobian / (cfg.pixel_standard_deviation**2)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    _require(
        eigenvalues[-1] > 0.0
        and eigenvalues[0] > eigenvalues[-1] / cfg.maximum_condition_number,
        "triangulation information is ill-conditioned",
    )
    covariance = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
    covariance *= cfg.covariance_inflation
    covariance += cfg.covariance_floor_m2 * np.eye(3)
    return 0.5 * (covariance + covariance.T)


Triangulator = Callable[
    [Mapping[str, np.ndarray], np.ndarray],
    tuple[np.ndarray | None, Mapping[str, Any]],
]


def triangulate_disjoint_camera_groups(
    tracks_by_camera: Mapping[str, Mapping[int, np.ndarray]],
    point_ids: np.ndarray,
    initial_points_m: np.ndarray,
    camera_groups: Sequence[Sequence[str]],
    projection_matrices: Mapping[str, np.ndarray],
    triangulator: Triangulator,
    *,
    config: GroupedMultiviewConfig | None = None,
) -> GroupedMultiviewObservation:
    """Triangulate each material identity independently in disjoint panels."""

    cfg = config or GroupedMultiviewConfig()
    ids = np.asarray(point_ids, dtype=np.int64)
    initial = np.asarray(initial_points_m, dtype=np.float64)
    _require(
        ids.ndim == 1 and len(ids) == len(np.unique(ids)),
        "point_ids must be a unique vector",
    )
    _require(
        initial.ndim == 2
        and initial.shape[1] == 3
        and np.all(np.isfinite(initial)),
        "initial_points_m must have shape (N, 3)",
    )
    _require(
        np.all((ids >= 0) & (ids < len(initial))),
        "point ID exceeds initial points",
    )
    groups = tuple(tuple(str(camera) for camera in group) for group in camera_groups)
    _require(len(groups) == cfg.group_count, "camera group count changed")
    flattened = [camera for group in groups for camera in group]
    _require(
        len(flattened) == len(set(flattened)),
        "camera groups must be disjoint",
    )
    _require(
        all(len(group) >= cfg.minimum_cameras_per_group for group in groups),
        "camera group is too small",
    )
    group_ids = tuple(f"disjoint-camera-group-{index}" for index in range(len(groups)))
    shape = (len(groups), len(ids))
    points: np.ndarray = np.full((*shape, 3), np.nan, dtype=np.float64)
    valid: np.ndarray = np.zeros(shape, dtype=bool)
    covariance: np.ndarray = np.full(
        (*shape, 3, 3), np.nan, dtype=np.float64
    )
    reliability: np.ndarray = np.zeros(shape, dtype=np.float64)
    association: np.ndarray = np.zeros(shape, dtype=np.float64)
    inlier_count: np.ndarray = np.zeros(shape, dtype=np.int64)
    reprojection: np.ndarray = np.full(shape, np.nan, dtype=np.float64)
    ray_angle: np.ndarray = np.full(shape, np.nan, dtype=np.float64)
    for group_index, camera_group in enumerate(groups):
        for point_index, point_id in enumerate(ids):
            observations = {
                camera: np.asarray(tracks_by_camera[camera][int(point_id)])
                for camera in camera_group
                if camera in tracks_by_camera
                and int(point_id) in tracks_by_camera[camera]
            }
            point, diagnostic = triangulator(
                observations,
                initial[int(point_id)],
            )
            if point is None or diagnostic.get("accepted") is not True:
                continue
            inlier_cameras = tuple(
                str(camera) for camera in diagnostic["inlier_cameras"]
            )
            try:
                point_covariance = triangulation_covariance_m2(
                    point,
                    inlier_cameras,
                    projection_matrices,
                    config=cfg,
                )
            except ValueError:
                continue
            median_reprojection = float(
                diagnostic["median_reprojection_error_px"]
            )
            view_count = int(diagnostic["inlier_view_count"])
            angle = float(diagnostic["maximum_ray_angle_degrees"])
            points[group_index, point_index] = point
            valid[group_index, point_index] = True
            covariance[group_index, point_index] = point_covariance
            inlier_count[group_index, point_index] = view_count
            reprojection[group_index, point_index] = median_reprojection
            ray_angle[group_index, point_index] = angle
            reliability[group_index, point_index] = float(
                np.exp(
                    -0.5
                    * (median_reprojection / cfg.pixel_standard_deviation) ** 2
                )
            )
            association[group_index, point_index] = min(
                1.0,
                view_count / len(camera_group),
            )
    return GroupedMultiviewObservation(
        group_ids=group_ids,
        camera_groups=groups,
        points_m=points,
        valid=valid,
        covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        inlier_view_count=inlier_count,
        median_reprojection_px=reprojection,
        maximum_ray_angle_degrees=ray_angle,
    )


__all__ = [
    "GroupedMultiviewConfig",
    "GroupedMultiviewObservation",
    "partition_disjoint_camera_groups",
    "triangulate_disjoint_camera_groups",
    "triangulation_covariance_m2",
]
