from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.grouped_multiview_observation import (
    partition_disjoint_camera_groups,
    triangulate_disjoint_camera_groups,
    triangulation_covariance_m2,
)


def _camera_geometry() -> tuple[
    tuple[str, ...],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    names = tuple(f"camera-{index}" for index in range(6))
    origins: dict[str, np.ndarray] = {}
    projections: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        angle = 2.0 * np.pi * index / len(names)
        origin = np.asarray([3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.0])
        origins[name] = origin
        forward = -origin / np.linalg.norm(origin)
        up_seed = np.asarray([0.0, 0.0, 1.0])
        right = np.cross(forward, up_seed)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation_world_to_camera = np.stack((right, up, forward))
        translation = -rotation_world_to_camera @ origin
        extrinsic = np.column_stack((rotation_world_to_camera, translation))
        intrinsic = np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        projections[name] = intrinsic @ extrinsic
    return names, origins, projections


def _project(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = matrix @ np.append(point, 1.0)
    return homogeneous[:2] / homogeneous[2]


def test_camera_partition_is_disjoint_deterministic_and_spread() -> None:
    names, origins, _ = _camera_geometry()
    points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    first = partition_disjoint_camera_groups(names, origins, points)
    second = partition_disjoint_camera_groups(tuple(reversed(names)), origins, points)

    assert first == second
    assert tuple(len(group) for group in first) == (2, 2, 2)
    flattened = [camera for group in first for camera in group]
    assert len(flattened) == len(set(flattened)) == 6


def test_metric_covariance_is_finite_symmetric_and_positive() -> None:
    names, _, projections = _camera_geometry()

    covariance = triangulation_covariance_m2(
        np.asarray([0.0, 0.0, 0.0]),
        names[:2],
        projections,
    )

    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_grouped_triangulation_keeps_independent_camera_panels() -> None:
    names, origins, projections = _camera_geometry()
    initial = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]
    )
    moved = initial + np.asarray([0.0, 0.0, 0.01])
    tracks = {
        camera: {
            point_id: _project(point, projections[camera])
            for point_id, point in enumerate(moved)
        }
        for camera in names
    }
    groups = partition_disjoint_camera_groups(names, origins, initial)

    def triangulator(
        observations: dict[str, np.ndarray],
        initial_point: np.ndarray,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        del initial_point
        cameras = tuple(sorted(observations))
        point_id = min(
            range(len(moved)),
            key=lambda index: float(
                sum(
                    np.linalg.norm(
                        observations[camera]
                        - _project(moved[index], projections[camera])
                    )
                    for camera in cameras
                )
            ),
        )
        return moved[point_id], {
            "accepted": True,
            "inlier_cameras": list(cameras),
            "inlier_view_count": len(cameras),
            "median_reprojection_error_px": 0.25,
            "maximum_ray_angle_degrees": 60.0,
        }

    result = triangulate_disjoint_camera_groups(
        tracks,
        np.arange(len(initial)),
        initial,
        groups,
        projections,
        triangulator,
    )

    assert result.valid.shape == (3, 3)
    assert np.all(result.valid)
    assert np.all(result.prior_reliability > 0.0)
    assert np.all(result.association_probability == 1.0)
    assert all(len(group) == 2 for group in result.camera_groups)
    assert not result.points_m.flags.writeable


def test_duplicate_camera_cannot_enter_two_independent_groups() -> None:
    names, origins, projections = _camera_geometry()
    groups = partition_disjoint_camera_groups(
        names,
        origins,
        np.zeros((2, 3)),
    )
    invalid_groups = (groups[0], groups[1], (groups[2][0], groups[0][0]))

    with pytest.raises(ValueError, match="disjoint"):
        triangulate_disjoint_camera_groups(
            {},
            np.asarray([0]),
            np.zeros((1, 3)),
            invalid_groups,
            projections,
            lambda observations, initial: (None, {}),
        )


def test_invalid_grouped_rows_have_zero_support_and_nan_geometry() -> None:
    names, origins, projections = _camera_geometry()
    groups = partition_disjoint_camera_groups(
        names,
        origins,
        np.zeros((2, 3)),
    )

    result = triangulate_disjoint_camera_groups(
        {},
        np.asarray([0, 1]),
        np.zeros((2, 3)),
        groups,
        projections,
        lambda observations, initial: (None, {"accepted": False}),
    )

    assert not np.any(result.valid)
    assert np.all(np.isnan(result.points_m))
    assert np.all(result.prior_reliability == 0.0)
    assert np.all(result.association_probability == 0.0)
