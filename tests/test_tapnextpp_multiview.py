from __future__ import annotations

import numpy as np

from bayesian_phystwin.tapnextpp_multiview import (
    TAPNextPPMultiviewConfig,
    camera_projection_matrix,
    conservative_triangulation_covariance_m2,
    fuse_causal_multiview_tracks,
    project_world_point,
)


def _camera_fixture() -> tuple[np.ndarray, np.ndarray]:
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    intrinsics = np.repeat(intrinsic[None], 3, axis=0)
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[0, 0, 3] = -0.2
    poses[1, 0, 3] = 0.2
    poses[2, 1, 3] = 0.2
    return intrinsics, poses


def _render_observations(
    points_world_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intrinsics, poses = _camera_fixture()
    frame_count = len(points_world_m)
    tracks = np.zeros((3, frame_count, 1, 2), dtype=np.float32)
    depths = np.ones((3, frame_count, 100, 100), dtype=np.float32)
    masks = np.ones_like(depths, dtype=bool)
    for camera in range(3):
        projection = camera_projection_matrix(
            intrinsics[camera],
            poses[camera],
        )
        for frame, point in enumerate(points_world_m):
            tracks[camera, frame, 0], _ = project_world_point(
                point,
                projection,
            )
    return tracks, depths, masks, intrinsics


def test_multiview_lift_recovers_metric_motion() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]])
    tracks, depths, masks, intrinsics = _render_observations(points)
    _, poses = _camera_fixture()
    result = fuse_causal_multiview_tracks(
        tracks,
        np.ones(tracks.shape[:3]),
        depths,
        masks,
        intrinsics,
        poses,
        points[:1],
    )
    assert np.all(result["accepted_support"])
    assert np.all(result["support_view_count"] == 3)
    assert np.allclose(result["trajectory_world_m"][:, 0], points, atol=1e-5)


def test_gross_camera_outlier_is_rejected_without_state_residual() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]])
    tracks, depths, masks, intrinsics = _render_observations(points)
    tracks[2, 1, 0, 0] += 30.0
    _, poses = _camera_fixture()
    first = fuse_causal_multiview_tracks(
        tracks,
        np.ones(tracks.shape[:3]),
        depths,
        masks,
        intrinsics,
        poses,
        points[:1],
    )
    arbitrary_physical_state_residual = np.asarray([0.0, 10.0])
    second = fuse_causal_multiview_tracks(
        tracks,
        np.ones(tracks.shape[:3]),
        depths,
        masks,
        intrinsics,
        poses,
        points[:1],
    )
    assert arbitrary_physical_state_residual[1] > arbitrary_physical_state_residual[0]
    assert np.array_equal(
        first["observation_reliability"],
        second["observation_reliability"],
    )
    assert first["accepted_support"][1, 0]
    assert first["support_view_count"][1, 0] == 2
    assert np.allclose(
        first["trajectory_world_m"][1, 0],
        points[1],
        atol=1e-5,
    )


def test_correlated_camera_duplication_cannot_erase_bias_floor() -> None:
    intrinsics, poses = _camera_fixture()
    point = np.asarray([0.0, 0.0, 1.0])
    config = TAPNextPPMultiviewConfig(
        shared_bias_standard_deviation_m=0.005
    )
    repeated_intrinsics = np.repeat(intrinsics, 20, axis=0)
    repeated_poses = np.repeat(poses, 20, axis=0)
    covariance = conservative_triangulation_covariance_m2(
        point,
        repeated_intrinsics,
        repeated_poses,
        np.ones(len(repeated_intrinsics)),
        config=config,
    )
    minimum_variance = config.shared_bias_standard_deviation_m**2
    assert np.min(np.linalg.eigvalsh(covariance)) >= minimum_variance * (
        1.0 - 1e-10
    )


def test_unknown_correlation_covariance_exceeds_naive_independence() -> None:
    intrinsics, poses = _camera_fixture()
    point = np.asarray([0.0, 0.0, 1.0])
    config = TAPNextPPMultiviewConfig()
    covariance = conservative_triangulation_covariance_m2(
        point,
        intrinsics,
        poses,
        np.ones(3),
        config=config,
    )
    information = np.zeros((3, 3))
    world_to_camera = np.linalg.inv(poses)
    for intrinsic, transform in zip(
        intrinsics,
        world_to_camera,
        strict=True,
    ):
        camera = transform[:3] @ np.append(point, 1.0)
        x, y, z = camera
        camera_jacobian = np.asarray(
            [
                [
                    intrinsic[0, 0] / z,
                    0.0,
                    -intrinsic[0, 0] * x / (z**2),
                ],
                [
                    0.0,
                    intrinsic[1, 1] / z,
                    -intrinsic[1, 1] * y / (z**2),
                ],
            ]
        )
        jacobian = camera_jacobian @ transform[:3, :3]
        information += (jacobian.T @ jacobian) / (
            config.pixel_standard_deviation_px**2
        )
    naive = np.linalg.pinv(information, hermitian=True)
    assert np.all(
        np.linalg.eigvalsh(covariance - naive) >= -1e-12
    )


def test_assignment_mixture_spread_increases_covariance() -> None:
    intrinsics, poses = _camera_fixture()
    point = np.asarray([0.0, 0.0, 1.0])
    base = conservative_triangulation_covariance_m2(
        point,
        intrinsics,
        poses,
        np.ones(3),
    )
    spread = np.diag([1e-4, 0.0, 0.0])
    ambiguous = conservative_triangulation_covariance_m2(
        point,
        intrinsics,
        poses,
        np.ones(3),
        mixture_spread_m2=spread,
    )
    assert ambiguous[0, 0] > base[0, 0]
