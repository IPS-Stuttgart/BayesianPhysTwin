import numpy as np

from bayesian_phystwin.phystwin_cotracker3_cues import (
    project_world_points,
    triangulate_multiview_tracks,
)


def test_project_world_points_uses_camera_to_world_pose() -> None:
    intrinsic = np.array(
        [[100.0, 0.0, 20.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
    )
    camera_to_world = np.eye(4)
    camera_to_world[0, 3] = 1.0

    pixels, depth = project_world_points(
        np.array([[1.0, 0.0, 5.0], [2.0, 1.0, 5.0]]),
        intrinsic,
        camera_to_world,
    )

    np.testing.assert_allclose(pixels, [[20.0, 10.0], [40.0, 30.0]])
    np.testing.assert_allclose(depth, [5.0, 5.0])


def test_triangulate_multiview_tracks_recovers_point_and_zero_error() -> None:
    intrinsics = np.repeat(np.eye(3)[None], 2, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    camera_to_world = np.repeat(np.eye(4)[None], 2, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    point = np.array([[0.25, -0.10, 5.0]])
    tracks = np.empty((2, 1, 1, 2), dtype=float)
    for camera in range(2):
        tracks[camera, 0], _ = project_world_points(
            point,
            intrinsics[camera],
            camera_to_world[camera],
        )

    reconstructed, error, count = triangulate_multiview_tracks(
        tracks,
        np.ones((2, 1, 1), dtype=bool),
        np.ones((2, 1, 1), dtype=float),
        intrinsics,
        camera_to_world,
    )

    np.testing.assert_allclose(reconstructed[0, 0], point[0], atol=1e-7)
    np.testing.assert_allclose(error, 0.0, atol=1e-7)
    np.testing.assert_array_equal(count, [[2]])


def test_triangulate_multiview_tracks_ignores_invalid_nan_view() -> None:
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    camera_to_world = np.repeat(np.eye(4)[None], 3, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    camera_to_world[2, 1, 3] = 1.0
    point = np.array([[0.25, -0.10, 5.0]])
    tracks = np.full((3, 1, 1, 2), np.nan, dtype=float)
    for camera in range(2):
        tracks[camera, 0], _ = project_world_points(
            point,
            intrinsics[camera],
            camera_to_world[camera],
        )
    valid = np.array([[[True]], [[True]], [[False]]])

    reconstructed, error, count = triangulate_multiview_tracks(
        tracks,
        valid,
        valid.astype(float),
        intrinsics,
        camera_to_world,
    )

    np.testing.assert_allclose(reconstructed[0, 0], point[0], atol=1e-7)
    np.testing.assert_allclose(error, 0.0, atol=1e-7)
    np.testing.assert_array_equal(count, [[2]])
