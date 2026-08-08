import numpy as np

from bayesian_phystwin.mask_distance import (
    _interior_mask_distance_fallback,
    interior_mask_distance,
)
from bayesian_phystwin.phystwin_cotracker3_cues import (
    project_world_points,
    triangulate_multiview_tracks,
)
from bayesian_phystwin.phystwin_motioncrafter_assimilation import (
    _mask_boundary_distance,
)


def test_boundary_distance_treats_image_exterior_as_background() -> None:
    mask = np.ones((3, 4), dtype=bool)
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(interior_mask_distance(mask), expected)
    np.testing.assert_allclose(_mask_boundary_distance(mask), expected)


def test_numpy_boundary_fallback_is_exact_euclidean() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    fallback = _interior_mask_distance_fallback(mask)
    np.testing.assert_allclose(fallback, expected)
    np.testing.assert_allclose(interior_mask_distance(mask), fallback)


def test_triangulation_rejects_behind_camera_support() -> None:
    intrinsics = np.repeat(np.eye(3)[None], 2, axis=0)
    camera_to_world = np.repeat(np.eye(4)[None], 2, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    camera_to_world[1, 2, 3] = 1.0
    point = np.array([[0.0, 0.0, 0.5]])
    tracks = np.empty((2, 1, 1, 2), dtype=float)
    depths = []
    for camera in range(2):
        tracks[camera, 0], depth = project_world_points(
            point, intrinsics[camera], camera_to_world[camera]
        )
        depths.append(float(depth[0]))

    assert depths[0] > 0.0
    assert depths[1] < 0.0
    reconstructed, error, count = triangulate_multiview_tracks(
        tracks,
        np.ones((2, 1, 1), dtype=bool),
        np.ones((2, 1, 1), dtype=float),
        intrinsics,
        camera_to_world,
    )

    np.testing.assert_array_equal(count, [[1]])
    assert np.all(np.isnan(reconstructed[0, 0]))
    assert np.isnan(error[0, 0])
