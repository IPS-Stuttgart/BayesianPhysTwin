import numpy as np

from causal4d.molmo_adapter import camera_to_world_points, farthest_point_indices


def test_farthest_points_are_deterministic_and_distributed() -> None:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [1.5, 0.0]]
    )
    first = farthest_point_indices(points, 3)
    second = farthest_point_indices(points, 3)
    assert np.array_equal(first, second)
    assert {0, 3} <= set(first.tolist())


def test_camera_forecast_transforms_to_world_coordinates() -> None:
    points = np.asarray([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    transform = np.eye(4)
    transform[:3, 3] = [0.5, -1.0, 2.0]
    world = camera_to_world_points(points, transform)
    assert np.allclose(world, points + np.asarray([0.5, -1.0, 2.0]))

