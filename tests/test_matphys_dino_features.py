import numpy as np

from bayesian_phystwin.matphys_dino_features import (
    project_world_points,
    transfer_observed_features,
)


def test_project_world_points_uses_metric_pinhole_geometry() -> None:
    points = np.array([[0.0, 0.0, 2.0], [1.0, -0.5, 2.0]])
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )

    uv, depth = project_world_points(points, np.eye(4), intrinsics)

    np.testing.assert_allclose(uv, [[50.0, 40.0], [100.0, 15.0]])
    np.testing.assert_allclose(depth, [2.0, 2.0])


def test_transfer_features_preserves_direct_nodes_and_marks_imputation() -> None:
    observed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    structure = np.vstack((observed, [[1.1, 0.0, 0.0]]))
    features = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    counts = np.array([2, 0, 3])

    transferred, contributor_count, nearest = transfer_observed_features(
        observed,
        structure,
        features,
        counts,
    )

    np.testing.assert_allclose(transferred[0], [1.0, 0.0])
    np.testing.assert_allclose(transferred[2], [0.0, 1.0])
    np.testing.assert_allclose(transferred[1], [1.0, 0.0])
    np.testing.assert_allclose(transferred[3], [0.0, 1.0])
    np.testing.assert_array_equal(contributor_count, [2, 0, 3, 0])
    np.testing.assert_array_equal(nearest, [0, 0, 2, 2])
