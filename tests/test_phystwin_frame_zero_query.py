import numpy as np
import pytest

from bayesian_phystwin.phystwin_frame_zero_query import (
    associate_frame_zero_queries,
)


def test_associate_frame_zero_queries_returns_deterministic_nearest_nodes() -> None:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    queries = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.0]])

    association = associate_frame_zero_queries(nodes, queries)

    np.testing.assert_array_equal(association.node_indices, [1, 2])
    np.testing.assert_allclose(
        association.distance_m,
        [np.sqrt(0.02), np.sqrt(0.05)],
    )


def test_associate_frame_zero_queries_has_no_trajectory_input() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    frame_zero = np.array([[0.75, 0.0, 0.0]])

    first = associate_frame_zero_queries(nodes, frame_zero)
    second = associate_frame_zero_queries(nodes, frame_zero.copy())

    np.testing.assert_array_equal(first.node_indices, second.node_indices)
    np.testing.assert_array_equal(first.distance_m, second.distance_m)


def test_associate_frame_zero_queries_accepts_empty_query_set() -> None:
    association = associate_frame_zero_queries(
        np.array([[0.0, 0.0, 0.0]]),
        np.empty((0, 3)),
    )

    assert association.node_indices.shape == (0,)
    assert association.distance_m.shape == (0,)


def test_associate_frame_zero_queries_rejects_nonfinite_geometry() -> None:
    with pytest.raises(ValueError, match="finite"):
        associate_frame_zero_queries(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[np.nan, 0.0, 0.0]]),
        )
