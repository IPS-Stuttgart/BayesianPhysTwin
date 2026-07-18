import numpy as np
import pytest

from bayesian_phystwin.matphys_graph_parts import graph_semantic_parts


def _chain(length: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack(
        (np.arange(length, dtype=float), np.zeros(length), np.zeros(length))
    )
    edges = np.column_stack((np.arange(length - 1), np.arange(1, length)))
    return points, edges


def test_graph_semantic_parts_are_deterministic_connected_and_nonempty() -> None:
    points, edges = _chain(12)
    features = np.zeros((12, 4), dtype=float)
    features[:4, 0] = 1.0
    features[4:8, 1] = 1.0
    features[8:, 2] = 1.0

    first = graph_semantic_parts(points, edges, features, part_count=3)
    second = graph_semantic_parts(points, edges, features, part_count=3)

    np.testing.assert_array_equal(first.assignments, second.assignments)
    np.testing.assert_array_equal(first.seeds, second.seeds)
    np.testing.assert_array_equal(first.part_counts, second.part_counts)
    assert np.all(first.part_counts > 0)
    for part in range(3):
        members = np.flatnonzero(first.assignments == part)
        assert np.all(np.diff(members) == 1)
    assert 0.0 < first.boundary_edge_fraction < 1.0


def test_semantic_boundary_changes_graph_geodesic_partition() -> None:
    points, edges = _chain(10)
    uniform = np.ones((10, 2), dtype=float)
    semantic = np.zeros((10, 2), dtype=float)
    semantic[:7, 0] = 1.0
    semantic[7:, 1] = 1.0

    geometric = graph_semantic_parts(
        points,
        edges,
        uniform,
        part_count=2,
        semantic_edge_weight=4.0,
    )
    informed = graph_semantic_parts(
        points,
        edges,
        semantic,
        part_count=2,
        semantic_edge_weight=20.0,
    )

    assert not np.array_equal(geometric.assignments, informed.assignments)
    assert informed.assignments[6] != informed.assignments[7]


def test_graph_semantic_parts_reject_zero_features() -> None:
    points, edges = _chain(4)

    with pytest.raises(ValueError, match="nonzero semantic feature"):
        graph_semantic_parts(points, edges, np.zeros((4, 2)), part_count=2)
