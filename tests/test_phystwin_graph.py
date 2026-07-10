import numpy as np
import pytest

from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)


def _config(**overrides):
    values = {
        "object_radius": 1.01,
        "object_max_neighbours": 3,
        "controller_radius": 0.6,
        "controller_max_neighbours": 2,
    }
    values.update(overrides)
    return PhysTwinSpringGraphConfig(**values)


def test_graph_matches_object_then_controller_order():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    controls = np.array([[0.5, 0.0, 0.0]])

    graph = build_phystwin_spring_graph(points, controls, config=_config())

    assert graph.vertices.dtype == np.float32
    assert graph.springs.dtype == np.int32
    assert graph.rest_lengths.dtype == np.float32
    assert graph.num_object_springs == 3
    np.testing.assert_array_equal(
        graph.springs,
        np.array([[0, 1], [1, 2], [2, 3], [4, 0], [4, 1]], dtype=np.int32),
    )
    np.testing.assert_allclose(graph.rest_lengths, [1.0, 1.0, 1.0, 0.5, 0.5])
    np.testing.assert_array_equal(graph.masses, np.ones(5, dtype=np.float32))


def test_graph_deduplicates_undirected_edges_and_obeys_neighbor_limit():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ]
    )

    graph = build_phystwin_spring_graph(
        points,
        None,
        config=_config(object_radius=2.0, object_max_neighbours=2),
    )

    assert graph.num_object_springs == 3
    assert {tuple(sorted(edge)) for edge in graph.springs.tolist()} == {
        (0, 1),
        (1, 2),
        (2, 3),
    }


def test_graph_skips_coincident_short_springs():
    points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )

    graph = build_phystwin_spring_graph(
        points,
        None,
        config=_config(object_radius=1.0, object_max_neighbours=3),
    )

    assert graph.num_object_springs == 2
    assert np.all(graph.rest_lengths > 1e-4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("object_radius", 0.0),
        ("controller_radius", -1.0),
        ("object_max_neighbours", 0),
        ("controller_max_neighbours", 0),
    ],
)
def test_graph_rejects_invalid_configuration(field, value):
    with pytest.raises(ValueError):
        build_phystwin_spring_graph(
            np.zeros((2, 3)),
            None,
            config=_config(**{field: value}),
        )
