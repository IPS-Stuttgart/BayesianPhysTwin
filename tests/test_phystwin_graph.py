import numpy as np
import pytest

from bayesian_phystwin.phystwin_graph import (
    PhysTwinPiecewiseSpringGraphConfig,
    PhysTwinSpringGraphConfig,
    build_piecewise_phystwin_spring_graph,
    build_phystwin_spring_graph,
    part_pair_spring_grouping,
    spatial_spring_region_ids,
    transfer_teacher_spring_field,
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


def test_one_region_piecewise_graph_is_byte_compatible_with_standard_graph():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ]
    )
    controls = np.array([[0.3, 0.0, 0.0]])
    standard = build_phystwin_spring_graph(points, controls, config=_config())
    piecewise = build_piecewise_phystwin_spring_graph(
        points,
        controls,
        np.zeros(len(points), dtype=np.int32),
        config=PhysTwinPiecewiseSpringGraphConfig(
            object_radii=(1.01,),
            object_max_neighbours=(3,),
            controller_radius=0.6,
            controller_max_neighbours=2,
        ),
    )

    np.testing.assert_array_equal(piecewise.vertices, standard.vertices)
    np.testing.assert_array_equal(piecewise.springs, standard.springs)
    np.testing.assert_array_equal(piecewise.rest_lengths, standard.rest_lengths)
    np.testing.assert_array_equal(piecewise.masses, standard.masses)
    assert piecewise.num_object_springs == standard.num_object_springs
    assert piecewise.num_object_points == standard.num_object_points == 4


def test_piecewise_regions_change_topology_and_transfer_teacher_field():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ]
    )
    teacher = build_phystwin_spring_graph(
        points,
        None,
        config=_config(object_radius=1.0, object_max_neighbours=2),
    )
    candidate = build_piecewise_phystwin_spring_graph(
        points,
        None,
        np.array([0, 0, 1, 1], dtype=np.int32),
        config=PhysTwinPiecewiseSpringGraphConfig(
            object_radii=(1.0, 1.0),
            object_max_neighbours=(2, 4),
            controller_radius=0.6,
            controller_max_neighbours=2,
        ),
    )

    assert teacher.num_object_springs == 3
    assert candidate.num_object_springs == 6
    transferred = transfer_teacher_spring_field(
        teacher,
        candidate,
        np.array([10.0, 20.0, 30.0]),
    )
    assert transferred.exact_edge_count == 3
    assert transferred.interpolated_edge_count == 3
    assert transferred.removed_teacher_edge_count == 0
    assert transferred.spring_y.dtype == np.float32
    assert np.all(transferred.spring_y > 0.0)
    teacher_by_edge = {
        tuple(sorted(edge)): value
        for edge, value in zip(teacher.springs.tolist(), [10.0, 20.0, 30.0])
    }
    for edge, value in zip(candidate.springs.tolist(), transferred.spring_y):
        key = tuple(sorted(edge))
        if key in teacher_by_edge:
            assert value == pytest.approx(teacher_by_edge[key])


def test_piecewise_graph_rejects_missing_or_unknown_region_assignments():
    config = PhysTwinPiecewiseSpringGraphConfig(
        object_radii=(1.0, 2.0),
        object_max_neighbours=(2, 3),
        controller_radius=0.5,
        controller_max_neighbours=1,
    )
    points = np.zeros((3, 3))
    with pytest.raises(ValueError, match="label every"):
        build_piecewise_phystwin_spring_graph(
            points,
            None,
            np.array([0, 1]),
            config=config,
        )
    with pytest.raises(ValueError, match="exceeds"):
        build_piecewise_phystwin_spring_graph(
            points,
            None,
            np.array([0, 1, 2]),
            config=config,
        )


def test_spatial_regions_are_deterministic_balanced_and_reserve_controller_group():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    springs = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
        dtype=np.int32,
    )

    first = spatial_spring_region_ids(
        vertices,
        springs,
        num_object_springs=4,
        region_count=2,
    )
    second = spatial_spring_region_ids(
        vertices,
        springs,
        num_object_springs=4,
        region_count=2,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.bincount(first[:4]), [2, 2])
    assert first[-1] == 2


def test_part_pair_groups_preserve_cross_part_and_controller_structure():
    springs = np.array(
        [[0, 1], [1, 2], [2, 3], [0, 3], [4, 1], [4, 2]],
        dtype=np.int32,
    )
    assignments = np.array([2, 2, 5, 5], dtype=np.int32)

    grouping = part_pair_spring_grouping(
        springs,
        assignments,
        num_object_springs=4,
    )

    np.testing.assert_array_equal(
        grouping.object_part_pairs,
        [[2, 2], [2, 5], [5, 5]],
    )
    np.testing.assert_array_equal(grouping.group_ids, [0, 1, 2, 1, 3, 3])
    np.testing.assert_array_equal(grouping.group_counts, [1, 2, 1, 2])
    assert grouping.controller_group == 3


def test_part_pair_groups_reject_missing_object_assignments():
    with pytest.raises(ValueError, match="endpoint exceeds"):
        part_pair_spring_grouping(
            np.array([[0, 2]], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            num_object_springs=1,
        )
