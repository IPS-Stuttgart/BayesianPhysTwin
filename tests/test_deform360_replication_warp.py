from __future__ import annotations

import numpy as np

from causal4d_public.deform360_phystwin_feasibility import (
    deform360_xyz_to_warp_xzy,
)
from causal4d_public.deform360_replication_graph import Deform360SparseGraph
from causal4d_public.deform360_replication_warp import (
    Deform360WarpForecastCase,
    sparse_graph_strain_summary,
    sparse_trajectory_chamfer_m,
    symmetric_chamfer_distance_m,
    warp_xzy_to_deform360_xyz,
)


def _graph() -> Deform360SparseGraph:
    positions = np.asarray(
        [[0.0, 0.1, 0.0], [0.1, 0.1, 0.0], [0.2, 0.1, 0.0], [0.3, 0.1, 0.0]]
    )
    return Deform360SparseGraph(
        positions_m=positions,
        spring_edges=np.asarray([[0, 1], [1, 2], [2, 3], [0, 2]], dtype=np.int32),
        spring_families=np.asarray([0, 0, 0, 1], dtype=np.int8),
        masses=np.ones(4),
        stratum="filament",
        diagnostics={},
    )


def test_warp_coordinate_transform_round_trip() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(7, 5, 3))
    warp = deform360_xyz_to_warp_xzy(
        values, initial_support_height_m=-0.17, clearance_m=0.003
    )
    restored = warp_xzy_to_deform360_xyz(
        warp, initial_support_height_m=-0.17, clearance_m=0.003
    )
    np.testing.assert_allclose(restored, values)


def test_chamfer_accepts_unequal_point_set_sizes() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    prediction = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert symmetric_chamfer_distance_m(reference, prediction) == 1.0 / 12.0


def test_sparse_trajectory_reports_late_third() -> None:
    prediction = np.zeros((6, 4, 3))
    references = []
    for frame in range(6):
        references.append(np.full((3, 3), float(frame)))
    result = sparse_trajectory_chamfer_m(references, prediction)

    expected = [np.sqrt(3.0) * frame for frame in range(6)]
    np.testing.assert_allclose(result["per_frame_m"], expected)
    np.testing.assert_allclose(result["mean_m"], np.mean(expected))
    np.testing.assert_allclose(result["late_mean_m"], np.mean(expected[4:]))


def test_case_defaults_to_zero_velocity_and_validates_contacts() -> None:
    case = Deform360WarpForecastCase(
        episode_id="unit",
        graph=_graph(),
        controller_positions_m=np.zeros((8, 1, 3)),
        contact_active=np.ones((8, 1), dtype=bool),
        contact_node_indices=(2,),
        contact_rest_lengths_m=np.asarray([0.001]),
        dt_seconds=1.0 / 30.0,
    )
    np.testing.assert_array_equal(case.initial_velocities_m_s, np.zeros((4, 3)))
    assert case.support_height_m is None
    expected_rest = np.linalg.norm(
        case.graph.positions_m[case.graph.spring_edges[:, 1]]
        - case.graph.positions_m[case.graph.spring_edges[:, 0]],
        axis=1,
    )
    np.testing.assert_array_equal(case.object_rest_lengths_m, expected_rest)


def test_case_accepts_object_persistent_rest_lengths() -> None:
    graph = _graph()
    rest = np.asarray([0.08, 0.08, 0.08, 0.10])
    case = Deform360WarpForecastCase(
        episode_id="unit",
        graph=graph,
        controller_positions_m=np.zeros((8, 1, 3)),
        contact_active=np.ones((8, 1), dtype=bool),
        contact_node_indices=(2,),
        contact_rest_lengths_m=np.asarray([0.001]),
        dt_seconds=1.0 / 30.0,
        object_rest_lengths_m=rest,
    )
    np.testing.assert_array_equal(case.object_rest_lengths_m, rest)
    current = np.repeat(graph.positions_m[None], 2, axis=0)
    summary = sparse_graph_strain_summary(
        graph, current, rest_lengths_m=case.object_rest_lengths_m
    )
    assert summary["maximum"] == 1.0
    stretch = sparse_graph_strain_summary(
        graph,
        current,
        rest_lengths_m=case.object_rest_lengths_m,
        spring_family=0,
    )
    assert stretch["maximum"] == 0.25


def test_case_accepts_registered_support_height() -> None:
    case = Deform360WarpForecastCase(
        episode_id="unit",
        graph=_graph(),
        controller_positions_m=np.zeros((8, 1, 3)),
        contact_active=np.ones((8, 1), dtype=bool),
        contact_node_indices=(2,),
        contact_rest_lengths_m=np.asarray([0.001]),
        dt_seconds=1.0 / 30.0,
        support_height_m=0.075,
    )

    assert case.support_height_m == 0.075


def test_strain_summary_is_zero_for_identity_trajectory() -> None:
    graph = _graph()
    trajectory = np.repeat(graph.positions_m[None], 4, axis=0)
    assert sparse_graph_strain_summary(graph, trajectory) == {
        "p95": 0.0,
        "p99": 0.0,
        "maximum": 0.0,
    }
