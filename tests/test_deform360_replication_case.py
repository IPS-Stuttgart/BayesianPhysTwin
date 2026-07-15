import numpy as np
import pytest

from causal4d_public.deform360_replication_case import (
    ReplicationWarpObservation,
    contact_propagated_initial_velocity,
    score_constant_persistence,
    score_replication_warp_prediction,
)
from causal4d_public.deform360_replication_graph import Deform360SparseGraph
from causal4d_public.deform360_replication_warp import Deform360WarpForecastCase


def _observation() -> ReplicationWarpObservation:
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    graph = Deform360SparseGraph(
        positions_m=points,
        spring_edges=np.array([[0, 1], [1, 2], [2, 3]]),
        spring_families=np.zeros(3, dtype=np.int8),
        masses=np.ones(4),
        stratum="filament",
        diagnostics={},
    )
    case = Deform360WarpForecastCase(
        episode_id="fixture",
        graph=graph,
        controller_positions_m=np.zeros((14, 1, 3)),
        contact_active=np.zeros((14, 1), dtype=bool),
        contact_node_indices=(0,),
        contact_rest_lengths_m=np.array([0.01]),
        dt_seconds=1 / 30,
    )
    return ReplicationWarpObservation(
        case=case,
        raw_hull_frame_indices=np.array([5, 7, 11, 18]),
        reference_hulls_m=(points, points + 0.01, points + 0.02, points + 0.03),
        prefix_endpoint_frame=5,
        contact_associations=(),
    )


def test_prediction_scoring_uses_raw_frame_offsets() -> None:
    observation = _observation()
    prediction = np.repeat(observation.case.graph.positions_m[None], 14, axis=0)
    prediction[2] += 0.01
    prediction[6] += 0.02
    prediction[13] += 0.03
    metrics = score_replication_warp_prediction(observation, prediction)
    assert metrics["mean_m"] < 1e-12


def test_persistence_scores_all_future_hulls() -> None:
    metrics = score_constant_persistence(_observation())
    assert len(metrics["per_frame_m"]) == 3
    assert metrics["mean_m"] > 0.0


def test_prefix_only_case_is_valid_but_not_scorable() -> None:
    full = _observation()
    prefix_only = ReplicationWarpObservation(
        case=full.case,
        raw_hull_frame_indices=np.array([5]),
        reference_hulls_m=(full.reference_hulls_m[0],),
        prefix_endpoint_frame=5,
        contact_associations=(),
    )
    with pytest.raises(ValueError, match="future hull references"):
        score_constant_persistence(prefix_only)
    with pytest.raises(ValueError, match="future hull references"):
        score_replication_warp_prediction(
            prefix_only,
            np.repeat(full.case.graph.positions_m[None], 14, axis=0),
        )


def test_contact_velocity_propagates_causally_and_decays_over_graph() -> None:
    graph = _observation().case.graph
    velocity = contact_propagated_initial_velocity(
        graph,
        (0,),
        np.asarray([[0.6, 0.0, 0.0]]),
        np.asarray([True]),
        length_scale_fraction=1.0,
    )
    assert velocity[0, 0] == pytest.approx(0.6)
    assert np.all(np.diff(velocity[:, 0]) < 0.0)
    np.testing.assert_array_equal(velocity[:, 1:], 0.0)

    inactive = contact_propagated_initial_velocity(
        graph,
        (0,),
        np.asarray([[0.6, 0.0, 0.0]]),
        np.asarray([False]),
    )
    np.testing.assert_array_equal(inactive, 0.0)
