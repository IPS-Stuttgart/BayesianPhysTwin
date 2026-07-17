from __future__ import annotations

import numpy as np

from causal4d_public.deform360_action_support import (
    GraphActionSupportEpisode,
    dynamic_contact_anchor_indices,
    fit_source_graph_action_support,
    graph_action_support_prediction,
    graph_readout_action_support,
)
from causal4d_public.deform360_phystwin_trust import CausalTrustEpisode


def _episode(episode_id: str) -> GraphActionSupportEpisode:
    frames = 6
    nodes = 4
    initial = np.column_stack(
        (np.arange(nodes, dtype=float), np.zeros(nodes), np.zeros(nodes))
    )
    progress = np.linspace(0.0, 1.0, frames)[:, None]
    response = np.zeros((frames, nodes, 3))
    response[..., 1] = 0.12 * progress
    target = np.repeat(initial[None], frames, axis=0)
    target[:, 0, 1] = 0.10 * progress[:, 0]
    episode = CausalTrustEpisode(
        episode_id=episode_id,
        target_m=target,
        visibility=np.ones((frames, nodes), dtype=bool),
        validity=np.ones((frames, nodes), dtype=bool),
        driven_m=initial[None] + response,
        zero_action_m=np.repeat(initial[None], frames, axis=0),
        train_stop_frame=4,
        source_data_sha256="a" * 64,
        driven_trajectory_sha256="b" * 64,
        zero_action_trajectory_sha256="c" * 64,
    )
    return GraphActionSupportEpisode(
        episode=episode,
        readout_weights=np.eye(nodes),
        node_contact_distance_m=np.asarray([0.0, 0.1, 0.2, 0.3]),
    )


def test_graph_readout_support_is_convex_and_distance_ordered() -> None:
    support = graph_readout_action_support(
        np.eye(3),
        np.asarray([0.0, 0.1, 0.2]),
        length_scale_m=0.1,
    )
    np.testing.assert_allclose(support, np.exp(-np.asarray([0.0, 1.0, 2.0])))


def test_dynamic_contact_anchors_match_groupwise_nearest_nodes() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    controls = np.asarray(
        [
            [0.005, 0.0, 0.0],
            [0.006, 0.0, 0.0],
            [0.195, 0.0, 0.0],
            [0.196, 0.0, 0.0],
        ]
    )

    anchors = dynamic_contact_anchor_indices(
        points,
        controls,
        controller_group_size=2,
        maximum_contact_distance_m=0.01,
    )

    np.testing.assert_array_equal(anchors, [0, 2])


def test_zero_action_support_is_exact_persistence() -> None:
    source = _episode("first")
    prediction = graph_action_support_prediction(
        source,
        action_response=0.0,
        length_scale_m=0.05,
    )
    np.testing.assert_array_equal(
        prediction,
        np.repeat(source.episode.target_m[:1], len(prediction), axis=0),
    )


def test_source_fit_prefers_local_action_support_and_preserves_tail_gate() -> None:
    result = fit_source_graph_action_support(
        (_episode("first"), _episode("second")),
        length_scale_grid_m=(0.05, 1e6),
        action_response_grid=(0.0, 0.5, 1.0),
        transfer_episodes=(_episode("third"),),
    )
    assert result["selected"]["length_scale_m"] == 0.05
    assert result["selected"]["action_response"] > 0.0
    assert result["selected"]["pooled_train_relative_score_vs_persistence"] < 1.0
    assert result["selected"]["pooled_tail_relative_score_vs_persistence"] < 1.0
    assert result["information_boundary"]["source_tails_used_for_selection"] is False
    assert (
        result["information_boundary"]["held_out_source_transfer_used_for_selection"]
        is False
    )
    assert (
        result["held_out_source_transfer"]["third"]["untouched_tail"][
            "relative_score_vs_persistence"
        ]
        < 1.0
    )
