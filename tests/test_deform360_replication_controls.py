from __future__ import annotations

import numpy as np

from causal4d_public.deform360_replication_controls import (
    ContactTransitionEpisode,
    build_pooling_control_selection_artifact,
    evaluate_pooling_control,
    fit_causal_contact_transition,
    predict_causal_contact_transition,
    select_pooling_controls,
    validate_pooling_control_selection_artifact,
)


def test_pooling_control_preserves_every_single_source_selection() -> None:
    scores = np.asarray(
        [
            [0.4, 0.7, 0.7],
            [0.5, 0.3, 0.6],
            [0.6, 0.6, 0.2],
            [0.45, 0.45, 0.45],
        ]
    )
    selection = select_pooling_controls(scores)
    assert selection.pooled_candidate_index == 3
    assert selection.single_source_candidate_indices == (0, 1, 2)
    assert selection.unique_single_source_candidate_indices == (0, 1, 2)

    result = evaluate_pooling_control(selection, np.asarray([0.6, 0.4, 0.5, 0.3]))
    assert result["single_source_median_target_chamfer_m"] == 0.5
    assert result["pooled_relative_improvement_over_single_source_median"] == 0.4
    assert result["pooled_better_than_single_source_median"] is True


def test_single_source_selection_does_not_require_other_source_validity() -> None:
    scores = np.asarray(
        [
            [0.1, np.inf],
            [np.inf, 0.2],
            [0.3, 0.3],
        ]
    )
    selection = select_pooling_controls(scores)
    assert selection.pooled_candidate_index == 2
    assert selection.single_source_candidate_indices == (0, 1)


def test_pooling_control_artifact_seals_source_selected_candidates_only() -> None:
    fit = {
        "artifact_kind": "unit-source-fit",
        "result_sha256": "a" * 64,
        "candidate_scores": [
            {
                "candidate_index": candidate,
                "parameters": {"value": candidate},
                "per_episode": [
                    {
                        "episode_id": f"source-{source}",
                        "chamfer_distance_m": value,
                    }
                    for source, value in enumerate(row)
                ],
            }
            for candidate, row in enumerate(([0.2, 0.5], [0.4, 0.15], [0.25, 0.25]))
        ],
    }
    artifact = build_pooling_control_selection_artifact(fit)
    result = validate_pooling_control_selection_artifact(artifact)
    assert result["passed"] is True
    assert artifact["selection"]["pooled_candidate_index"] == 2
    assert artifact["selection"]["single_source_candidate_indices"] == (0, 1)
    assert artifact["sealed_candidate_indices"] == [0, 1, 2]
    assert artifact["information_boundary"]["target_future_geometry_read"] is False


def _transition_episode(
    episode: int, *, onset: int, release: int
) -> ContactTransitionEpisode:
    frame_count = 42
    time = np.arange(frame_count, dtype=np.float64)
    proximity = 0.12 - 0.006 * np.minimum(time, onset)
    proximity[onset:release] = 0.012
    proximity[release:] = 0.012 + 0.008 * (time[release:] - release)
    openings = np.full((frame_count, 1), 0.08)
    openings[onset:release] = 0.015
    controllers = np.zeros((frame_count, 1, 3))
    controllers[:, 0, 0] = proximity
    objects = np.zeros((frame_count, 5, 3))
    active = np.zeros((frame_count, 1), dtype=bool)
    active[onset:release] = True
    return ContactTransitionEpisode(
        episode_id=f"episode-{episode}",
        openings_m=openings,
        controller_positions_m=controllers,
        predicted_object_positions_m=objects,
        contact_active=active,
        dt_seconds=0.1,
    )


def test_contact_transition_hazard_learns_onset_and_release_without_future_labels() -> (
    None
):
    source = [
        _transition_episode(0, onset=13, release=29),
        _transition_episode(1, onset=14, release=30),
        _transition_episode(2, onset=12, release=28),
    ]
    calibration = [_transition_episode(3, onset=13, release=30)]
    fitted = fit_causal_contact_transition(source, calibration)
    assert fitted.calibration_metrics["brier_score"] < 0.1
    assert fitted.calibration_metrics["balanced_accuracy"] > 0.9

    target = _transition_episode(4, onset=14, release=29)
    probabilities, states = predict_causal_contact_transition(
        fitted.model,
        target.openings_m,
        target.controller_positions_m,
        target.predicted_object_positions_m,
        dt_seconds=target.dt_seconds,
        initial_contact_state=target.contact_active[0],
    )
    assert probabilities.shape == states.shape == target.contact_active.shape
    assert np.mean(states == target.contact_active) > 0.9
    assert states[0, 0] == target.contact_active[0, 0]
