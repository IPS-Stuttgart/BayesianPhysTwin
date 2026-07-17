from __future__ import annotations

import copy

import numpy as np
import pytest

from causal4d_public.deform360_reusable_sota_selection import (
    evaluate_frozen_pooling_controls,
    fit_pooling_controls,
    normalized_physics_score,
)


def test_normalized_score_balances_track_and_chamfer_units() -> None:
    track = np.array([[0.02, 0.04], [0.04, 0.02]])
    chamfer = np.array([[0.06, 0.03], [0.03, 0.06]])
    score = normalized_physics_score(
        track,
        chamfer,
        np.array([0.04, 0.04]),
        np.array([0.06, 0.06]),
    )
    np.testing.assert_allclose(score, [[0.75, 0.75], [0.75, 0.75]])


def test_pooling_controls_freeze_without_held_outcomes() -> None:
    labels = ["specialist-a", "robust", "specialist-c"]
    fit_ids = [1, 3, 4, 6, 7, 9]
    # Specialists win individual actions; the robust candidate wins in aggregate.
    score = np.array(
        [
            [0.3, 1.4, 1.4, 1.4, 1.4, 1.4],
            [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
            [1.4, 1.4, 0.3, 1.4, 1.4, 1.4],
        ]
    )
    persistence = np.ones(len(fit_ids))
    selection = fit_pooling_controls(
        labels,
        fit_ids,
        track_error_m=score,
        chamfer_m=score,
        persistence_track_error_m=persistence,
        persistence_chamfer_m=persistence,
    )
    assert selection["pooled_candidate_label"] == "robust"
    assert selection["held_episode_outcomes_used"] is False
    assert selection["leave_one_out_persistence_win_fraction"] == 1.0

    held_score = np.array(
        [
            [1.2, 1.1],
            [0.6, 0.8],
            [1.3, 1.2],
        ]
    )
    result = evaluate_frozen_pooling_controls(
        selection,
        [0, 2],
        track_error_m=held_score,
        chamfer_m=held_score,
        persistence_track_error_m=np.ones(2),
        persistence_chamfer_m=np.ones(2),
    )
    assert result["pooled_candidate_label"] == "robust"
    assert result["persistence_win_fraction"] == 1.0
    assert result["selection_refit_on_held_outcomes"] is False


def test_held_evaluator_rejects_a_tainted_selection() -> None:
    selection = fit_pooling_controls(
        ["a", "b"],
        [1, 3, 4],
        track_error_m=np.array([[0.5, 0.6, 0.7], [0.8, 0.7, 0.6]]),
        chamfer_m=np.array([[0.5, 0.6, 0.7], [0.8, 0.7, 0.6]]),
        persistence_track_error_m=np.ones(3),
        persistence_chamfer_m=np.ones(3),
    )
    tainted = copy.deepcopy(selection)
    tainted["held_episode_outcomes_used"] = True
    with pytest.raises(ValueError, match="held outcomes"):
        evaluate_frozen_pooling_controls(
            tainted,
            [0],
            track_error_m=np.array([[0.5], [0.6]]),
            chamfer_m=np.array([[0.5], [0.6]]),
            persistence_track_error_m=np.ones(1),
            persistence_chamfer_m=np.ones(1),
        )
