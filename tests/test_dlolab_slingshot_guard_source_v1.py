from __future__ import annotations

import json

import numpy as np

from bayesian_phystwin_experiments.dlolab_slingshot_belief import sample_worlds
from bayesian_phystwin_experiments.dlolab_slingshot_guard_source_v1 import (
    SENSOR_DRAWS,
    WORLD_COUNT,
    candidates,
    cross_fitted_decisions,
    infer_candidates,
    pre_outcome_checks,
    protocol,
    score,
    select_candidate,
)


def _blank_decisions() -> np.ndarray:
    return np.zeros(
        (WORLD_COUNT, SENSOR_DRAWS, len(candidates())), dtype=np.int64
    )


def test_candidate_bank_and_protocol_are_frozen_and_source_only() -> None:
    bank = candidates()
    value = protocol(sample_worlds("calibration"))
    assert len(bank) == 13
    assert bank[0]["name"] == "exact_blind_fallback"
    assert [row["index"] for row in bank] == list(range(len(bank)))
    assert value["parent_calibration_reward_before_decision_barrier"] is False
    assert value["fresh_world_automatically_authorized"] is False
    assert value["retry_authorized"] is False
    assert value["official_benchmark_or_sota_claim"] is False
    json.dumps(value, sort_keys=True, allow_nan=False)


def test_inference_preserves_exact_fallback_and_posterior_spread() -> None:
    history = np.zeros((27, 3, 4, 3), dtype=np.float64)
    history[:, 0, 0, 0] = np.arange(27) * 0.02
    reward = np.zeros((27, 7), dtype=np.float64)
    reward[:, 0] = 1.0
    reward[np.arange(27), 1 + np.arange(27) % 6] = 1.2
    truth = np.zeros((WORLD_COUNT, 3, 4, 3), dtype=np.float64)
    truth[:, 0, 0, 0] = np.arange(WORLD_COUNT) % 27 * 0.02
    result = infer_candidates(history, reward, truth)
    assert result["posterior_weights"].shape == (
        WORLD_COUNT,
        SENSOR_DRAWS,
        27,
    )
    np.testing.assert_allclose(result["posterior_weights"].sum(axis=-1), 1.0)
    blind = int(result["blind_action"])
    assert np.all(result["candidate_decisions"][:, :, 0] == blind)
    assert np.all(result["posterior_gain_std"] >= 0)
    assert np.all(result["posterior_positive_gain_probability"] >= 0)
    assert np.all(result["posterior_positive_gain_probability"] <= 1)


def test_selection_uses_only_fit_worlds_and_falls_back_exactly() -> None:
    decisions = _blank_decisions()
    decisions[:, :, 1] = 1
    rewards = np.zeros((WORLD_COUNT, 7), dtype=np.float64)
    rewards[:, 0] = 1.0
    rewards[:, 1] = 1.01
    fit = np.arange(WORLD_COUNT - 1)
    positive = select_candidate(fit, decisions, rewards)
    assert positive["selected_candidate_index"] == 1
    assert positive["exact_fallback_selected"] is False

    rewards[:, 1] = 0.99
    negative = select_candidate(fit, decisions, rewards)
    assert negative["selected_candidate_index"] == 0
    assert negative["exact_fallback_selected"] is True
    selected, folds, full = cross_fitted_decisions(decisions, rewards)
    assert np.array_equal(selected, np.zeros_like(selected))
    assert all(row["exact_fallback_selected"] for row in folds)
    assert full["exact_fallback_selected"] is True


def test_cross_fitted_positive_guard_passes_locked_source_checks() -> None:
    decisions = _blank_decisions()
    decisions[:, :, 1] = 1
    candidate_data = {
        "candidate_decisions": decisions,
        "blind_action": np.asarray(0, dtype=np.int64),
        "active_bayes_action": np.full(
            (WORLD_COUNT, SENSOR_DRAWS), 2, dtype=np.int64
        ),
        "active_map_action": np.full(
            (WORLD_COUNT, SENSOR_DRAWS), 3, dtype=np.int64
        ),
    }
    rewards = np.zeros((WORLD_COUNT, 7), dtype=np.float64)
    rewards[:, 0] = 1.0
    rewards[:, 1] = 1.01
    rewards[:, 2] = 0.99
    rewards[:, 3] = 0.98
    result = score(candidate_data, rewards, all_native_qa=True)
    assert result["source_gate_passed"] is True
    assert result["full_fit_selection"]["selected_candidate_index"] == 1
    assert result["arms"]["cross_fitted_guard"]["mean_gain_over_blind"] > 0
    assert result["cross_fitted_gain_over_active_bayes"]["mean_gain"] > 0


def test_pre_outcome_gate_never_uses_source_rewards() -> None:
    decisions = _blank_decisions()
    decisions[:, :, 1] = 1
    decisions[:8, :, 2] = 2
    data = {
        "candidate_decisions": decisions,
        "active_bayes_action": np.ones(
            (WORLD_COUNT, SENSOR_DRAWS), dtype=np.int64
        ),
        "blind_action": np.asarray(0, dtype=np.int64),
    }
    result = pre_outcome_checks(data, all_prefix_qa=True)
    assert result["pre_outcome_gate_passed"] is True
    assert "reward" not in result
