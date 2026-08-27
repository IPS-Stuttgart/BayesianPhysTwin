"""Synthetic contracts only; no native simulator or study outcomes."""

import dataclasses

import numpy as np
import pytest

from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
from bayesian_phystwin_experiments.dlolab_regret_study import (
    ARMS,
    MODES,
    action_offsets,
    commands_for_action,
    infer_parts,
    loss_table,
    make_decisions,
    observe_prefix,
    particle_parameters,
    protocol,
    qualify_geometry,
    realized_losses,
    sample_worlds,
    score_decisions,
)


def _calibration():
    return {name: RegretCalibration(0.9, 39, 36, 0.0) for name in MODES}


def test_design_counts_and_only_simulation_claims():
    value = protocol()
    assert value["calibration_rank"] == 36
    assert value["evaluation_count"] == 64
    assert len(value["arms"]) == 7
    for name in (
        "physical_safety_claim",
        "official_benchmark_claim",
        "protected_data_read",
        "new_physical_recordings",
        "automatic_target_authorization",
    ):
        assert value[name] is False
    bending, velocity = particle_parameters()
    assert bending.shape == velocity.shape == (15,)
    assert (bending[7], velocity[7]) == (100000.0, 0.0)
    assert np.unique(action_offsets(), axis=0).shape == (9, 3)
    np.testing.assert_array_equal(action_offsets()[0], 0)


def test_truth_draws_are_deterministic_and_partitioned():
    a, b = sample_worlds("calibration"), sample_worlds("evaluation")
    assert a["goals"].shape == (39, 3)
    assert b["goals"].shape == (64, 3)
    for name, value in a.items():
        np.testing.assert_array_equal(value, sample_worlds("calibration")[name])
        assert not np.array_equal(value, b[name][:39])
    assert np.all((b["bending"] >= 50000) & (b["bending"] <= 200000))
    with pytest.raises(ValueError, match="partition"):
        sample_worlds("target")


def test_prefix_observation_budget_and_cubic_clamps():
    prefix = np.arange(2 * 25 * 16 * 3).reshape(2, 25, 16, 3).astype(float)
    error = np.ones((2, 3, 4, 3))
    observations = observe_prefix(prefix, error)
    np.testing.assert_array_equal(observations[:, -1, -1], prefix[:, 24, 15] + 1)
    with pytest.raises(ValueError, match="prefix"):
        observe_prefix(np.concatenate([prefix, prefix], axis=1), error)
    with pytest.raises(ValueError, match="nonfinite"):
        observe_prefix(prefix, error * np.nan)
    clamps = prefix[:, 0, :2]
    commands = commands_for_action(clamps, action_offsets()[1])
    np.testing.assert_array_equal(commands[0], clamps)
    np.testing.assert_allclose(commands[-1], clamps + action_offsets()[1])
    hold = commands_for_action(clamps, action_offsets()[0])
    np.testing.assert_array_equal(hold, np.broadcast_to(clamps, hold.shape))


def test_loss_table_matches_per_world_outcomes():
    rng = np.random.default_rng(10)
    future = rng.normal(size=(2, 9, 40, 16, 3))
    goals = rng.normal(size=(2, 3))
    table = loss_table(future, goals)
    truth = realized_losses(future, goals)
    np.testing.assert_array_equal(truth, table[np.arange(2), np.arange(2)])
    with pytest.raises(ValueError, match="nonfinite"):
        realized_losses(future * np.nan, goals)


def test_inference_accepts_no_evaluation_future_argument():
    prefix = np.zeros((15, 25, 16, 3))
    future = np.zeros((15, 9, 40, 16, 3))
    parts = infer_parts(np.zeros((2, 3, 4, 3)), np.zeros((2, 3)), prefix, future)
    np.testing.assert_allclose(parts["weights"], 1 / 15)
    np.testing.assert_allclose(parts["iid_weights"], 1 / 15)
    np.testing.assert_array_equal(make_decisions(parts, _calibration()), 0)
    with pytest.raises(ValueError, match="observation"):
        infer_parts(np.zeros((2, 4, 4, 3)), np.zeros((2, 3)), prefix, future)
    with pytest.raises(ValueError, match="particle bank"):
        infer_parts(np.zeros((2, 3, 4, 3)), np.zeros((2, 3)), prefix, future[:14])


def test_score_fails_gate_for_placebo_even_with_perfect_coverage():
    decisions = np.zeros((64, len(ARMS)), dtype=np.int64)
    losses = np.ones((64, 9))
    result = score_decisions(decisions, losses, np.zeros((64, 3, 9)), _calibration())
    assert not result["source_gate_passed"]
    assert result["ordinary_evaluation_episodes"] == 64
    assert result["simultaneous_action_bound_coverage"]["joint"] == 1
    assert result["arms"]["joint_regret_guard"]["harmful_decisions"] == 0
    assert result["arms"]["joint_regret_guard"]["harm_probability_upper_95"] < 0.05
    for name in ("mean_regret_guard", "independent_regret_guard"):
        assert result["paired_gain_ci95_m2"][name] == [0.0, 0.0]


def test_score_positive_control_must_beat_both_calibrated_controls():
    decisions = np.zeros((64, len(ARMS)), dtype=np.int64)
    decisions[:, 6] = np.arange(64) % 3 + 1
    losses = np.ones((64, 9))
    losses[np.arange(64), decisions[:, 6]] = 0.5
    result = score_decisions(decisions, losses, np.zeros((64, 3, 9)), _calibration())
    assert result["source_gate_passed"]
    decisions[:, 4] = decisions[:, 6]
    result = score_decisions(decisions, losses, np.zeros((64, 3, 9)), _calibration())
    assert not result["source_gate_passed"]
    assert not result["checks"]["paired_gain_lower_ci_positive_vs_mean_regret_guard"]


@pytest.mark.parametrize(
    "change", ["missing", "nan", "wrong_hold", "wrong_calibration"]
)
def test_no_missing_denominator_or_relabelled_calibration(change):
    decisions = np.zeros((64, len(ARMS)), dtype=np.int64)
    losses = np.ones((64, 9))
    calibrations = _calibration()
    if change == "missing":
        decisions = decisions[:63]
    elif change == "nan":
        losses[0, 0] = np.nan
    elif change == "wrong_hold":
        decisions[0, 0] = 1
    else:
        calibrations["joint"] = dataclasses.replace(
            calibrations["joint"], count=29, rank=27
        )
    with pytest.raises(ValueError):
        score_decisions(decisions, losses, np.zeros((64, 3, 9)), calibrations)


def test_physical_gate_rejects_invalid_geometry():
    positions = np.zeros((2, 3, 16, 3))
    positions[..., 0] = np.arange(16) * 0.025
    qualify_geometry(positions)
    positions[:, :, -1, 0] += 0.01
    with pytest.raises(ValueError, match="length"):
        qualify_geometry(positions)
