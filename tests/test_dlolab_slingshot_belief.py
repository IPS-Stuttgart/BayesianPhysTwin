from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    ARMS,
    BASELINE,
    COUNTS,
    MODES,
    ORDER,
    calibrate,
    commands_for_decisions,
    controls,
    decide,
    infer,
    particle_worlds,
    prefix_observations,
    prior_weights,
    protocol,
    sample_worlds,
    score,
    sensor_errors,
)
from bayesian_phystwin_experiments.dlolab_slingshot_value import action_bank, worlds


def _calibrations():
    return {mode: RegretCalibration(0.9, 19, 18, 0.0) for mode in MODES}


def _bank(placebo=False):
    prefix = np.zeros((27, 3, 4, 3))
    base = np.linspace(7, 8, 27)
    reward = np.repeat(base[:, None], 7, axis=1)
    if not placebo:
        reward[:, 1] += 0.02
        reward[:, [0, 2, 3, 4, 6]] -= 0.1
    return prefix, reward


def test_frozen_design_uses_fresh_continuous_worlds_and_source_selected_baseline():
    p = protocol()
    assert p["baseline_action"] == 5
    assert p["calibration_rank"] == 18
    assert len(particle_worlds()) == 27 and prior_weights().sum() == 1
    assert particle_worlds()[13]["bending_E"] == 1e5
    assert particle_worlds()[13]["stretching_K"] == 8e5
    old = {
        tuple(w[k] for k in ("x_offset_m", "bending_E", "stretching_K"))
        for w in worlds()
    }
    grid = {
        tuple(w[k] for k in ("x_offset_m", "bending_E", "stretching_K"))
        for w in particle_worlds()
    }
    assert old <= grid
    for role, count in COUNTS.items():
        selected = sample_worlds(role)
        assert selected == sample_worlds(role) and len(selected) == count
        assert not any(
            tuple(w[k] for k in ("x_offset_m", "bending_E", "stretching_K")) in grid
            for w in selected
        )
        assert sensor_errors(role).shape == (count, 3, 4, 3)
        np.testing.assert_array_equal(sensor_errors(role), sensor_errors(role))
    assert not p["evaluation_future_before_decision_seal"]
    assert not p["official_benchmark_or_sota_claim"]


def test_actions_keep_seven_source_controls_and_exact_baseline_fallback():
    source = np.arange(18, dtype=np.float64).reshape(1, 3, 6) / 200
    bank = controls(source)
    assert bank[:7].tobytes() == action_bank(source)[:7].tobytes()
    assert bank[7].tobytes() == bank[BASELINE].tobytes()
    chosen = np.full(len(ARMS), BASELINE, dtype=np.int64)
    commands = commands_for_decisions(bank, chosen)
    assert commands[0] is commands[-1]
    assert commands[0].tobytes() == bank[BASELINE : BASELINE + 1].tobytes()
    assert np.shares_memory(commands[0], bank)


def test_only_declared_causal_times_and_identities_are_observed():
    rod = np.arange(300 * 8 * 12 * 3, dtype=float).reshape(300, 8, 12, 3)
    sphere = np.arange(300 * 8 * 3, dtype=float).reshape(300, 8, 3)
    observed = prefix_observations({"rod_pos_m": rod, "sphere_pos_m": sphere})
    assert observed.shape == (8, 3, 4, 3)
    np.testing.assert_array_equal(observed[2, 1, :3], rod[219, 2, [3, 6, 8]])
    np.testing.assert_array_equal(observed[2, 1, 3], sphere[219, 2])
    with pytest.raises(ValueError, match="300-frame"):
        prefix_observations(
            {
                "rod_pos_m": np.zeros((900, 8, 12, 3)),
                "sphere_pos_m": np.zeros((900, 8, 3)),
            }
        )


def test_shared_physics_changes_only_coupling_and_not_action_marginals():
    prefix, reward = _bank()
    parts = infer(np.zeros((3, 4, 3)), prefix, reward)
    np.testing.assert_allclose(parts["weights"], prior_weights())
    slot = ORDER.index(1)
    assert parts["raw_upper"][2, slot] == pytest.approx(-0.018)
    assert parts["raw_upper"][1, slot] > 0.1
    decisions = decide(parts, _calibrations())
    assert decisions[ARMS.index("joint_regret_guard")] == 1
    assert decisions[ARMS.index("independent_regret_guard")] == BASELINE
    assert decisions[ARMS.index("mean_regret_guard")] == 1


def test_placebo_never_admits_a_spurious_improvement():
    prefix, reward = _bank(placebo=True)
    rng = np.random.default_rng(89)
    for _ in range(32):
        observation = rng.normal(0, 0.005, (3, 4, 3))
        decisions = decide(infer(observation, prefix, reward), _calibrations())
        assert np.all(decisions[-3:] == BASELINE)


def test_mean_controller_and_posterior_mean_have_the_same_linear_reward_objective():
    prefix, reward = _bank()
    parts = infer(np.zeros((3, 4, 3)), prefix, reward)
    expected = -prior_weights() @ reward[:, ORDER]
    np.testing.assert_allclose(parts["expected_losses"], expected)
    assert decide(parts, _calibrations())[4] == ORDER[int(np.argmin(expected))]


def test_calibration_is_rank18_max_over_all_alternatives_not_selected_only():
    prefix, reward = _bank()
    parts = [infer(np.zeros((3, 4, 3)), prefix, reward) for _ in range(19)]
    for p in parts:
        p["raw_upper"][:] = 0
    realized = np.full((19, 7), 7.0)
    realized[:, 1] -= np.arange(19) / 100
    values = calibrate(parts, realized)
    assert all(c.offset == pytest.approx(0.17) for c in values.values())
    with pytest.raises(ValueError, match="19"):
        calibrate(parts[:-1], realized[:-1])


def test_equal_mean_controls_do_not_count_as_a_novel_joint_gain():
    prefix, bank = _bank()
    part = infer(np.zeros((3, 4, 3)), prefix, bank)
    parts = [deepcopy(part) for _ in range(32)]
    decisions = np.stack([decide(p, _calibrations()) for p in parts])
    reward = np.tile(bank[13], (32, 1))
    result = score(decisions, parts, reward, _calibrations(), all_native_qa=True)
    assert not result["source_gate_passed"]
    assert not result["checks"]["positive_paired_ci_vs_mean_regret_guard"]
    assert (
        result["arms"]["joint_regret_guard"]["harmful_decisions_beyond_numeric_margin"]
        == 0
    )
    assert result["arms"]["joint_regret_guard"]["harm_probability_upper95"] < 0.1
    changed = decisions.copy()
    changed[0, -1] = 0
    with pytest.raises(ValueError, match="reproduce"):
        score(changed, parts, reward, _calibrations(), all_native_qa=True)
    with pytest.raises(ValueError, match="32-world"):
        score(
            decisions[:-1], parts[:-1], reward[:-1], _calibrations(), all_native_qa=True
        )


@pytest.mark.parametrize("kind", ["observation", "prefix", "reward"])
def test_invalid_or_missing_observations_are_not_filled(kind):
    prefix, reward = _bank()
    observation = np.zeros((3, 4, 3))
    {"observation": observation, "prefix": prefix, "reward": reward}[kind].flat[0] = (
        np.nan
    )
    with pytest.raises(ValueError):
        infer(observation, prefix, reward)


def test_wrong_calibration_and_sensor_partition_rejected():
    prefix, reward = _bank()
    with pytest.raises(ValueError):
        decide(
            infer(np.zeros((3, 4, 3)), prefix, reward),
            {"joint": RegretCalibration(0.9, 9, 9, 0.0)},
        )
    with pytest.raises(ValueError):
        sample_worlds("target")
    with pytest.raises(ValueError):
        sensor_errors("target")
