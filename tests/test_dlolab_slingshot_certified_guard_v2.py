from __future__ import annotations

import numpy as np

from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    ARMS,
    BASELINE,
    MODES,
    decide,
    infer,
)
from bayesian_phystwin_experiments.dlolab_slingshot_certified_guard_v2 import (
    ARM_NAMES,
    BOOTSTRAP_REPLICATES,
    MEAN_CALIBRATION_OFFSET,
    PREFIX_BATCH_COUNT,
    SENSOR_DRAWS,
    WORLD_COUNT,
    _decisions_for_observations,
    continuous_worlds,
    future_task,
    pre_future_checks,
    prefix_task,
    protocol,
    score,
    validate_world,
)


def _bank(seed: int = 260830) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prefix = rng.normal(0, 0.02, (27, 3, 4, 3))
    reward = rng.normal(7, 0.4, (27, 7))
    return prefix, reward


def test_fresh_worlds_and_tasks_are_deterministic() -> None:
    worlds = continuous_worlds()
    assert len(worlds) == WORLD_COUNT
    assert (
        len(
            {
                (row["x_offset_m"], row["bending_E"], row["stretching_K"])
                for row in worlds
            }
        )
        == WORLD_COUNT
    )
    for world in worlds:
        validate_world(world)
    assert PREFIX_BATCH_COUNT == 36
    assert prefix_task(0)["world_indices"] == list(range(8))
    assert prefix_task(PREFIX_BATCH_COUNT - 1)["world_indices"] == list(range(280, 288))
    assert future_task(WORLD_COUNT - 1)["world_index"] == WORLD_COUNT - 1


def test_protocol_freezes_post_open_candidate_and_claim_boundary() -> None:
    frozen = protocol()
    assert frozen["parent_source_gate_passed"] is False
    assert frozen["parent_gate_reclassified"] is False
    assert frozen["parent_mean_guard_used_for_candidate_selection"] is True
    assert frozen["parent_evaluation_is_development_evidence_only"] is True
    assert frozen["method"].endswith("with_exact_fallback")
    assert frozen["sensor_draws_per_world"] == SENSOR_DRAWS
    assert frozen["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert frozen["retry_authorized"] is False
    assert frozen["replacement_authorized"] is False
    assert frozen["official_benchmark_or_sota_claim"] is False
    assert frozen["new_recordings"] is False


def test_vectorized_decisions_match_frozen_parent_implementation() -> None:
    prefix, reward = _bank()
    rng = np.random.default_rng(260831)
    observations = rng.normal(0, 0.02, (17, 3, 4, 3))
    calibration = {
        "mean": RegretCalibration(0.9, 19, 18, MEAN_CALIBRATION_OFFSET),
        "independent": RegretCalibration(0.9, 19, 18, 0.06471667098999023),
        "joint": RegretCalibration(0.9, 19, 18, 0.06905016708374023),
    }
    assert set(calibration) == set(MODES)
    expected = []
    for observation in observations:
        legacy = decide(infer(observation, prefix, reward), calibration)
        expected.append(
            [
                legacy[ARMS.index("incumbent")],
                legacy[ARMS.index("posterior_predictive_mean")],
                legacy[ARMS.index("mean_regret_guard")],
            ]
        )
    actual = _decisions_for_observations(observations, prefix, reward)
    assert np.array_equal(actual, np.asarray(expected, dtype=np.int64))
    assert np.all(actual[:, 0] == BASELINE)


def test_pre_future_gate_requires_real_nonfallback_support() -> None:
    decisions = np.full(
        (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES)), BASELINE, dtype=np.int64
    )
    failed = pre_future_checks(decisions, all_prefix_qa=True)
    assert failed["pre_future_gate_passed"] is False
    threshold = WORLD_COUNT * SENSOR_DRAWS // 100
    flat = decisions[:, :, 2].reshape(-1)
    flat[:threshold] = 4
    decisions[:32, :, 1] = 3
    decisions[:32, :, 2] = 4
    passed = pre_future_checks(decisions, all_prefix_qa=True)
    assert passed["pre_future_gate_passed"] is True


def test_exact_fallback_arm_is_byte_level_reward_identity() -> None:
    rng = np.random.default_rng(260832)
    rewards = rng.normal(7, 0.05, (WORLD_COUNT, 7))
    decisions = np.full(
        (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES)), BASELINE, dtype=np.int64
    )
    result = score(
        decisions,
        rewards,
        all_native_qa=True,
        pre_future_gate_passed=False,
    )
    incumbent = result["arms"]["incumbent"]
    guard = result["arms"]["mean_regret_guard"]
    assert incumbent["mean_gain_over_incumbent"] == 0
    assert guard["mean_gain_over_incumbent"] == 0
    assert guard["mean_native_reward"] == incumbent["mean_native_reward"]
    assert guard["nonfallback_sensor_decisions"] == 0
    assert guard["harmful_worlds_beyond_numeric_margin"] == 0
    assert result["source_gate_passed"] is False
