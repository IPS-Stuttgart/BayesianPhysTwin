"""Fresh-world active Bayesian identification for native Slingshot control."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_slingshot_belief import (
    BIAS_STD_M,
    NOISE_STD_M,
    prior_weights,
)
from .dlolab_slingshot_task_probe_dev import _posterior_weights

Array: TypeAlias = NDArray[Any]

WORLD_COUNT = 32
SENSOR_DRAWS = 8
WORLD_SEED = 261301
SENSOR_SEED = 261302
BOOTSTRAP_SEED = 261303
BOOTSTRAP_REPLICATES = 20000
ACTIVE_FRACTION = 0.70
REWARD_MARGIN = 0.002
ARM_NAMES = (
    "blind_prior",
    "passive_map",
    "passive_bayes",
    "active_map",
    "active_bayes",
)


def continuous_worlds() -> list[dict[str, Any]]:
    rng = np.random.default_rng(WORLD_SEED)
    x = rng.uniform(-0.02, 0.02, WORLD_COUNT)
    bending = 1e5 * np.exp(
        rng.uniform(np.log(0.5), np.log(2.0), WORLD_COUNT)
    )
    stretching = 8e5 * np.exp(
        rng.uniform(np.log(0.5), np.log(2.0), WORLD_COUNT)
    )
    return [
        {
            "index": index,
            "x_offset_m": float(x[index]),
            "bending_E": float(bending[index]),
            "stretching_K": float(stretching[index]),
        }
        for index in range(WORLD_COUNT)
    ]


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-active-bayes-source-v1",
        "role": "fresh_continuous_world_test_of_posthoc_generated_bayes_vs_map_hypothesis",
        "parent_particle_gate_passed": False,
        "parent_particle_gate_reclassified": False,
        "parent_output_retried": False,
        "worlds": continuous_worlds(),
        "world_seed": WORLD_SEED,
        "world_count": WORLD_COUNT,
        "world_distribution": {
            "x_offset_m": "uniform[-0.02,0.02]",
            "bending_E": "log_uniform[50000,200000]",
            "stretching_K": "log_uniform[400000,1600000]",
        },
        "worlds_disjoint_from_parent_continuous_partitions": True,
        "probe_names": ["passive_original", "active_frontload_70"],
        "active_frontload_fraction": ACTIVE_FRACTION,
        "probe_role": "separate_matched_reset_identification_episode",
        "probe_cost_in_task_reward": False,
        "task_starts_from_fresh_native_reset": True,
        "particle_count": 27,
        "task_action_count": 7,
        "task_action_duplicate_slot": 7,
        "arms": list(ARM_NAMES),
        "primary_arm": "active_bayes",
        "primary_hypothesis": "posterior_integration_beats_active_plug_in_map_on_fresh_continuous_worlds",
        "sensor_draws_per_world": SENSOR_DRAWS,
        "sensor_seed": SENSOR_SEED,
        "shared_bias_std_m": BIAS_STD_M,
        "independent_noise_std_m": NOISE_STD_M,
        "statistical_unit": "continuous_world_after_averaging_registered_sensor_draws",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pre_future_gate": {
            "all_prefix_batches_native_qualified": True,
            "active_bayes_nonblind_sensor_decisions_at_least": 32,
            "active_bayes_differs_from_active_map_at_least": 16,
            "distinct_active_bayes_actions_at_least": 2,
        },
        "source_gate": {
            "all_32_worlds_and_native_qa": True,
            "distinct_oracle_actions_at_least": 3,
            "active_bayes_gain_over_blind_at_least": 0.003,
            "active_bayes_gain_over_passive_bayes_at_least": 0.001,
            "active_bayes_gain_over_active_map_at_least": 0.005,
            "paired_ci95_lower_vs_blind_passive_bayes_active_map_above": 0.0,
            "oracle_headroom_fraction_at_least": 0.20,
            "active_bayes_harm_worlds_no_more_than_active_map": True,
        },
        "stage_order": [
            "all_passive_and_active_prefixes",
            "all_noisy_observations_and_decisions",
            "decision_barrier_and_pre_future_gate",
            "all_task_action_futures",
            "score",
        ],
        "task_future_before_decision_barrier": False,
        "retry_authorized": False,
        "replacement_authorized": False,
        "continuous_truth_protocol_automatically_authorized": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
        "new_recordings": False,
        "gpu_work": False,
        "push_or_merge": False,
    }


def prefix_task(probe: int, batch: int) -> dict[str, Any]:
    if (
        type(probe) is not int
        or probe not in range(2)
        or type(batch) is not int
        or batch not in range(4)
    ):
        raise ValueError("registered active-Bayes prefix task required")
    indices = list(range(8 * batch, 8 * batch + 8))
    return {
        "kind": "prefix_only",
        "name": f"prefix-{'passive' if probe == 0 else 'active'}-{batch}",
        "probe": probe,
        "batch": batch,
        "world_indices": indices,
    }


def future_task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(WORLD_COUNT):
        raise ValueError("registered active-Bayes future task required")
    return {
        "kind": "task_action_future",
        "name": f"future-{index:02d}",
        "world_index": index,
    }


def sensor_errors() -> tuple[Array, Array]:
    rng = np.random.default_rng(SENSOR_SEED)
    bias = rng.normal(
        0,
        BIAS_STD_M,
        (WORLD_COUNT, SENSOR_DRAWS, 1, 1, 3),
    )
    noise = rng.normal(
        0,
        NOISE_STD_M,
        (WORLD_COUNT, SENSOR_DRAWS, 3, 4, 3),
    )
    return np.asarray(bias, dtype=np.float64), np.asarray(noise, dtype=np.float64)


def infer_decisions(histories: Array, rewards: Array, truth_prefixes: Array) -> dict[str, Array]:
    history = np.asarray(histories, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    truth = np.asarray(truth_prefixes, dtype=np.float64)
    prior = np.asarray(prior_weights(), dtype=np.float64)
    if (
        history.shape != (2, 27, 3, 4, 3)
        or reward.shape != (27, 7)
        or truth.shape != (2, WORLD_COUNT, 3, 4, 3)
        or any(not np.isfinite(value).all() for value in (history, reward, truth))
    ):
        raise ValueError("complete finite active-Bayes input banks required")
    bias, noise = sensor_errors()
    observation = truth[:, :, None] + bias[None] + noise[None]
    weights: Array = np.empty(
        (2, WORLD_COUNT, SENSOR_DRAWS, 27), dtype=np.float64
    )
    map_action: Array = np.empty(
        (2, WORLD_COUNT, SENSOR_DRAWS), dtype=np.int64
    )
    bayes_action: Array = np.empty_like(map_action)
    for candidate in range(2):
        for world in range(WORLD_COUNT):
            posterior = _posterior_weights(
                observation[candidate, world], history[candidate], prior
            )
            weights[candidate, world] = posterior
            map_world = np.argmax(posterior, axis=1)
            map_action[candidate, world] = np.argmax(reward[map_world], axis=1)
            bayes_action[candidate, world] = np.argmax(posterior @ reward, axis=1)
    blind = int(np.argmax(prior @ reward))
    decisions = np.stack(
        [
            np.full((WORLD_COUNT, SENSOR_DRAWS), blind, dtype=np.int64),
            map_action[0],
            bayes_action[0],
            map_action[1],
            bayes_action[1],
        ],
        axis=-1,
    )
    return {
        "truth_prefix_m": truth,
        "shared_bias_m": bias,
        "independent_noise_m": noise,
        "observation_m": observation,
        "posterior_weights": weights,
        "decisions": decisions,
    }


def pre_future_checks(decisions: Array, *, all_prefix_qa: bool) -> dict[str, Any]:
    value = np.asarray(decisions)
    if (
        value.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or value.dtype.kind not in "iu"
        or np.any((value < 0) | (value > 6))
        or np.any(value[:, :, 0] != value[0, 0, 0])
    ):
        raise ValueError("complete valid pre-future decisions required")
    nonblind = int(np.count_nonzero(value[:, :, 4] != value[:, :, 0]))
    differs_map = int(np.count_nonzero(value[:, :, 4] != value[:, :, 3]))
    distinct = int(len(np.unique(value[:, :, 4])))
    checks = {
        "all_prefix_batches_native_qualified": bool(all_prefix_qa),
        "active_bayes_nonblind_sensor_decisions_at_least_32": nonblind >= 32,
        "active_bayes_differs_from_active_map_at_least_16": differs_map >= 16,
        "distinct_active_bayes_actions_at_least_2": distinct >= 2,
    }
    return {
        "active_bayes_nonblind_sensor_decisions": nonblind,
        "active_bayes_differs_from_active_map": differs_map,
        "distinct_active_bayes_actions": distinct,
        "checks": checks,
        "pre_future_gate_passed": bool(all(checks.values())),
    }


def _bootstrap_ci(values: Array) -> list[float]:
    difference = np.asarray(values, dtype=np.float64)
    if difference.shape != (WORLD_COUNT,) or not np.isfinite(difference).all():
        raise ValueError("one finite value per registered world required")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        WORLD_COUNT,
        size=(BOOTSTRAP_REPLICATES, WORLD_COUNT),
    )
    quantile: Array = np.asarray(
        np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    )
    return [float(quantile[0]), float(quantile[1])]


def score(decisions: Array, rewards: Array, *, all_native_qa: bool) -> dict[str, Any]:
    decision = np.asarray(decisions)
    reward = np.asarray(rewards, dtype=np.float64)
    if (
        decision.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or decision.dtype.kind not in "iu"
        or reward.shape != (WORLD_COUNT, 7)
        or not np.isfinite(reward).all()
        or np.any((decision < 0) | (decision > 6))
    ):
        raise ValueError("complete active-Bayes decisions and rewards required")
    selected = np.take_along_axis(
        reward[:, None, :],
        decision,
        axis=2,
    )
    world_reward = selected.mean(axis=1)
    blind = world_reward[:, 0]
    gain = world_reward - blind[:, None]
    arms: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES):
        world_harm = int(np.count_nonzero(gain[:, index] < -REWARD_MARGIN))
        draw_gain = selected[:, :, index] - selected[:, :, 0]
        arms[name] = {
            "mean_native_reward": float(world_reward[:, index].mean()),
            "mean_gain_over_blind": float(gain[:, index].mean()),
            "gain_ci95": _bootstrap_ci(gain[:, index]),
            "action_counts": [
                int(np.count_nonzero(decision[:, :, index] == action))
                for action in range(7)
            ],
            "nonblind_sensor_decisions": int(
                np.count_nonzero(decision[:, :, index] != decision[:, :, 0])
            ),
            "worlds_harmed_beyond_numeric_margin": world_harm,
            "sensor_decisions_harmed_beyond_numeric_margin": int(
                np.count_nonzero(draw_gain < -REWARD_MARGIN)
            ),
            "oracle_action_rate": float(
                np.mean(decision[:, :, index] == np.argmax(reward, axis=1)[:, None])
            ),
        }
    primary = world_reward[:, 4]
    paired: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES[:-1]):
        difference = primary - world_reward[:, index]
        paired[name] = {
            "mean_gain": float(difference.mean()),
            "ci95": _bootstrap_ci(difference),
        }
    oracle = np.max(reward, axis=1)
    oracle_headroom = float(np.mean(oracle - blind))
    active_gain = arms["active_bayes"]["mean_gain_over_blind"]
    fraction = active_gain / oracle_headroom if oracle_headroom > 0 else 0.0
    checks = {
        "complete_32_world_denominator": True,
        "all_native_qa": bool(all_native_qa),
        "distinct_oracle_actions_at_least_3": len(np.unique(np.argmax(reward, axis=1)))
        >= 3,
        "active_bayes_gain_over_blind_at_least_0_003": active_gain >= 0.003,
        "active_bayes_gain_over_passive_bayes_at_least_0_001": paired[
            "passive_bayes"
        ]["mean_gain"]
        >= 0.001,
        "active_bayes_gain_over_active_map_at_least_0_005": paired["active_map"][
            "mean_gain"
        ]
        >= 0.005,
        "positive_paired_ci_vs_blind": paired["blind_prior"]["ci95"][0] > 0,
        "positive_paired_ci_vs_passive_bayes": paired["passive_bayes"]["ci95"][0]
        > 0,
        "positive_paired_ci_vs_active_map": paired["active_map"]["ci95"][0] > 0,
        "captures_at_least_20pct_oracle_headroom": fraction >= 0.20,
        "active_bayes_harm_worlds_no_more_than_active_map": arms["active_bayes"][
            "worlds_harmed_beyond_numeric_margin"
        ]
        <= arms["active_map"]["worlds_harmed_beyond_numeric_margin"],
    }
    return {
        "schema": "dlolab-slingshot-active-bayes-score-v1",
        "arms": arms,
        "paired_active_bayes_gain": paired,
        "oracle_mean_native_reward": float(oracle.mean()),
        "oracle_headroom_over_blind": oracle_headroom,
        "oracle_headroom_fraction_captured": float(fraction),
        "distinct_oracle_actions": int(len(np.unique(np.argmax(reward, axis=1)))),
        "checks": checks,
        "source_gate_passed": bool(all(checks.values())),
        "ordinary_worlds": WORLD_COUNT,
        "sensor_draws_per_world": SENSOR_DRAWS,
        "technical_failures": 0,
        "replacements": 0,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "new_recordings": False,
    }
