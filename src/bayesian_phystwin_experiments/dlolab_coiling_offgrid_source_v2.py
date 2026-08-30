"""Source-only off-grid transfer screen for guarded coiling decisions."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .deform_state_restart import array_digest
from .dlolab_coiling_query_competence_v1 import (
    ACTION_NAMES,
    DRAW_COUNT,
    INDEPENDENT_NOISE_STD_M,
    PAIR_MARGIN,
    SHARED_BIAS_STD_M,
    action_bank,
    native_qa,
)

Array: TypeAlias = NDArray[Any]

SOURCE_WORLD_ROWS = (
    (650.0, 900.0, -0.040, -0.008),
    (650.0, 3500.0, -0.026, 0.006),
    (650.0, 7000.0, -0.012, -0.004),
    (1400.0, 650.0, 0.004, 0.008),
    (1400.0, 3500.0, 0.018, -0.006),
    (1400.0, 7000.0, 0.034, 0.004),
    (3500.0, 650.0, -0.034, 0.002),
    (3500.0, 1800.0, -0.018, -0.008),
    (3500.0, 7000.0, 0.000, 0.006),
    (7000.0, 650.0, 0.012, -0.004),
    (7000.0, 1800.0, 0.026, 0.008),
    (7000.0, 4500.0, 0.040, -0.006),
)
DRAW_SEED = 271003
MINIMUM_EXPECTED_GAIN = 0.003
MAXIMUM_POSTERIOR_HARM_PROBABILITY = 0.05
PARENT_DEVELOPMENT_SUMMARY_SHA256 = (
    "410c246e375df47b6159b2c57995e77aaa097dd0ac63575507926fb108166051"
)


def source_worlds() -> list[dict[str, int | float]]:
    return [
        {
            "index": index,
            "bending_E": bending,
            "twisting_G": twisting,
            "offset_x_m": offset_x,
            "offset_y_m": offset_y,
        }
        for index, (bending, twisting, offset_x, offset_y) in enumerate(
            SOURCE_WORLD_ROWS
        )
    ]


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(len(source_worlds())):
        raise ValueError("unregistered coiling off-grid source world")
    return {
        "index": index,
        "name": f"source-world-{index:02d}",
        "world": source_worlds()[index],
    }


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-coiling-offgrid-source-v2",
        "role": "source_only_transfer_gate_not_prospective_evidence",
        "task": "coiling",
        "parent_opened_world_role": "action_bank_development_only_not_in_denominator",
        "parent_development_summary_sha256": PARENT_DEVELOPMENT_SUMMARY_SHA256,
        "source_worlds": source_worlds(),
        "source_world_count": len(source_worlds()),
        "source_world_selection": "fixed_before_any_v2_native_execution",
        "source_worlds_disjoint_from_opened_nominal_world": True,
        "public_native_position_randomization_bounds_m": {
            "x": [-0.05, 0.05],
            "y": [-0.01, 0.01],
        },
        "controls_sha256": array_digest(action_bank()),
        "action_names": list(ACTION_NAMES),
        "unique_actions": 7,
        "shared_prefix_and_observation_policy_unchanged_from_v1": True,
        "native_reward_unchanged": True,
        "observation_noise": {
            "independent_std_m": INDEPENDENT_NOISE_STD_M,
            "shared_translation_bias_std_m": SHARED_BIAS_STD_M,
            "assumed_not_sensor_calibrated": True,
        },
        "guard": {
            "crossfit": "leave_one_source_world_out",
            "baseline": "best_fixed_action_on_training_fold",
            "minimum_posterior_expected_gain": MINIMUM_EXPECTED_GAIN,
            "maximum_posterior_harm_probability": MAXIMUM_POSTERIOR_HARM_PROBABILITY,
            "harm_margin": PAIR_MARGIN,
            "fallback": "exact_training_fold_best_fixed_action",
            "draw_count": DRAW_COUNT,
            "draw_seed": DRAW_SEED,
        },
        "source_gates": {
            "minimum_best_fixed_gain_over_prefix_hold": 0.005,
            "minimum_adjusted_oracle_headroom": 0.003,
            "minimum_distinct_oracle_actions": 2,
            "minimum_crossfit_guarded_mean_gain": 0.002,
            "minimum_worlds_with_gain_at_least_0_001": 4,
            "minimum_worlds_with_admission_probability_at_least_0_1": 3,
            "maximum_mean_observation_draw_harm_probability": 0.05,
            "maximum_single_world_observation_draw_harm_probability": 0.20,
        },
        "all_source_worlds_sealed_before_crossfit": True,
        "source_failure_stops_before_target_world_selection": True,
        "prospective_worlds_selected": False,
        "prospective_execution_authorized": False,
        "retry_authorized": False,
        "new_recordings": False,
        "gpu_work": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
    }


def rederive_native_qa(
    arrays: dict[str, Array], native: dict[str, Any], world: dict[str, Any]
) -> dict[str, Any]:
    value = cast(dict[str, Any], native_qa(arrays, native, world))
    value["checks"] = {key: bool(check) for key, check in value["checks"].items()}
    expected_offset = [world["offset_x_m"], world["offset_y_m"], 0.0]
    state = native.get("state_realization")
    measurements = native.get("state_measurements")
    value["checks"]["registered_state_offset"] = bool(
        state == {"offset_m": expected_offset}
    )
    value["checks"]["state_offset_realization"] = bool(
        isinstance(measurements, dict)
        and measurements.get("maximum_offset_realization_error_m", np.inf) <= 1e-12
    )
    value["state_realization"] = state
    value["state_measurements"] = measurements
    value["passed"] = bool(all(value["checks"].values()))
    return value


def _posterior(observation: Array, particles: Array) -> Array:
    observed = np.asarray(observation, dtype=np.float64)
    predicted = np.asarray(particles, dtype=np.float64)
    if (
        observed.ndim != 4
        or predicted.ndim != 4
        or observed.shape[1:] != predicted.shape[1:]
        or predicted.shape[1:] != (3, 5, 3)
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("aligned finite off-grid prefix observations required")
    residual = (observed[:, None] - predicted[None]).reshape(
        len(observed), len(predicted), -1, 3
    )
    mean = residual.mean(axis=2)
    centered = residual - mean[:, :, None]
    independent_var = INDEPENDENT_NOISE_STD_M**2
    shared_var = SHARED_BIAS_STD_M**2
    count = residual.shape[2]
    distance = np.sum(centered**2, axis=(2, 3)) / independent_var
    distance += count * np.sum(mean**2, axis=2) / (independent_var + count * shared_var)
    log_weight = -np.log(len(predicted)) - 0.5 * distance
    log_weight -= log_weight.max(axis=1, keepdims=True)
    weight = np.exp(log_weight)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.asarray(weight, dtype=np.float64)


def guarded_action_distribution(
    truth_prefix: Array,
    particle_prefix: Array,
    particle_rewards: Array,
    baseline_action: int,
    *,
    seed: int,
) -> dict[str, Any]:
    truth = np.asarray(truth_prefix, dtype=np.float64)
    particles = np.asarray(particle_prefix, dtype=np.float64)
    rewards = np.asarray(particle_rewards, dtype=np.float64)
    if (
        truth.shape != (3, 5, 3)
        or particles.ndim != 4
        or particles.shape[1:] != truth.shape
        or rewards.shape != (len(particles), 7)
        or type(baseline_action) is not int
        or baseline_action not in range(7)
        or type(seed) is not int
        or not np.isfinite(truth).all()
        or not np.isfinite(particles).all()
        or not np.isfinite(rewards).all()
    ):
        raise ValueError("complete source particles and baseline required")
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, SHARED_BIAS_STD_M, (DRAW_COUNT, 1, 1, 3))
    independent = rng.normal(0, INDEPENDENT_NOISE_STD_M, (DRAW_COUNT, 3, 5, 3))
    counts: NDArray[np.int64] = np.zeros(7, dtype=np.int64)
    admitted = 0
    batch_size = 256
    for start in range(0, DRAW_COUNT, batch_size):
        stop = min(start + batch_size, DRAW_COUNT)
        posterior = _posterior(
            truth + shared[start:stop] + independent[start:stop], particles
        )
        expected = posterior @ rewards
        candidate = np.argmax(expected, axis=1)
        row = np.arange(len(candidate))
        gain = expected[row, candidate] - expected[:, baseline_action]
        harm = rewards[:, candidate].T + PAIR_MARGIN < rewards[:, baseline_action]
        harm_probability = np.sum(posterior * harm, axis=1)
        accept = (gain >= MINIMUM_EXPECTED_GAIN) & (
            harm_probability <= MAXIMUM_POSTERIOR_HARM_PROBABILITY
        )
        selected = np.where(accept, candidate, baseline_action)
        counts += np.bincount(selected, minlength=7)
        admitted += int(np.count_nonzero(accept))
    probability = counts.astype(np.float64) / DRAW_COUNT
    if not np.isclose(probability.sum(), 1.0):
        raise RuntimeError("guarded action distribution did not normalize")
    return {
        "baseline_action": baseline_action,
        "action_probabilities": probability.tolist(),
        "admission_probability": admitted / DRAW_COUNT,
        "exact_fallback_probability": 1 - admitted / DRAW_COUNT,
    }


def source_crossfit(prefix: Array, rewards: Array) -> dict[str, Any]:
    feature = np.asarray(prefix, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    count = len(source_worlds())
    if (
        feature.shape != (count, 3, 5, 3)
        or reward.shape != (count, 7)
        or not np.isfinite(feature).all()
        or not np.isfinite(reward).all()
        or np.any((reward <= 0) | (reward > 1))
    ):
        raise ValueError("complete finite off-grid source bank required")
    expected = reward.mean(axis=0)
    full_fixed = int(np.argmax(expected))
    oracle_action = np.argmax(reward, axis=1)
    oracle_reward = float(np.mean(np.max(reward, axis=1)))
    best_fixed_reward = float(expected[full_fixed])
    rows = []
    for held_out in range(count):
        training = np.arange(count) != held_out
        training_rewards = reward[training]
        baseline = int(np.argmax(training_rewards.mean(axis=0)))
        decision = guarded_action_distribution(
            feature[held_out],
            feature[training],
            training_rewards,
            baseline,
            seed=DRAW_SEED + held_out,
        )
        probability = np.asarray(decision["action_probabilities"], dtype=np.float64)
        guarded_reward = float(probability @ reward[held_out])
        baseline_reward = float(reward[held_out, baseline])
        harm_probability = float(
            probability @ (reward[held_out] + PAIR_MARGIN < baseline_reward)
        )
        rows.append(
            {
                "held_out_world": held_out,
                **decision,
                "guarded_reward": guarded_reward,
                "baseline_reward": baseline_reward,
                "gain": guarded_reward - baseline_reward,
                "observation_draw_harm_probability": harm_probability,
            }
        )
    gains = np.asarray([row["gain"] for row in rows], dtype=np.float64)
    admission = np.asarray(
        [row["admission_probability"] for row in rows], dtype=np.float64
    )
    harm = np.asarray(
        [row["observation_draw_harm_probability"] for row in rows],
        dtype=np.float64,
    )
    oracle_headroom = oracle_reward - best_fixed_reward
    checks = {
        "best_fixed_gain_over_hold_at_least_0_005": bool(
            best_fixed_reward - float(expected[0]) >= 0.005
        ),
        "adjusted_oracle_headroom_at_least_0_003": bool(
            oracle_headroom - PAIR_MARGIN >= 0.003
        ),
        "at_least_two_distinct_oracle_actions": bool(
            len(set(oracle_action.tolist())) >= 2
        ),
        "crossfit_guarded_mean_gain_at_least_0_002": bool(gains.mean() >= 0.002),
        "at_least_four_worlds_gain_0_001": bool(np.count_nonzero(gains >= 0.001) >= 4),
        "at_least_three_worlds_admit_0_1": bool(
            np.count_nonzero(admission >= 0.1) >= 3
        ),
        "mean_harm_probability_at_most_0_05": bool(harm.mean() <= 0.05),
        "maximum_harm_probability_at_most_0_20": bool(harm.max() <= 0.20),
    }
    return {
        "best_fixed_action": full_fixed,
        "best_fixed_action_name": ACTION_NAMES[full_fixed],
        "best_fixed_reward": best_fixed_reward,
        "prefix_hold_reward": float(expected[0]),
        "oracle_reward": oracle_reward,
        "oracle_headroom": oracle_headroom,
        "distinct_oracle_actions": len(set(oracle_action.tolist())),
        "oracle_actions": oracle_action.tolist(),
        "crossfit_rows": rows,
        "crossfit_guarded_mean_reward": float(
            np.mean([row["guarded_reward"] for row in rows])
        ),
        "crossfit_baseline_mean_reward": float(
            np.mean([row["baseline_reward"] for row in rows])
        ),
        "crossfit_guarded_mean_gain": float(gains.mean()),
        "worlds_with_gain_at_least_0_001": int(np.count_nonzero(gains >= 0.001)),
        "worlds_with_admission_probability_at_least_0_1": int(
            np.count_nonzero(admission >= 0.1)
        ),
        "mean_observation_draw_harm_probability": float(harm.mean()),
        "maximum_observation_draw_harm_probability": float(harm.max()),
        "checks": checks,
        "source_gate_passed": bool(all(checks.values())),
        "prospective_execution_authorized": False,
    }
