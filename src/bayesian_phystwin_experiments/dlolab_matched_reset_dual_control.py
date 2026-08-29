"""Matched-reset active material identification and native task control."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from .coupled_action_regret import bias_marginalized_weights
from .deform_state_restart import array_digest
from .dlolab_native import DloLabConfig

PARTICLE_SCALES = np.asarray(
    [0.45, 0.58, 0.74, 0.90, 1.08, 1.30, 1.58, 1.92, 2.30],
    dtype=np.float64,
)
PROBE_NAMES = (
    "null_hold",
    "low_frequency_lateral",
    "high_frequency_lateral",
    "high_frequency_vertical",
)
PROBE_STEPS = 60
PROBE_TIMES = (14, 29, 44, 59)
PROBE_NODES = (4, 8, 12, 15)
PROBE_AMPLITUDE_M = 0.018
FIXED_CONTROL_PROBE_INDEX = 1
ACTION_STEPS = 80
ACTION_AMPLITUDES_M = np.asarray(
    [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10],
    dtype=np.float64,
)
GOALS_Y_M = np.asarray([-0.035, 0.035], dtype=np.float64)
TRUTH_COUNT = 72
TRUTH_SEED = 261101
NOISE_SEED = 261102
BOOTSTRAP_SEED = 261103
MI_SEED = 261104
MI_DRAWS_PER_WORLD = 1024
NOISE_STD_M = 0.0015
SHARED_BIAS_STD_M = 0.004
EFFORT_WEIGHT = 0.002
PAIR_TOLERANCE_M2 = 1e-12
ARMS = (
    "best_fixed",
    "null_bayes",
    "fixed_probe_bayes",
    "active_map",
    "active_bayes",
    "active_guarded",
    "oracle",
)


def protocol() -> dict[str, Any]:
    config = DloLabConfig()
    return {
        "schema": "dlolab-matched-reset-dual-control-source-v1",
        "role": "controlled_public_native_dual_control_source_study_not_confirmation",
        "native_interface": "DloLabRuntime public Genesis ROD CPU float64",
        "config": dataclasses.asdict(config),
        "hidden_variable": "continuous bending modulus only",
        "particle_bending_scales": PARTICLE_SCALES.tolist(),
        "truth_bending_scale": "log-uniform[0.48,2.20]",
        "truth_count": TRUTH_COUNT,
        "truth_seed": TRUTH_SEED,
        "probe_names": list(PROBE_NAMES),
        "probe_steps": PROBE_STEPS,
        "probe_times_zero_based": list(PROBE_TIMES),
        "probe_nodes": list(PROBE_NODES),
        "probe_amplitude_m": PROBE_AMPLITUDE_M,
        "probe_selection": "maximum_expected_material_mutual_information",
        "probe_selection_uses_task_reward_or_future": False,
        "probe_selection_tie_break": "lowest_index",
        "fixed_active_probe_control_index": FIXED_CONTROL_PROBE_INDEX,
        "probe_draws_per_world": MI_DRAWS_PER_WORLD,
        "probe_monte_carlo_seed": MI_SEED,
        "action_steps": ACTION_STEPS,
        "action_amplitudes_m": ACTION_AMPLITUDES_M.tolist(),
        "action_profile": "single lateral half-sine returning both clamps exactly",
        "goal_lateral_offsets_m": GOALS_Y_M.tolist(),
        "task_loss": "last-eight-frame four-tip-node squared lateral error plus effort",
        "effort_weight": EFFORT_WEIGHT,
        "noise_std_m": NOISE_STD_M,
        "shared_translation_bias_std_m": SHARED_BIAS_STD_M,
        "noise_seed": NOISE_SEED,
        "arms": list(ARMS),
        "primary_arm": "active_bayes",
        "matched_reset": {
            "probe_and_task_are_separate_native_branches": True,
            "every_branch_restores_one_preprobe_native_snapshot": True,
            "restore_requires_all_native_state_field_digests_equal": True,
            "probe_mechanics_cannot_enter_task_state": True,
            "command_and_realized_state_are_not_conflated": True,
        },
        "staged_information_boundary": [
            "particle_probe_and_task_bank",
            "reward_blind_probe_selection",
            "truth_probe_observations",
            "decision_seal",
            "truth_task_futures",
            "score",
        ],
        "probe_gate": {
            "selected_probe_nonnull": True,
            "selected_probe_differs_from_fixed_control": True,
            "minimum_information_gain_over_null_nats": 0.10,
            "minimum_map_accuracy_gain_over_null": 0.10,
        },
        "task_headroom_gate": {
            "minimum_material_dependent_oracle_actions_per_goal": 2,
            "minimum_oracle_gain_over_goal_conditioned_best_fixed": 0.05,
        },
        "source_value_gate": {
            "all_truth_episodes_complete_no_replacements": True,
            "minimum_active_gain_over_best_fixed": 0.03,
            "minimum_active_gain_over_null_bayes": 0.01,
            "minimum_active_gain_over_fixed_probe_bayes": 0.005,
            "paired_bootstrap_lower_gain_over_all_controls": 0.0,
            "minimum_active_nonfixed_decisions": 12,
            "maximum_active_harm_fraction_vs_null": 0.20,
        },
        "bootstrap_replicates": 10000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "exact_fallback": "best_fixed goal-conditioned action",
        "source_only": True,
        "public_native_simulator_only": True,
        "new_recordings": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
        "automatic_fresh_evaluation": False,
        "retry_authorized": False,
    }


def particle_bending() -> np.ndarray:
    return DloLabConfig().bending_modulus * PARTICLE_SCALES


def truth_partition() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(TRUTH_SEED)
    scale = np.exp(rng.uniform(np.log(0.48), np.log(2.20), TRUTH_COUNT))
    goal_index = np.arange(TRUTH_COUNT, dtype=np.int64) % len(GOALS_Y_M)
    return {
        "bending": DloLabConfig().bending_modulus * scale,
        "goal_index": goal_index,
        "goal_y_m": GOALS_Y_M[goal_index],
    }


def _initial_clamps(clamps: np.ndarray) -> np.ndarray:
    value = np.asarray(clamps, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (2, 3) or not np.isfinite(value).all():
        raise ValueError("finite batched two-clamp initial state required")
    return value


def probe_commands(clamps: np.ndarray, probe_index: int) -> np.ndarray:
    base = _initial_clamps(clamps)
    if type(probe_index) is not int or probe_index not in range(len(PROBE_NAMES)):
        raise ValueError("registered probe index required")
    phase = np.linspace(0.0, 1.0, PROBE_STEPS)
    offset = np.zeros((PROBE_STEPS, 3), dtype=np.float64)
    if probe_index == 1:
        offset[:, 1] = PROBE_AMPLITUDE_M * np.sin(np.pi * phase)
    elif probe_index == 2:
        offset[:, 1] = PROBE_AMPLITUDE_M * np.sin(2 * np.pi * phase)
    elif probe_index == 3:
        offset[:, 2] = PROBE_AMPLITUDE_M * np.sin(2 * np.pi * phase)
    offset[[0, -1]] = 0.0
    result = base[None] + offset[:, None, None]
    if (
        result.shape != (PROBE_STEPS, len(base), 2, 3)
        or not np.array_equal(result[0], base)
        or not np.allclose(result[-1], base, rtol=0, atol=1e-17)
        or np.max(np.linalg.norm(result - base[None], axis=-1))
        > PROBE_AMPLITUDE_M + 1e-12
    ):
        raise ValueError("probe command contract changed")
    result[-1] = base
    return result


def action_commands(clamps: np.ndarray, action_index: int) -> np.ndarray:
    base = _initial_clamps(clamps)
    if type(action_index) is not int or action_index not in range(
        len(ACTION_AMPLITUDES_M)
    ):
        raise ValueError("registered action index required")
    phase = np.linspace(0.0, 1.0, ACTION_STEPS)
    offset = ACTION_AMPLITUDES_M[action_index] * np.sin(np.pi * phase)
    offset[[0, -1]] = 0.0
    result = base[None].copy() + np.stack(
        [np.zeros_like(offset), offset, np.zeros_like(offset)], axis=1
    )[:, None, None]
    if (
        result.shape != (ACTION_STEPS, len(base), 2, 3)
        or not np.array_equal(result[0], base)
        or not np.allclose(result[-1], base, rtol=0, atol=1e-17)
    ):
        raise ValueError("task action contract changed")
    result[-1] = base
    return result


def probe_features(trajectory: np.ndarray, initial: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectory, dtype=np.float64)
    origin = np.asarray(initial, dtype=np.float64)
    if (
        value.ndim != 4
        or value.shape[1] != PROBE_STEPS
        or value.shape[2:] != (DloLabConfig().node_count, 3)
        or origin.shape != (len(value), DloLabConfig().node_count, 3)
        or not np.isfinite(value).all()
        or not np.isfinite(origin).all()
    ):
        raise ValueError("complete native probe trajectory required")
    return value[:, PROBE_TIMES][:, :, PROBE_NODES] - origin[:, None, PROBE_NODES]


def noisy_probe_observations(features: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(features, dtype=np.float64)
    if value.shape[1:] != (len(PROBE_TIMES), len(PROBE_NODES), 3):
        raise ValueError("registered probe features required")
    rng = np.random.default_rng(NOISE_SEED)
    bias = rng.normal(0, SHARED_BIAS_STD_M, (len(value), 1, 1, 3))
    noise = rng.normal(0, NOISE_STD_M, value.shape)
    return {"observation": value + bias + noise, "bias": bias, "noise": noise}


def posterior_weights(observation: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return bias_marginalized_weights(
        observation,
        predictions,
        noise_std_m=NOISE_STD_M,
        shared_bias_std_m=SHARED_BIAS_STD_M,
    )


def probe_information(particle_features: np.ndarray) -> dict[str, Any]:
    value = np.asarray(particle_features, dtype=np.float64)
    expected = (
        len(PROBE_NAMES),
        len(PARTICLE_SCALES),
        len(PROBE_TIMES),
        len(PROBE_NODES),
        3,
    )
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError("complete reward-blind particle probe bank required")
    rng = np.random.default_rng(MI_SEED)
    entropy = np.zeros(len(PROBE_NAMES), dtype=np.float64)
    accuracy = np.zeros(len(PROBE_NAMES), dtype=np.float64)
    for probe in range(len(PROBE_NAMES)):
        for world in range(len(PARTICLE_SCALES)):
            bias = rng.normal(
                0,
                SHARED_BIAS_STD_M,
                (MI_DRAWS_PER_WORLD, 1, 1, 3),
            )
            noise = rng.normal(
                0,
                NOISE_STD_M,
                (MI_DRAWS_PER_WORLD, len(PROBE_TIMES), len(PROBE_NODES), 3),
            )
            for draw in range(MI_DRAWS_PER_WORLD):
                weight = posterior_weights(
                    value[probe, world] + bias[draw] + noise[draw], value[probe]
                )
                positive = weight > 0
                entropy[probe] -= float(
                    np.sum(weight[positive] * np.log(weight[positive]))
                ) / (len(PARTICLE_SCALES) * MI_DRAWS_PER_WORLD)
                accuracy[probe] += float(np.argmax(weight) == world) / (
                    len(PARTICLE_SCALES) * MI_DRAWS_PER_WORLD
                )
    information = math.log(len(PARTICLE_SCALES)) - entropy
    selected = int(np.argmax(information))
    checks = {
        "selected_probe_nonnull": selected != 0,
        "selected_probe_differs_from_fixed_control": selected
        != FIXED_CONTROL_PROBE_INDEX,
        "information_gain_over_null": information[selected] - information[0] >= 0.10,
        "map_accuracy_gain_over_null": accuracy[selected] - accuracy[0] >= 0.10,
    }
    return {
        "selected_probe_index": selected,
        "selected_probe_name": PROBE_NAMES[selected],
        "mutual_information_nats": information.tolist(),
        "map_accuracy": accuracy.tolist(),
        "information_gain_over_null_nats": float(information[selected] - information[0]),
        "map_accuracy_gain_over_null": float(accuracy[selected] - accuracy[0]),
        "checks": checks,
        "passed": all(checks.values()),
        "task_reward_read": False,
        "particle_probe_sha256": array_digest(value),
    }


def task_losses(futures: np.ndarray, goals_y_m: np.ndarray) -> np.ndarray:
    value = np.asarray(futures, dtype=np.float64)
    goals = np.asarray(goals_y_m, dtype=np.float64)
    expected = (
        len(ACTION_AMPLITUDES_M),
        ACTION_STEPS,
        DloLabConfig().node_count,
        3,
    )
    if value.ndim != 5 or value.shape[1:] != expected or goals.ndim != 1:
        raise ValueError("complete action futures and one-dimensional goals required")
    if not np.isfinite(value).all() or not np.isfinite(goals).all():
        raise ValueError("task futures and goals must be finite")
    final = value[:, :, -8:, -4:, 1].mean(axis=(2, 3))
    error = final[:, None, :] - goals[None, :, None]
    effort = EFFORT_WEIGHT * ACTION_AMPLITUDES_M**2
    return error**2 + effort[None, None]


def realized_task_losses(
    futures: np.ndarray, goals_y_m: np.ndarray
) -> np.ndarray:
    value = np.asarray(futures, dtype=np.float64)
    goals = np.asarray(goals_y_m, dtype=np.float64)
    if value.ndim != 5 or goals.shape != (len(value),):
        raise ValueError("one registered goal per truth trajectory required")
    table = task_losses(value, goals)
    return table[np.arange(len(value)), np.arange(len(value))]


def task_headroom(particle_losses: np.ndarray) -> dict[str, Any]:
    loss = np.asarray(particle_losses, dtype=np.float64)
    expected = (len(PARTICLE_SCALES), len(GOALS_Y_M), len(ACTION_AMPLITUDES_M))
    if loss.shape != expected or not np.isfinite(loss).all():
        raise ValueError("complete particle task-loss table required")
    fixed = np.argmin(loss.mean(axis=0), axis=1)
    oracle = np.argmin(loss, axis=2)
    fixed_loss = np.take_along_axis(
        loss, np.broadcast_to(fixed[None, :, None], loss.shape[:2] + (1,)), axis=2
    )[..., 0]
    oracle_loss = np.min(loss, axis=2)
    denominator = max(float(fixed_loss.mean()), np.finfo(np.float64).eps)
    gain = float((fixed_loss.mean() - oracle_loss.mean()) / denominator)
    distinct = [int(len(np.unique(oracle[:, goal]))) for goal in range(len(GOALS_Y_M))]
    checks = {
        "at_least_two_material_actions_per_goal": min(distinct) >= 2,
        "oracle_gain_over_best_fixed_at_least_5pct": gain >= 0.05,
    }
    return {
        "goal_conditioned_best_fixed_action": fixed.tolist(),
        "oracle_actions": oracle.tolist(),
        "distinct_oracle_actions_per_goal": distinct,
        "oracle_relative_gain_over_best_fixed": gain,
        "checks": checks,
        "passed": all(checks.values()),
    }


def seal_decisions(
    active_observations: np.ndarray,
    null_observations: np.ndarray,
    fixed_probe_observations: np.ndarray,
    active_particle_features: np.ndarray,
    null_particle_features: np.ndarray,
    fixed_probe_particle_features: np.ndarray,
    particle_losses: np.ndarray,
    goal_index: np.ndarray,
) -> dict[str, np.ndarray]:
    active = np.asarray(active_observations, dtype=np.float64)
    null = np.asarray(null_observations, dtype=np.float64)
    fixed_observation = np.asarray(fixed_probe_observations, dtype=np.float64)
    active_prediction = np.asarray(active_particle_features, dtype=np.float64)
    null_prediction = np.asarray(null_particle_features, dtype=np.float64)
    fixed_prediction = np.asarray(fixed_probe_particle_features, dtype=np.float64)
    losses = np.asarray(particle_losses, dtype=np.float64)
    goals = np.asarray(goal_index)
    count = len(active)
    if (
        active.shape != null.shape
        or active.shape != fixed_observation.shape
        or active.shape[1:] != (len(PROBE_TIMES), len(PROBE_NODES), 3)
        or active_prediction.shape != (len(PARTICLE_SCALES),) + active.shape[1:]
        or null_prediction.shape != active_prediction.shape
        or fixed_prediction.shape != active_prediction.shape
        or losses.shape
        != (len(PARTICLE_SCALES), len(GOALS_Y_M), len(ACTION_AMPLITUDES_M))
        or goals.shape != (count,)
        or goals.dtype.kind not in "iu"
        or np.any((goals < 0) | (goals >= len(GOALS_Y_M)))
    ):
        raise ValueError("registered decision inputs required")
    fixed = np.argmin(losses.mean(axis=0), axis=1)
    decisions = {name: np.zeros(count, dtype=np.int64) for name in ARMS[:-1]}
    active_weight = np.zeros((count, len(PARTICLE_SCALES)), dtype=np.float64)
    null_weight = np.zeros_like(active_weight)
    fixed_weight = np.zeros_like(active_weight)
    for episode in range(count):
        a_weight = posterior_weights(active[episode], active_prediction)
        n_weight = posterior_weights(null[episode], null_prediction)
        f_weight = posterior_weights(fixed_observation[episode], fixed_prediction)
        active_weight[episode] = a_weight
        null_weight[episode] = n_weight
        fixed_weight[episode] = f_weight
        table = losses[:, goals[episode]]
        decisions["best_fixed"][episode] = fixed[goals[episode]]
        decisions["null_bayes"][episode] = int(np.argmin(n_weight @ table))
        decisions["fixed_probe_bayes"][episode] = int(np.argmin(f_weight @ table))
        decisions["active_map"][episode] = int(np.argmin(table[np.argmax(a_weight)]))
        decisions["active_bayes"][episode] = int(np.argmin(a_weight @ table))
        expected = a_weight @ table
        fixed_action = int(fixed[goals[episode]])
        candidate = int(np.argmin(expected))
        improve_probability = float(
            a_weight[table[:, candidate] + PAIR_TOLERANCE_M2 < table[:, fixed_action]].sum()
        )
        decisions["active_guarded"][episode] = (
            candidate
            if expected[candidate] + PAIR_TOLERANCE_M2 < expected[fixed_action]
            and improve_probability >= 0.8
            else fixed_action
        )
    return {
        **decisions,
        "active_weights": active_weight,
        "null_weights": null_weight,
        "fixed_probe_weights": fixed_weight,
        "goal_index": goals.astype(np.int64),
    }


def bootstrap_interval(delta: np.ndarray) -> list[float]:
    value = np.asarray(delta, dtype=np.float64)
    if value.shape != (TRUTH_COUNT,) or not np.isfinite(value).all():
        raise ValueError("complete truth denominator required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(10000, dtype=np.float64)
    for start in range(0, len(means), 250):
        size = min(250, len(means) - start)
        index = rng.integers(0, len(value), (size, len(value)))
        means[start : start + size] = value[index].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def score_source(decisions: dict[str, np.ndarray], truth_losses: np.ndarray) -> dict[str, Any]:
    losses = np.asarray(truth_losses, dtype=np.float64)
    if losses.shape != (TRUTH_COUNT, len(ACTION_AMPLITUDES_M)):
        raise ValueError("complete truth action losses required")
    selected_loss: dict[str, np.ndarray] = {}
    for arm in ARMS[:-1]:
        index = np.asarray(decisions[arm])
        if index.shape != (TRUTH_COUNT,) or index.dtype.kind not in "iu":
            raise ValueError("complete sealed decisions required")
        selected_loss[arm] = losses[np.arange(TRUTH_COUNT), index]
    selected_loss["oracle"] = losses.min(axis=1)
    means = {arm: float(value.mean()) for arm, value in selected_loss.items()}
    fixed = selected_loss["best_fixed"]
    null = selected_loss["null_bayes"]
    fixed_probe = selected_loss["fixed_probe_bayes"]
    active = selected_loss["active_bayes"]
    fixed_gain = (fixed - active) / max(float(fixed.mean()), np.finfo(float).eps)
    null_gain = (null - active) / max(float(null.mean()), np.finfo(float).eps)
    fixed_probe_gain = (fixed_probe - active) / max(
        float(fixed_probe.mean()), np.finfo(float).eps
    )
    fixed_ci = bootstrap_interval(fixed - active)
    null_ci = bootstrap_interval(null - active)
    fixed_probe_ci = bootstrap_interval(fixed_probe - active)
    harm = float(np.mean(active > null + PAIR_TOLERANCE_M2))
    nonfixed = int(
        np.sum(decisions["active_bayes"] != decisions["best_fixed"])
    )
    checks = {
        "complete_truth_denominator": all(len(value) == TRUTH_COUNT for value in selected_loss.values()),
        "active_gain_over_best_fixed_at_least_3pct": float(fixed_gain.mean()) >= 0.03,
        "active_gain_over_null_bayes_at_least_1pct": float(null_gain.mean()) >= 0.01,
        "active_gain_over_fixed_probe_bayes_at_least_0_5pct": float(
            fixed_probe_gain.mean()
        )
        >= 0.005,
        "bootstrap_lower_gain_positive_vs_fixed": fixed_ci[0] > 0,
        "bootstrap_lower_gain_positive_vs_null": null_ci[0] > 0,
        "bootstrap_lower_gain_positive_vs_fixed_probe": fixed_probe_ci[0] > 0,
        "at_least_12_nonfixed_decisions": nonfixed >= 12,
        "harm_fraction_vs_null_at_most_20pct": harm <= 0.20,
    }
    return {
        "mean_loss_m2": means,
        "active_relative_gain_over_best_fixed": float(fixed_gain.mean()),
        "active_relative_gain_over_null_bayes": float(null_gain.mean()),
        "active_relative_gain_over_fixed_probe_bayes": float(
            fixed_probe_gain.mean()
        ),
        "paired_gain_ci95_m2": {
            "best_fixed": fixed_ci,
            "null_bayes": null_ci,
            "fixed_probe_bayes": fixed_probe_ci,
        },
        "active_harm_fraction_vs_null": harm,
        "active_nonfixed_decisions": nonfixed,
        "checks": checks,
        "source_gate_passed": all(checks.values()),
    }
