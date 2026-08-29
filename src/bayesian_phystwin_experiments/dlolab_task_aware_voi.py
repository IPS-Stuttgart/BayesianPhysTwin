"""Task-aware value-of-information controls for matched-reset DLO probing."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .coupled_action_regret import bias_marginalized_weights
from .deform_state_restart import array_digest
from .dlolab_native import DloLabConfig

BENDING_SCALES = np.asarray([0.52, 0.72, 1.0, 1.4, 1.95], dtype=np.float64)
TWISTING_SCALES = np.asarray([0.20, 1.0, 5.0], dtype=np.float64)
PROBE_NAMES = (
    "null_hold",
    "slow_vertical_bend",
    "slow_lateral_bend",
    "slow_conical_twist",
    "fast_conical_twist",
)
NULL_PROBE_INDEX = 0
FIXED_PROBE_INDEX = 2
PROBE_STEPS = 72
PROBE_TIMES = (17, 35, 53, 71)
PROBE_NODES = (4, 8, 12, 15)
PROBE_AMPLITUDE_M = 0.0045
ACTION_STEPS = 100
ACTION_RAMP_STEPS = 50
ACTION_ROOT_Z_M = np.asarray(
    [-0.006, -0.0045, -0.003, -0.0015, 0.0, 0.0015, 0.003, 0.0045, 0.006],
    dtype=np.float64,
)
GOAL_TIP_Z_M = np.asarray([0.53, 0.58, 0.63], dtype=np.float64)
TRUTH_COUNT = 90
TRUTH_SEED = 261201
OBSERVATION_SEED = 261202
SELECTOR_SEED = 261203
BOOTSTRAP_SEED = 261204
SELECTOR_DRAWS_PER_WORLD = 384
NOISE_STD_M = 0.0010
SHARED_BIAS_STD_M = 0.0030
EFFORT_WEIGHT = 0.5
GUARD_MIN_EXPECTED_RELATIVE_GAIN = 0.005
ARMS = (
    "best_fixed",
    "null_bayes",
    "fixed_probe_bayes",
    "mi_probe_bayes",
    "task_aware_map",
    "task_aware_bayes",
    "task_aware_guarded",
    "oracle",
)
Array: TypeAlias = NDArray[Any]


def protocol() -> dict[str, Any]:
    config = DloLabConfig()
    return {
        "schema": "dlolab-task-aware-voi-source-v1",
        "role": "controlled_public_native_task_aware_dual_control_source_study",
        "native_interface": "DloLabRuntime public Genesis ROD CPU float64",
        "config": dataclasses.asdict(config),
        "hidden_variables": ["bending modulus", "twisting modulus"],
        "particle_bending_scales": BENDING_SCALES.tolist(),
        "particle_twisting_scales": TWISTING_SCALES.tolist(),
        "particle_world_count": particle_count(),
        "truth_bending_scale": "log-uniform[0.56,1.82]",
        "truth_twisting_scale": "log-uniform[0.24,4.20]",
        "truth_count": TRUTH_COUNT,
        "truth_seed": TRUTH_SEED,
        "probe_names": list(PROBE_NAMES),
        "probe_steps": PROBE_STEPS,
        "probe_times_zero_based": list(PROBE_TIMES),
        "probe_nodes": list(PROBE_NODES),
        "probe_amplitude_m": PROBE_AMPLITUDE_M,
        "fixed_probe_control_index": FIXED_PROBE_INDEX,
        "generic_information_control": "maximum full-latent mutual information",
        "primary_probe_selection": "minimum expected downstream Bayes task loss",
        "primary_probe_uses_particle_task_table": True,
        "primary_probe_uses_truth_futures": False,
        "selector_draws_per_particle_world": SELECTOR_DRAWS_PER_WORLD,
        "selector_seed": SELECTOR_SEED,
        "action_steps": ACTION_STEPS,
        "action_ramp_steps": ACTION_RAMP_STEPS,
        "action_root_vertical_offsets_m": ACTION_ROOT_Z_M.tolist(),
        "action_profile": "smooth root-segment vertical tilt followed by hold",
        "goal_tip_heights_m": GOAL_TIP_Z_M.tolist(),
        "task_loss": "last-16-frame four-tip-node squared height error plus effort",
        "effort_weight": EFFORT_WEIGHT,
        "observation_noise_std_m": NOISE_STD_M,
        "shared_translation_bias_std_m": SHARED_BIAS_STD_M,
        "observation_seed": OBSERVATION_SEED,
        "arms": list(ARMS),
        "primary_arm": "task_aware_bayes",
        "matched_reset": {
            "probe_and_task_are_separate_native_branches": True,
            "every_branch_restores_one_preprobe_native_snapshot": True,
            "restore_requires_all_native_state_field_digests_equal": True,
            "probe_mechanics_cannot_enter_task_state": True,
        },
        "staged_information_boundary": [
            "particle_probe_and_task_bank",
            "particle_selector_and_headroom_analysis",
            "truth_probe_observations",
            "decision_seal",
            "truth_task_futures",
            "score",
        ],
        "selector_gate": {
            "task_aware_probe_nonnull": True,
            "generic_mi_probe_nonnull": True,
            "task_aware_probe_differs_from_generic_mi": True,
            "minimum_task_aware_relative_risk_gain_over_null": 0.03,
            "minimum_task_aware_relative_risk_gain_over_generic_mi": 0.01,
            "minimum_task_aware_relative_risk_gain_over_fixed_probe": 0.01,
        },
        "task_headroom_gate": {
            "minimum_distinct_bending_conditioned_oracle_actions_per_goal": 2,
            "minimum_oracle_relative_gain_over_best_fixed": 0.08,
            "maximum_twisting_only_oracle_disagreement_fraction": 0.34,
        },
        "source_value_gate": {
            "all_truth_episodes_complete_no_replacements": True,
            "minimum_task_aware_gain_over_best_fixed": 0.03,
            "minimum_task_aware_gain_over_null_bayes": 0.01,
            "minimum_task_aware_gain_over_generic_mi_bayes": 0.005,
            "minimum_task_aware_gain_over_fixed_probe_bayes": 0.005,
            "paired_bootstrap_upper_loss_difference_below_zero": True,
            "minimum_task_aware_vs_mi_action_differences": 12,
            "maximum_task_aware_harm_fraction_vs_mi": 0.25,
        },
        "guard_minimum_expected_relative_gain": GUARD_MIN_EXPECTED_RELATIVE_GAIN,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "exact_fallback": "goal-conditioned best-fixed action",
        "distinct_from_closed_mi_only_protocol": True,
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


def particle_count() -> int:
    return len(BENDING_SCALES) * len(TWISTING_SCALES)


def particle_parameters() -> dict[str, Array]:
    bending, twisting = np.meshgrid(BENDING_SCALES, TWISTING_SCALES, indexing="ij")
    config = DloLabConfig()
    return {
        "bending": config.bending_modulus * bending.reshape(-1),
        "twisting": config.twisting_modulus * twisting.reshape(-1),
        "bending_index": np.repeat(np.arange(len(BENDING_SCALES)), len(TWISTING_SCALES)),
        "twisting_index": np.tile(np.arange(len(TWISTING_SCALES)), len(BENDING_SCALES)),
    }


def truth_partition() -> dict[str, Array]:
    rng = np.random.default_rng(TRUTH_SEED)
    config = DloLabConfig()
    bending = np.exp(rng.uniform(np.log(0.56), np.log(1.82), TRUTH_COUNT))
    twisting = np.exp(rng.uniform(np.log(0.24), np.log(4.20), TRUTH_COUNT))
    goal_index: Array = np.arange(TRUTH_COUNT, dtype=np.int64) % len(GOAL_TIP_Z_M)
    return {
        "bending": config.bending_modulus * bending,
        "twisting": config.twisting_modulus * twisting,
        "goal_index": goal_index,
        "goal_tip_z_m": GOAL_TIP_Z_M[goal_index],
    }


def _initial_clamps(clamps: Array) -> Array:
    value = np.asarray(clamps, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (2, 3) or not np.isfinite(value).all():
        raise ValueError("finite batched two-clamp initial state required")
    return value


def probe_commands(clamps: Array, probe_index: int) -> Array:
    base = _initial_clamps(clamps)
    if type(probe_index) is not int or probe_index not in range(len(PROBE_NAMES)):
        raise ValueError("registered probe index required")
    phase = np.linspace(0.0, 1.0, PROBE_STEPS)
    moving: Array = np.zeros((PROBE_STEPS, 3), dtype=np.float64)
    if probe_index == 1:
        moving[:, 2] = PROBE_AMPLITUDE_M * np.sin(np.pi * phase)
    elif probe_index == 2:
        moving[:, 1] = PROBE_AMPLITUDE_M * np.sin(np.pi * phase)
    elif probe_index in (3, 4):
        cycles = probe_index - 2
        angle = cycles * 2 * np.pi * phase
        moving[:, 1] = PROBE_AMPLITUDE_M / np.sqrt(2) * np.sin(angle)
        moving[:, 2] = 0.5 * PROBE_AMPLITUDE_M * (1 - np.cos(angle))
    moving[[0, -1]] = 0.0
    result: Array = np.broadcast_to(base[None], (PROBE_STEPS,) + base.shape).copy()
    result[:, :, 1] += moving[:, None]
    result[-1] = base
    if (
        result.shape != (PROBE_STEPS, len(base), 2, 3)
        or not np.array_equal(result[0], base)
        or not np.array_equal(result[-1], base)
        or np.max(np.linalg.norm(result - base[None], axis=-1))
        > PROBE_AMPLITUDE_M + 1e-12
    ):
        raise ValueError("probe command contract changed")
    return result


def action_commands(clamps: Array, action_index: int) -> Array:
    base = _initial_clamps(clamps)
    if type(action_index) is not int or action_index not in range(len(ACTION_ROOT_Z_M)):
        raise ValueError("registered task action index required")
    phase = np.linspace(0.0, 1.0, ACTION_RAMP_STEPS)
    ramp = phase * phase * (3 - 2 * phase)
    offset = np.concatenate(
        [
            ACTION_ROOT_Z_M[action_index] * ramp,
            np.full(ACTION_STEPS - ACTION_RAMP_STEPS, ACTION_ROOT_Z_M[action_index]),
        ]
    )
    result: Array = np.broadcast_to(base[None], (ACTION_STEPS,) + base.shape).copy()
    result[:, :, 1, 2] += offset[:, None]
    if (
        result.shape != (ACTION_STEPS, len(base), 2, 3)
        or not np.array_equal(result[0], base)
        or not np.allclose(
            result[-1, :, 1, 2] - base[:, 1, 2],
            ACTION_ROOT_Z_M[action_index],
            rtol=0,
            atol=1e-15,
        )
    ):
        raise ValueError("task action contract changed")
    return result


def probe_features(trajectory: Array, initial: Array) -> Array:
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
        raise ValueError("complete task-aware native probe trajectory required")
    return np.asarray(
        value[:, PROBE_TIMES][:, :, PROBE_NODES] - origin[:, None, PROBE_NODES]
    )


def posterior_weights(observation: Array, predictions: Array) -> Array:
    return np.asarray(
        bias_marginalized_weights(
            observation,
            predictions,
            noise_std_m=NOISE_STD_M,
            shared_bias_std_m=SHARED_BIAS_STD_M,
        ),
        dtype=np.float64,
    )


def task_losses(futures: Array, goals_z_m: Array) -> Array:
    value = np.asarray(futures, dtype=np.float64)
    goals = np.asarray(goals_z_m, dtype=np.float64)
    expected = (
        len(ACTION_ROOT_Z_M),
        ACTION_STEPS,
        DloLabConfig().node_count,
        3,
    )
    if value.ndim != 5 or value.shape[1:] != expected or goals.ndim != 1:
        raise ValueError("complete action futures and one-dimensional goals required")
    if not np.isfinite(value).all() or not np.isfinite(goals).all():
        raise ValueError("task futures and goals must be finite")
    final_tip = value[:, :, -16:, -4:, 2].mean(axis=(2, 3))
    error = final_tip[:, None, :] - goals[None, :, None]
    effort = EFFORT_WEIGHT * ACTION_ROOT_Z_M**2
    return np.asarray(error**2 + effort[None, None], dtype=np.float64)


def realized_task_losses(futures: Array, goals_z_m: Array) -> Array:
    value = np.asarray(futures, dtype=np.float64)
    goals = np.asarray(goals_z_m, dtype=np.float64)
    if value.ndim != 5 or goals.shape != (len(value),):
        raise ValueError("one registered task goal per truth trajectory required")
    table = task_losses(value, goals)
    return np.asarray(table[np.arange(len(value)), np.arange(len(value))])


def _relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(
        abs(reference), float(np.finfo(np.float64).eps)
    )


def selector_analysis(
    particle_features: Array,
    particle_losses: Array,
    *,
    draws_per_world: int = SELECTOR_DRAWS_PER_WORLD,
    seed: int = SELECTOR_SEED,
) -> dict[str, Any]:
    features = np.asarray(particle_features, dtype=np.float64)
    losses = np.asarray(particle_losses, dtype=np.float64)
    expected_features = (
        len(PROBE_NAMES),
        particle_count(),
        len(PROBE_TIMES),
        len(PROBE_NODES),
        3,
    )
    expected_losses = (particle_count(), len(GOAL_TIP_Z_M), len(ACTION_ROOT_Z_M))
    if (
        features.shape != expected_features
        or losses.shape != expected_losses
        or not np.isfinite(features).all()
        or not np.isfinite(losses).all()
        or type(draws_per_world) is not int
        or draws_per_world < 1
        or type(seed) is not int
    ):
        raise ValueError("complete finite particle features and task table required")
    rng = np.random.default_rng(seed)
    entropy: Array = np.zeros(len(PROBE_NAMES), dtype=np.float64)
    accuracy: Array = np.zeros(len(PROBE_NAMES), dtype=np.float64)
    expected_loss: Array = np.zeros(len(PROBE_NAMES), dtype=np.float64)
    denominator = particle_count() * draws_per_world
    for probe in range(len(PROBE_NAMES)):
        for world in range(particle_count()):
            bias = rng.normal(0, SHARED_BIAS_STD_M, (draws_per_world, 1, 1, 3))
            noise = rng.normal(
                0,
                NOISE_STD_M,
                (draws_per_world, len(PROBE_TIMES), len(PROBE_NODES), 3),
            )
            for draw in range(draws_per_world):
                weight = posterior_weights(
                    features[probe, world] + bias[draw] + noise[draw],
                    features[probe],
                )
                positive = weight > 0
                entropy[probe] -= float(np.sum(weight[positive] * np.log(weight[positive]))) / denominator
                accuracy[probe] += float(np.argmax(weight) == world) / denominator
                for goal in range(len(GOAL_TIP_Z_M)):
                    action = int(np.argmin(weight @ losses[:, goal]))
                    expected_loss[probe] += losses[world, goal, action] / (
                        denominator * len(GOAL_TIP_Z_M)
                    )
    information = math.log(particle_count()) - entropy
    mi_probe = int(np.argmax(information))
    task_probe = int(np.argmin(expected_loss))
    null_loss = float(expected_loss[NULL_PROBE_INDEX])
    mi_loss = float(expected_loss[mi_probe])
    fixed_loss = float(expected_loss[FIXED_PROBE_INDEX])
    task_loss = float(expected_loss[task_probe])
    checks = {
        "task_aware_probe_nonnull": task_probe != NULL_PROBE_INDEX,
        "generic_mi_probe_nonnull": mi_probe != NULL_PROBE_INDEX,
        "selectors_differ": task_probe != mi_probe,
        "task_risk_gain_over_null": _relative_gain(null_loss, task_loss) >= 0.03,
        "task_risk_gain_over_generic_mi": _relative_gain(mi_loss, task_loss) >= 0.01,
        "task_risk_gain_over_fixed_probe": _relative_gain(fixed_loss, task_loss) >= 0.01,
    }
    return {
        "task_aware_probe_index": task_probe,
        "task_aware_probe_name": PROBE_NAMES[task_probe],
        "generic_mi_probe_index": mi_probe,
        "generic_mi_probe_name": PROBE_NAMES[mi_probe],
        "mutual_information_nats": information.tolist(),
        "map_accuracy": accuracy.tolist(),
        "expected_task_loss": expected_loss.tolist(),
        "task_risk_relative_gain_over_null": _relative_gain(null_loss, task_loss),
        "task_risk_relative_gain_over_generic_mi": _relative_gain(mi_loss, task_loss),
        "task_risk_relative_gain_over_fixed_probe": _relative_gain(fixed_loss, task_loss),
        "checks": checks,
        "passed": all(checks.values()),
        "truth_futures_read": False,
        "particle_probe_sha256": array_digest(features),
        "particle_task_loss_sha256": array_digest(losses),
    }


def task_headroom(particle_losses: Array) -> dict[str, Any]:
    loss = np.asarray(particle_losses, dtype=np.float64)
    expected = (particle_count(), len(GOAL_TIP_Z_M), len(ACTION_ROOT_Z_M))
    if loss.shape != expected or not np.isfinite(loss).all():
        raise ValueError("complete particle task-loss table required")
    params = particle_parameters()
    fixed = np.argmin(loss.mean(axis=0), axis=1)
    oracle = np.argmin(loss, axis=2)
    fixed_loss = loss[:, np.arange(len(GOAL_TIP_Z_M)), fixed]
    oracle_loss = np.min(loss, axis=2)
    gain = _relative_gain(float(fixed_loss.mean()), float(oracle_loss.mean()))
    distinct_by_goal: list[int] = []
    twist_disagreement: list[float] = []
    for goal in range(len(GOAL_TIP_Z_M)):
        by_bending: list[int] = []
        disagreement = 0
        comparisons = 0
        for bend in range(len(BENDING_SCALES)):
            mask = params["bending_index"] == bend
            actions = oracle[mask, goal]
            values, counts = np.unique(actions, return_counts=True)
            by_bending.append(int(values[np.argmax(counts)]))
            disagreement += int(len(actions) - counts.max())
            comparisons += len(actions)
        distinct_by_goal.append(len(set(by_bending)))
        twist_disagreement.append(disagreement / comparisons)
    checks = {
        "bending_changes_oracle_action": min(distinct_by_goal) >= 2,
        "oracle_gain_over_best_fixed": gain >= 0.08,
        "twisting_is_not_dominant_task_axis": max(twist_disagreement) <= 0.34,
    }
    return {
        "goal_conditioned_best_fixed_action": fixed.tolist(),
        "oracle_actions": oracle.tolist(),
        "distinct_bending_conditioned_oracle_actions_per_goal": distinct_by_goal,
        "twisting_only_oracle_disagreement_fraction": twist_disagreement,
        "oracle_relative_gain_over_best_fixed": gain,
        "checks": checks,
        "passed": all(checks.values()),
    }


def noisy_probe_observations(features_by_probe: Array) -> dict[str, Array]:
    value = np.asarray(features_by_probe, dtype=np.float64)
    if (
        value.ndim != 5
        or value.shape[2:] != (len(PROBE_TIMES), len(PROBE_NODES), 3)
        or not np.isfinite(value).all()
    ):
        raise ValueError("complete selected truth-probe features required")
    rng = np.random.default_rng(OBSERVATION_SEED)
    bias = rng.normal(0, SHARED_BIAS_STD_M, (value.shape[1], 1, 1, 3))
    noise = rng.normal(0, NOISE_STD_M, value.shape[1:])
    return {
        "observation": value + bias[None] + noise[None],
        "shared_bias": bias,
        "independent_noise": noise,
    }


def seal_decisions(
    observations: Array,
    probe_indices: Array,
    particle_features: Array,
    particle_losses: Array,
    goal_index: Array,
    task_aware_probe: int,
    mi_probe: int,
) -> dict[str, Array]:
    observed = np.asarray(observations, dtype=np.float64)
    indices = np.asarray(probe_indices)
    features = np.asarray(particle_features, dtype=np.float64)
    losses = np.asarray(particle_losses, dtype=np.float64)
    goals = np.asarray(goal_index)
    required = np.asarray([NULL_PROBE_INDEX, FIXED_PROBE_INDEX, mi_probe, task_aware_probe])
    count = observed.shape[1] if observed.ndim == 5 else -1
    if (
        observed.shape != (4, count, len(PROBE_TIMES), len(PROBE_NODES), 3)
        or not np.array_equal(indices, required)
        or features.shape
        != (len(PROBE_NAMES), particle_count(), len(PROBE_TIMES), len(PROBE_NODES), 3)
        or losses.shape != (particle_count(), len(GOAL_TIP_Z_M), len(ACTION_ROOT_Z_M))
        or goals.shape != (count,)
        or goals.dtype.kind not in "iu"
        or np.any((goals < 0) | (goals >= len(GOAL_TIP_Z_M)))
        or task_aware_probe in (NULL_PROBE_INDEX, mi_probe)
        or mi_probe == NULL_PROBE_INDEX
        or not np.isfinite(observed).all()
    ):
        raise ValueError("registered task-aware decision inputs required")
    fixed = np.argmin(losses.mean(axis=0), axis=1)
    decisions: dict[str, Array] = {
        name: np.zeros(count, dtype=np.int64) for name in ARMS[:-1]
    }
    weights: Array = np.zeros((4, count, particle_count()), dtype=np.float64)
    expected_advantage: Array = np.zeros(count, dtype=np.float64)
    for episode in range(count):
        goal = int(goals[episode])
        for row, probe in enumerate(required):
            weights[row, episode] = posterior_weights(observed[row, episode], features[probe])
        decisions["best_fixed"][episode] = fixed[goal]
        decisions["null_bayes"][episode] = int(np.argmin(weights[0, episode] @ losses[:, goal]))
        decisions["fixed_probe_bayes"][episode] = int(
            np.argmin(weights[1, episode] @ losses[:, goal])
        )
        decisions["mi_probe_bayes"][episode] = int(
            np.argmin(weights[2, episode] @ losses[:, goal])
        )
        task_weight = weights[3, episode]
        task_action = int(np.argmin(task_weight @ losses[:, goal]))
        map_world = int(np.argmax(task_weight))
        decisions["task_aware_map"][episode] = int(np.argmin(losses[map_world, goal]))
        decisions["task_aware_bayes"][episode] = task_action
        fixed_expected = float(task_weight @ losses[:, goal, fixed[goal]])
        task_expected = float(task_weight @ losses[:, goal, task_action])
        expected_advantage[episode] = _relative_gain(fixed_expected, task_expected)
        decisions["task_aware_guarded"][episode] = (
            task_action
            if expected_advantage[episode] >= GUARD_MIN_EXPECTED_RELATIVE_GAIN
            else fixed[goal]
        )
    return {
        **decisions,
        "posterior_weights": weights,
        "task_aware_expected_relative_advantage": expected_advantage,
    }


def _bootstrap_difference(primary: Array, control: Array) -> list[float]:
    if primary.shape != control.shape or primary.ndim != 1:
        raise ValueError("paired one-dimensional losses required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(primary), (10000, len(primary)))
    difference = (primary[samples] - control[samples]).mean(axis=1)
    bounds = np.asarray(np.quantile(difference, [0.025, 0.975])).reshape(-1)
    return [float(bounds[0]), float(bounds[1])]


def score_source(decisions: dict[str, Array], truth_losses: Array) -> dict[str, Any]:
    losses = np.asarray(truth_losses, dtype=np.float64)
    if losses.shape != (TRUTH_COUNT, len(ACTION_ROOT_Z_M)) or not np.isfinite(losses).all():
        raise ValueError("complete truth task-loss table required")
    realized: dict[str, Array] = {}
    for arm in ARMS[:-1]:
        choice = np.asarray(decisions[arm])
        if choice.shape != (TRUTH_COUNT,) or choice.dtype.kind not in "iu" or np.any(
            (choice < 0) | (choice >= len(ACTION_ROOT_Z_M))
        ):
            raise ValueError(f"complete sealed decisions required for {arm}")
        realized[arm] = losses[np.arange(TRUTH_COUNT), choice]
    realized["oracle"] = losses.min(axis=1)
    mean = {arm: float(value.mean()) for arm, value in realized.items()}
    primary = realized["task_aware_bayes"]
    controls = ("best_fixed", "null_bayes", "fixed_probe_bayes", "mi_probe_bayes")
    gain = {arm: _relative_gain(mean[arm], mean["task_aware_bayes"]) for arm in controls}
    intervals = {arm: _bootstrap_difference(primary, realized[arm]) for arm in controls}
    mi_choice = np.asarray(decisions["mi_probe_bayes"])
    task_choice = np.asarray(decisions["task_aware_bayes"])
    differs = int(np.count_nonzero(mi_choice != task_choice))
    harm = float(np.mean(primary > realized["mi_probe_bayes"]))
    checks = {
        "complete_truth_episodes": len(primary) == TRUTH_COUNT,
        "gain_over_best_fixed": gain["best_fixed"] >= 0.03,
        "gain_over_null_bayes": gain["null_bayes"] >= 0.01,
        "gain_over_generic_mi": gain["mi_probe_bayes"] >= 0.005,
        "gain_over_fixed_probe": gain["fixed_probe_bayes"] >= 0.005,
        "bootstrap_upper_below_zero_for_all_controls": all(
            intervals[arm][1] < 0 for arm in controls
        ),
        "at_least_twelve_action_differences_vs_mi": differs >= 12,
        "harm_fraction_vs_mi_at_most_quarter": harm <= 0.25,
    }
    return {
        "mean_task_loss": mean,
        "task_aware_relative_gain": gain,
        "paired_loss_difference_95_interval": intervals,
        "task_aware_vs_mi_action_differences": differs,
        "task_aware_harm_fraction_vs_mi": harm,
        "checks": checks,
        "source_gate_passed": all(checks.values()),
        "truth_loss_sha256": array_digest(losses),
    }
