"""Frozen active-probe source study for native loop-wrapping decisions."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .deform_state_restart import array_digest
from .dlolab_benchmark import RIGID_FIELDS
from .dlolab_native import STATE_FIELDS
from .dlolab_wrapping_source import (
    ACTION_NAMES,
    POSTS,
    native_reward,
)
from .dlolab_wrapping_source import (
    action_bank as passive_action_bank,
)

N_ENVS = 9
N_ACTIONS = 8
N_PROBES = 4
PROBE_NAMES = (
    "null_hold",
    "symmetric_tension",
    "symmetric_compression",
    "axial_shear",
)
PROBE_SLOT = (0, 1, 2, 3, 0, 1, 2, 3, 0)
PROBE_MACROS = 6
FULL_MACROS = 14
STEPS_PER_MACRO = 200
PROBE_STEPS = PROBE_MACROS * STEPS_PER_MACRO
FULL_STEPS = FULL_MACROS * STEPS_PER_MACRO
OBSERVATION_FRAMES = (399, 599, 799, 999, 1199)
OBSERVED_NODES = (0, 8, 25, 41, 49)
WORLD_ORDER = (4, 4, 4, 0, 1, 2, 3, 5, 6, 7, 8)
POSITION_BUDGET_M = 0.001
REWARD_BUDGET = 0.001
PAIR_MARGIN = 2 * REWARD_BUDGET
PROBE_DRAWS = 4096
DECISION_DRAWS = 8192
NOISE_SEED = 261001
MEMORY_NAMES = tuple(f"memory_RigidSolverState.{key}" for key in RIGID_FIELDS) + tuple(
    f"memory_RODSolverState.{key}" for key in STATE_FIELDS
)
POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m")


def worlds() -> list[dict[str, int | float]]:
    return [
        {"index": i, "stretching_K": k, "bending_E": e}
        for i, (k, e) in enumerate(itertools.product((2e4, 1e5, 5e5), (1e3, 1e4, 1e5)))
    ]


def validate_world(world: dict) -> None:
    if (
        set(world) != {"index", "stretching_K", "bending_E"}
        or type(world["index"]) is not int
        or world["index"] not in range(9)
        or any(
            type(world[key]) not in (int, float)
            for key in ("stretching_K", "bending_E")
        )
        or world != worlds()[world["index"]]
    ):
        raise ValueError("unregistered active-wrapping material world")


def task(stage: str, index: int, probe_index: int | None = None) -> dict[str, Any]:
    if stage not in {"probe", "baseline", "active"}:
        raise ValueError("unregistered active-wrapping stage")
    if type(index) is not int or index not in range(len(WORLD_ORDER)):
        raise ValueError("unregistered active-wrapping task")
    if stage == "probe":
        if probe_index is not None:
            raise ValueError("probe bank has no selected probe")
    elif type(probe_index) is not int or probe_index not in range(N_PROBES):
        raise ValueError("full wrapping task requires a registered probe")
    world_index = WORLD_ORDER[index]
    suffix = "bank" if stage == "probe" else f"probe-{probe_index}"
    return {
        "stage": stage,
        "index": index,
        "name": f"{stage}-batch-{index:02d}-material-{world_index}-{suffix}",
        "world": worlds()[world_index],
        "probe_index": probe_index,
        "qualification_repeat": index in (1, 2),
    }


def _passive_position_paths() -> np.ndarray:
    controls = passive_action_bank()
    if controls.shape != (N_ENVS, 11, 12) or np.any(controls[..., 6:] != 0):
        raise ValueError("frozen wrapping continuation changed")
    flat = np.concatenate(
        [np.zeros((N_ENVS, 1, 6)), np.cumsum(controls[..., :6], axis=1)],
        axis=1,
    )
    return flat.reshape(N_ENVS, 12, 2, 3)


def probe_excursions() -> np.ndarray:
    base = _passive_position_paths()[0, 2]
    result = np.repeat(base[None], N_PROBES, axis=0)
    result[1, 0, 1] += 0.04
    result[1, 1, 1] -= 0.04
    result[2, 0, 1] -= 0.04
    result[2, 1, 1] += 0.04
    result[3, 0, 0] -= 0.04
    result[3, 1, 0] += 0.04
    return result


def _controls_from_paths(paths: np.ndarray) -> np.ndarray:
    if paths.ndim != 4 or paths.shape[-2:] != (2, 3):
        raise ValueError("finite registered gripper paths required")
    controls = np.zeros((paths.shape[0], paths.shape[1] - 1, 12), dtype=np.float64)
    controls[..., :6] = np.diff(paths, axis=1).reshape(paths.shape[0], -1, 6)
    if (
        not np.isfinite(controls).all()
        or np.linalg.norm(
            controls[..., :6].reshape(paths.shape[0], -1, 2, 3), axis=-1
        ).max()
        > 0.1000000001
    ):
        raise ValueError("registered active-wrapping motion budget changed")
    return controls


def probe_bank_controls() -> np.ndarray:
    old = _passive_position_paths()
    paths = np.zeros((N_ENVS, 7, 2, 3), dtype=np.float64)
    for slot, probe in enumerate(PROBE_SLOT):
        paths[slot] = np.stack(
            [
                old[0, 0],
                old[0, 1],
                old[0, 2],
                probe_excursions()[probe],
                old[0, 2],
                old[0, 2],
                old[0, 3],
            ]
        )
    controls = _controls_from_paths(paths)
    if (
        controls.shape != (N_ENVS, PROBE_MACROS, 12)
        or not np.array_equal(controls[0], controls[4])
        or not np.array_equal(controls[0], controls[8])
        or not np.array_equal(controls[1], controls[5])
        or not np.array_equal(controls[2], controls[6])
        or not np.array_equal(controls[3], controls[7])
        or np.unique(controls.reshape(N_ENVS, -1), axis=0).shape[0] != N_PROBES
    ):
        raise ValueError("active probe bank changed")
    return controls


def full_action_controls(probe_index: int) -> np.ndarray:
    if type(probe_index) is not int or probe_index not in range(N_PROBES):
        raise ValueError("registered probe index required")
    old = _passive_position_paths()
    paths = np.zeros((N_ENVS, 15, 2, 3), dtype=np.float64)
    for action in range(N_ENVS):
        paths[action] = np.concatenate(
            [
                np.stack(
                    [
                        old[0, 0],
                        old[0, 1],
                        old[0, 2],
                        probe_excursions()[probe_index],
                        old[0, 2],
                        old[0, 2],
                        old[0, 3],
                    ]
                ),
                old[action, 4:],
            ]
        )
    controls = _controls_from_paths(paths)
    if (
        controls.shape != (N_ENVS, FULL_MACROS, 12)
        or not np.array_equal(controls[1], controls[8])
        or np.unique(controls.reshape(N_ENVS, -1), axis=0).shape[0] != N_ACTIONS
        or not np.all(controls[:, :PROBE_MACROS] == controls[0, :PROBE_MACROS])
    ):
        raise ValueError("active wrapping continuation bank changed")
    return controls


def protocol() -> dict[str, Any]:
    passive_result = "results/sota/dlolab_wrapping_belief_source_v1/result.json"
    return {
        "schema": "dlolab-active-probe-wrapping-source-v1-1",
        "role": "finite_simulator_dual_control_source_screen_not_confirmation",
        "native_environment": "envs.env_wrapping.Train_Env_Wrapping",
        "native_reward_controller_scene_solver_unchanged": True,
        "only_native_material_randomization_hooks_overridden": [
            "bending",
            "stretching",
        ],
        "native_extensible_closed_loop_preserved": True,
        "probe_names": list(PROBE_NAMES),
        "probe_slot_map": list(PROBE_SLOT),
        "probe_controls_sha256": array_digest(probe_bank_controls()),
        "null_probe_index": 0,
        "probe_selection_uses_future_reward": False,
        "probe_selection": "maximum_expected_material_mutual_information",
        "probe_selection_tie_break": "lowest_probe_index",
        "probe_macros": PROBE_MACROS,
        "probe_native_steps": PROBE_STEPS,
        "settle_macros_after_excursion": 2,
        "all_probes_rejoin_same_tool_waypoint_before_continuation": True,
        "continuation_action_names": list(ACTION_NAMES),
        "null_full_controls_sha256": array_digest(full_action_controls(0)),
        "unique_continuation_actions": N_ACTIONS,
        "full_macros": FULL_MACROS,
        "full_native_steps": FULL_STEPS,
        "observation_frames_zero_based": list(OBSERVATION_FRAMES),
        "observed_material_identities": list(OBSERVED_NODES),
        "observation_units": "world_frame_metres",
        "independent_noise_sd_m": 0.002,
        "shared_translation_bias_sd_m": 0.005,
        "simulated_observation_noise_not_measured_sensor_calibration": True,
        "source_prior": [1 / 9] * 9,
        "probe_noise_draws_per_world": PROBE_DRAWS,
        "decision_noise_draws_per_world": DECISION_DRAWS,
        "noise_seed": NOISE_SEED,
        "probe_stage_batches": 11,
        "null_continuation_batches": 11,
        "selected_continuation_batches": 11,
        "native_trajectories_if_complete": 297,
        "first_three_batches_per_stage_nominal_repeat": True,
        "source_worlds": worlds(),
        "qualification": {
            "fresh_process_per_batch": True,
            "segment_length_ratio_range": [0.25, 3.0],
            "minimum_rod_height_m": -0.01,
            "maximum_attachment_distance_m": 0.01,
            "fixed_post_error_m": 1e-9,
            "common_prefix_error_m": 1e-5,
            "common_probe_endpoint_tool_error_m": 1e-5,
            "duplicate_and_repeat_position_budget_m": POSITION_BUDGET_M,
            "duplicate_and_repeat_reward_budget": REWARD_BUDGET,
            "native_final_reward_reconstruction_atol": 1e-7,
            "native_cumulative_reward_float32_exact": True,
            "native_failure_penalties_cannot_count_as_success": True,
        },
        "probe_gates": {
            "selected_probe_must_not_be_null": True,
            "minimum_mutual_information_gain_nats": 0.05,
            "minimum_material_classification_gain": 0.05,
        },
        "decision_gates": {
            "minimum_best_fixed_gain_over_hold": 0.05,
            "minimum_adjusted_active_oracle_headroom": 0.02,
            "minimum_distinct_active_oracle_actions": 2,
            "minimum_active_bayes_gain_over_fixed": 0.02,
            "minimum_active_bayes_gain_fraction_of_fixed_deficit": 0.05,
            "active_bayes_map_noninferiority_margin": PAIR_MARGIN,
            "minimum_active_bayes_gain_over_null_bayes": 0.005,
            "minimum_gain_difference_over_null": 0.005,
            "minimum_worlds_active_bayes_better_than_null": 5,
            "numeric_pair_margin": PAIR_MARGIN,
        },
        "primary_reward": "unchanged_native_final_winding_reward",
        "passive_comparator_path": passive_result,
        "old_wrapping_outcomes_not_used_for_probe_selection": True,
        "retained_zero_trajectory_parent_failure": {
            "original_receipt": "results/sota/dlolab_active_probe_wrapping_prelock_v1/prelock-failure.json",
            "correction_receipt": "results/sota/dlolab_active_probe_wrapping_prelock_v1/prelock-failure-correction.json",
            "native_trajectories": 0,
            "scientific_payload_generated": False,
        },
        "all_prefix_banks_sealed_before_probe_selection": True,
        "both_full_banks_sealed_before_decision_analysis": True,
        "fallback": "unchanged_cached_passive_wrapping_and_deform_results",
        "automatic_promotion": False,
        "fresh_evaluation_authorized": False,
        "published_controller_comparator_available": False,
        "new_recordings": False,
        "gpu_work": False,
        "protected_data_read": False,
        "retry_authorized": False,
    }


def prefix_observation(positions: np.ndarray, stage: str) -> np.ndarray:
    positions = np.asarray(positions)
    steps = PROBE_STEPS if stage == "probe" else FULL_STEPS
    if positions.shape != (steps, N_ENVS, 50, 3) or not np.isfinite(positions).all():
        raise ValueError("complete registered active-wrapping trace required")
    sampled = positions[list(OBSERVATION_FRAMES)]
    if stage == "probe":
        return sampled[:, :N_PROBES, list(OBSERVED_NODES)].transpose(1, 0, 2, 3).copy()
    if stage not in {"baseline", "active"}:
        raise ValueError("unregistered prefix stage")
    return sampled[:, 1, list(OBSERVED_NODES)].copy()


def _whitened_signal(prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(OBSERVATION_FRAMES) * len(OBSERVED_NODES)
    signal = prefix.reshape(9, count, 3) - prefix[4].reshape(1, count, 3)
    covariance = 0.002**2 * np.eye(count) + 0.005**2 * np.ones((count, count))
    chol = np.linalg.cholesky(covariance)
    return np.linalg.solve(chol, signal).reshape(9, -1), chol


def probe_information(prefix: np.ndarray) -> dict[str, Any]:
    if prefix.shape != (9, N_PROBES, 5, 5, 3) or not np.isfinite(prefix).all():
        raise ValueError("complete reward-blind probe prefix bank required")
    rng = np.random.default_rng(NOISE_SEED)
    shared = rng.normal(0, 0.005, (PROBE_DRAWS, 1, 3))
    independent = rng.normal(0, 0.002, (N_PROBES, PROBE_DRAWS, 25, 3))
    entropy = np.zeros(N_PROBES)
    correct = np.zeros(N_PROBES)
    for probe in range(N_PROBES):
        white, chol = _whitened_signal(prefix[:, probe])
        error = np.linalg.solve(chol, shared + independent[probe]).reshape(
            PROBE_DRAWS, -1
        )
        for world in range(9):
            for start in range(0, PROBE_DRAWS, 256):
                delta = white[world] + error[start : start + 256, None] - white
                log_weight = -0.5 * np.sum(delta**2, axis=-1)
                weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
                weight /= weight.sum(axis=1, keepdims=True)
                entropy[probe] -= np.sum(
                    weight * np.log(np.maximum(weight, 1e-300))
                ) / (9 * PROBE_DRAWS)
                correct[probe] += np.sum(np.argmax(weight, axis=1) == world) / (
                    9 * PROBE_DRAWS
                )
    information = np.log(9.0) - entropy
    selected = int(np.argmax(information))
    checks = {
        "selected_probe_is_not_null": selected != 0,
        "mutual_information_gain": float(information[selected] - information[0])
        >= 0.05,
        "material_classification_gain": float(correct[selected] - correct[0]) >= 0.05,
    }
    return {
        "selected_probe_index": selected,
        "selected_probe_name": PROBE_NAMES[selected],
        "expected_posterior_entropy_nats": entropy.tolist(),
        "mutual_information_nats": information.tolist(),
        "material_classification_accuracy": correct.tolist(),
        "mutual_information_gain_over_null_nats": float(
            information[selected] - information[0]
        ),
        "classification_gain_over_null": float(correct[selected] - correct[0]),
        "checks": checks,
        "passed": all(checks.values()),
        "future_reward_used": False,
    }


def decision_value(
    prefix: np.ndarray, rewards: np.ndarray, seed_offset: int
) -> dict[str, Any]:
    if (
        prefix.shape != (9, 5, 5, 3)
        or rewards.shape != (9, N_ACTIONS)
        or not np.isfinite(prefix).all()
        or not np.isfinite(rewards).all()
        or np.any((rewards <= -98) | (rewards > 1))
    ):
        raise ValueError("complete ordinary active-wrapping source bank required")
    rng = np.random.default_rng(NOISE_SEED + seed_offset)
    noise = rng.normal(0, 0.005, (DECISION_DRAWS, 1, 3)) + rng.normal(
        0, 0.002, (DECISION_DRAWS, 25, 3)
    )
    white, chol = _whitened_signal(prefix)
    error_aware = np.linalg.solve(chol, noise).reshape(DECISION_DRAWS, -1)
    signal_ignored = (prefix.reshape(9, -1) - prefix[4].reshape(1, -1)) / 0.002
    error_ignored = noise.reshape(DECISION_DRAWS, -1) / 0.002
    realized = np.zeros((DECISION_DRAWS, 3))
    per_world = np.zeros((9, 3))
    selection = np.zeros((3, N_ACTIONS))
    for world in range(9):
        for start in range(0, DECISION_DRAWS, 256):
            decisions: list[np.ndarray] = []
            delta = white[world] + error_aware[start : start + 256, None] - white
            log_weight = -0.5 * np.sum(delta**2, axis=-1)
            weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
            weight /= weight.sum(axis=1, keepdims=True)
            decisions.append(np.argmax(weight @ rewards, axis=1))
            decisions.append(np.argmax(rewards[np.argmax(weight, axis=1)], axis=1))
            delta = (
                signal_ignored[world]
                + error_ignored[start : start + 256, None]
                - signal_ignored
            )
            log_weight = -0.5 * np.sum(delta**2, axis=-1)
            ignored = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
            ignored /= ignored.sum(axis=1, keepdims=True)
            decisions.append(np.argmax(ignored @ rewards, axis=1))
            selected = np.stack(decisions, axis=1)
            values = rewards[world, selected]
            realized[start : start + 256] += values / 9
            per_world[world] += values.sum(axis=0) / DECISION_DRAWS
            for arm in range(3):
                selection[arm] += np.bincount(selected[:, arm], minlength=N_ACTIONS) / (
                    9 * DECISION_DRAWS
                )
    fixed_action = int(np.argmax(rewards.mean(axis=0)))
    fixed = float(rewards[:, fixed_action].mean())
    means = realized.mean(axis=0)
    oracle = rewards.max(axis=1)
    names = ("bias_aware_bayes", "bias_aware_map", "ignored_shared_bias")
    return {
        "arms": {
            name: {
                "expected_native_final_reward": float(means[i]),
                "gain_over_best_fixed": float(means[i] - fixed),
                "monte_carlo_standard_error": float(
                    realized[:, i].std(ddof=1) / np.sqrt(DECISION_DRAWS)
                ),
                "source_world_expected_rewards": per_world[:, i].tolist(),
                "action_probability": selection[i].tolist(),
            }
            for i, name in enumerate(names)
        },
        "best_fixed_action": fixed_action,
        "best_fixed_reward": fixed,
        "prefix_hold_reward": float(rewards[:, 0].mean()),
        "oracle_reward": float(oracle.mean()),
        "oracle_actions": np.argmax(rewards, axis=1).tolist(),
        "source_worlds_are_prior_support_not_independent_evaluation": True,
        "monte_carlo_only_integrates_assumed_sensor_noise": True,
    }


def active_decision_gate(active: dict, baseline: dict, passive: dict) -> dict[str, Any]:
    for row in (active, baseline):
        if not isinstance(row, dict) or "arms" not in row:
            raise ValueError("complete matched active/null decision summaries required")
    active_bayes = active["arms"]["bias_aware_bayes"]["expected_native_final_reward"]
    active_map = active["arms"]["bias_aware_map"]["expected_native_final_reward"]
    null_bayes = baseline["arms"]["bias_aware_bayes"]["expected_native_final_reward"]
    active_gain = active_bayes - active["best_fixed_reward"]
    null_gain = null_bayes - baseline["best_fixed_reward"]
    active_world = np.asarray(
        active["arms"]["bias_aware_bayes"]["source_world_expected_rewards"]
    )
    null_world = np.asarray(
        baseline["arms"]["bias_aware_bayes"]["source_world_expected_rewards"]
    )
    passive_bayes = passive["metrics"]["arms"]["bias_aware_bayes"][
        "expected_native_final_reward"
    ]
    checks = {
        "active_best_fixed_beats_hold": active["best_fixed_reward"]
        - active["prefix_hold_reward"]
        >= 0.05,
        "active_adjusted_oracle_headroom": active["oracle_reward"]
        - active["best_fixed_reward"]
        - PAIR_MARGIN
        >= 0.02,
        "active_distinct_oracle_actions": len(set(active["oracle_actions"])) >= 2,
        "active_adjusted_gain_over_fixed": active_gain - PAIR_MARGIN >= 0.02,
        "active_gain_fraction_of_fixed_deficit": active_gain - PAIR_MARGIN
        >= 0.05 * (1 - active["best_fixed_reward"]),
        "active_bayes_map_noninferiority": active_bayes >= active_map - PAIR_MARGIN,
        "active_bayes_gain_over_null": active_bayes - null_bayes >= 0.005,
        "gain_difference_over_null": active_gain - null_gain >= 0.005,
        "five_worlds_active_better_than_null": int(np.sum(active_world > null_world))
        >= 5,
    }
    return {
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": all(checks.values()),
        "active_bayes_reward": float(active_bayes),
        "null_bayes_reward": float(null_bayes),
        "passive_previous_bayes_reward": float(passive_bayes),
        "active_gain_over_fixed": float(active_gain),
        "null_gain_over_fixed": float(null_gain),
        "active_bayes_gain_over_null": float(active_bayes - null_bayes),
        "gain_difference_over_null": float(active_gain - null_gain),
        "worlds_active_better_than_null": int(np.sum(active_world > null_world)),
        "automatic_promotion": False,
        "fresh_evaluation_authorized": False,
    }


def native_qa(
    data: dict[str, np.ndarray],
    native: dict,
    world: dict,
    stage: str,
    probe_index: int | None,
) -> dict[str, Any]:
    validate_world(world)
    if stage == "probe":
        if probe_index is not None:
            raise ValueError("probe-bank qualification cannot select a probe")
        controls, steps = probe_bank_controls(), PROBE_STEPS
    elif stage in {"baseline", "active"}:
        if type(probe_index) is not int:
            raise ValueError("full qualification requires a registered probe")
        controls, steps = full_action_controls(probe_index), FULL_STEPS
    else:
        raise ValueError("unregistered active-wrapping qualification stage")
    if set(data) != set(MEMORY_NAMES) | {
        "rod_pos_m",
        "rod_vel_m_s",
        "post_pos_m",
        "gripper_pos_m",
        "robot_qpos",
        "controls",
        "joint_targets",
        "initial_rod_pos_m",
    }:
        raise ValueError("complete native active-wrapping field set required")
    shapes = {
        "rod_pos_m": (steps, N_ENVS, 50, 3),
        "rod_vel_m_s": (steps, N_ENVS, 50, 3),
        "post_pos_m": (steps, N_ENVS, 3, 3),
        "gripper_pos_m": (steps, N_ENVS, 2, 3),
        "robot_qpos": (steps, N_ENVS, 18),
    }
    if any(
        data[key].shape != shape or not np.isfinite(data[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("native active-wrapping trace changed")
    if (
        data["controls"].dtype != np.float64
        or array_digest(data["controls"]) != array_digest(controls)
        or data["joint_targets"].shape != (N_ENVS, controls.shape[1] * 10 + 1, 18)
        or data["initial_rod_pos_m"].shape != (N_ENVS, 50, 3)
        or any(not np.isfinite(value).all() for value in data.values())
        or native["native_steps"] != steps
        or native["world"] != world
        or native["stage"] != stage
        or native["probe_index"] != probe_index
        or native["device"] != "cpu"
        or native["twisting_stiffness_zero_preserved"] is not True
        or native["runtime_camera_rendered"] is not False
        or native["native_source_modified"] is not False
        or any(
            native["world_realization"][kind] != [world[key]] * N_ENVS
            for kind, key in (
                ("bending", "bending_E"),
                ("stretching", "stretching_K"),
            )
        )
        or any(
            np.asarray(native[key]).shape != (N_ENVS,)
            or not np.isfinite(native[key]).all()
            for key in ("native_final_reward", "native_cumulative_reward")
        )
    ):
        raise ValueError("native active-wrapping execution contract changed")
    final = native_reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    cumulative = np.zeros(N_ENVS, dtype=np.float32)
    for frame in range(19, steps, 20):
        cumulative += native_reward(
            data["rod_pos_m"][frame], data["post_pos_m"][frame]
        ).astype(np.float32) + np.float32(1)
    initial = data["initial_rod_pos_m"]
    rest = np.linalg.norm(np.roll(initial, -1, axis=1) - initial, axis=-1)
    ratios = (
        np.linalg.norm(
            np.roll(data["rod_pos_m"], -1, axis=2) - data["rod_pos_m"], axis=-1
        )
        / rest
    )
    attachment = float(
        np.linalg.norm(
            data["rod_pos_m"][:, :, [17, 33]] - data["gripper_pos_m"], axis=-1
        ).max()
    )
    fixed = float(np.abs(data["post_pos_m"] - POSTS).max())
    endpoint_frame = -1 if stage == "probe" else PROBE_STEPS - 1
    endpoint = float(np.ptp(data["gripper_pos_m"][endpoint_frame], axis=0).max())
    if stage == "probe":
        groups = ((0, 4, 8), (1, 5), (2, 6), (3, 7))
        duplicate = max(
            float(
                np.abs(
                    data[key][:, list(group)] - data[key][:, group[0] : group[0] + 1]
                ).max()
            )
            for key in POSITION_FIELDS
            for group in groups
        )
        common_prefix = 0.0
    else:
        duplicate = max(
            float(np.abs(data[key][:, 1] - data[key][:, 8]).max())
            for key in POSITION_FIELDS
        )
        common_prefix = max(
            float(np.abs(data[key][:PROBE_STEPS] - data[key][:PROBE_STEPS, 1:2]).max())
            for key in POSITION_FIELDS
        )
    final_error = float(np.abs(final - np.asarray(native["native_final_reward"])).max())
    checks = {
        "native_final_reward": final_error <= 1e-7,
        "native_cumulative_reward": np.array_equal(
            cumulative, np.asarray(native["native_cumulative_reward"], dtype=np.float32)
        ),
        "ordinary_native_success": bool(
            np.all(np.asarray(native["native_final_reward"]) > -98)
        ),
        "common_full_prefix": common_prefix <= 1e-5,
        "common_probe_endpoint_tools": endpoint <= 1e-5,
        "duplicate_positions": duplicate <= POSITION_BUDGET_M,
        "fixed_posts": fixed <= 1e-9,
        "finite_extensible_segments": bool(ratios.min() >= 0.25 and ratios.max() <= 3),
        "above_floor": float(data["rod_pos_m"][..., 2].min()) >= -0.01,
        "attached_material_points": attachment <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "maximum_common_prefix_error_m": common_prefix,
        "maximum_common_probe_endpoint_tool_error_m": endpoint,
        "maximum_duplicate_coordinate_error_m": duplicate,
        "fixed_post_error_m": fixed,
        "segment_length_ratio_range": [float(ratios.min()), float(ratios.max())],
        "maximum_attachment_distance_m": attachment,
        "final_reward_reconstruction_error": final_error,
        "final_rewards": final.tolist(),
    }


def repeat_qa(rows: list[dict[str, np.ndarray]], rewards: np.ndarray) -> dict[str, Any]:
    if len(rows) != 3 or rewards.shape != (3, N_ENVS) or not np.isfinite(rewards).all():
        raise ValueError("three complete nominal repetitions required")
    span = max(
        float(np.ptp(np.stack([row[key] for row in rows]), axis=0).max())
        for key in POSITION_FIELDS
    )
    reward_span = float(np.ptp(rewards, axis=0).max())
    return {
        "maximum_coordinate_span_m": span,
        "maximum_same_action_reward_span": reward_span,
        "passed": span <= POSITION_BUDGET_M and reward_span <= REWARD_BUDGET,
        "population_bound_claimed": False,
    }
