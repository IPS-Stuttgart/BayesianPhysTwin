"""Frozen, source-only decision-value screen for the native wiring-post task."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .deform_state_restart import array_digest
from .dlolab_benchmark import RIGID_FIELDS
from .dlolab_native import STATE_FIELDS

NATIVE_STEPS = 1800
PREFIX_STEPS = 600
FRAMES = (199, 399, 599)
NODES = (6, 12, 18, 24, 29)
WORLD_ORDER = (4, 4, 4, 0, 1, 2, 3, 5, 6, 7, 8)
ACTION_NAMES = (
    "hold_after_prefix",
    "nominal_route",
    "endpoint_x_minus_50mm",
    "endpoint_x_plus_50mm",
    "endpoint_y_minus_50mm",
    "endpoint_y_plus_50mm",
    "overshoot_then_return",
    "nominal_duplicate",
)
POSITION_BUDGET_M = 0.001
REWARD_BUDGET = 0.001
PAIR_MARGIN = 2 * REWARD_BUDGET
TRACE_SHAPES = {
    "rod_pos_m": (NATIVE_STEPS, 8, 30, 3),
    "rod_vel_m_s": (NATIVE_STEPS, 8, 30, 3),
    "post_pos_m": (NATIVE_STEPS, 8, 2, 3),
    "hidden_post_pos_m": (NATIVE_STEPS, 8, 2, 3, 3),
    "gripper_pos_m": (NATIVE_STEPS, 8, 3),
    "robot_qpos": (NATIVE_STEPS, 8, 9),
}
POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m", "hidden_post_pos_m")
MEMORY_NAMES = tuple(f"memory_RigidSolverState.{key}" for key in RIGID_FIELDS) + tuple(
    f"memory_RODSolverState.{key}" for key in STATE_FIELDS
)


def worlds() -> list[dict[str, int | float]]:
    return [
        {"index": i, "bending_E": e, "twisting_G": g}
        for i, (e, g) in enumerate(itertools.product((1e3, 1e4, 1e5), (1e2, 1e3, 1e4)))
    ]


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(len(WORLD_ORDER)):
        raise ValueError("unregistered wiring task")
    world_index = WORLD_ORDER[index]
    return {
        "index": index,
        "name": f"batch-{index:02d}-material-{world_index}",
        "world": worlds()[world_index],
        "qualification_repeat": index in (1, 2),
    }


def action_bank() -> np.ndarray:
    # Waypoints are set from public scene/goal geometry, never from rollouts.
    xy = np.asarray(
        [
            [0.36, 0.05],
            [0.36, 0.10],
            [0.35, 0.18],
            [0.27, 0.22],
            [0.18, 0.225],
            [0.095, 0.215],
            [0.03, 0.25],
            [0.025, 0.287],
            [0.02, 0.324],
            [0.02, 0.324],
        ],
        dtype=np.float64,
    )
    paths = np.repeat(xy[None], 8, axis=0)
    paths[0, 4:] = xy[3]
    for action, offset in (
        (2, [-0.05, 0.0]),
        (3, [0.05, 0.0]),
        (4, [0.0, -0.05]),
        (5, [0.0, 0.05]),
    ):
        endpoint = xy[8] + np.asarray(offset)
        paths[action, 7] = (xy[6] + endpoint) / 2
        paths[action, 8:] = endpoint
    paths[6, 8] += [-0.025, 0.025]
    controls = np.zeros((8, 9, 6), dtype=np.float64)
    controls[..., :2] = np.diff(paths, axis=1)
    if (
        np.linalg.norm(controls[..., :3], axis=-1).max() > 0.1
        or not np.all(controls[:, :3] == controls[1, :3])
        or not np.array_equal(controls[1], controls[7])
    ):
        raise ValueError("registered action geometry changed")
    return controls


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-wiring-belief-source-v1",
        "role": "finite_simulator_source_screen_not_confirmatory_evidence",
        "native_environment": "envs.env_wiring_post.Train_Env_Wiring_post",
        "native_reward_controller_scene_solver_unchanged": True,
        "only_native_material_randomization_hooks_overridden": ["bending", "twisting"],
        "stretching_parameter_varied": False,
        "native_inextensible_setting_preserved": True,
        "task_goal": "unchanged_public_wiring_post_finalpos.npy",
        "action_names": list(ACTION_NAMES),
        "controls_array_sha256": array_digest(action_bank()),
        "unique_actions": 7,
        "native_batches": 11,
        "native_trajectories": 88,
        "tasks": [task(i) for i in range(11)],
        "n_envs": 8,
        "native_steps": NATIVE_STEPS,
        "native_steps_per_macro": 200,
        "micro_controls_per_macro": 10,
        "branch_native_step": PREFIX_STEPS,
        "prefix_frames_zero_based": list(FRAMES),
        "observed_material_identities": list(NODES),
        "observation_units": "world_frame_metres",
        "simulated_observation_noise_not_measured_sensor_calibration": True,
        "independent_noise_sd_m": 0.002,
        "shared_translation_bias_sd_m": 0.005,
        "source_prior": [1 / 9] * 9,
        "noise_draws_per_world": 8192,
        "noise_seed": 260928,
        "primary_reward": "native_final_reward_not_cumulative_reward",
        "qualification": {
            "fresh_process_per_batch": True,
            "first_three_batches_nominal_repeat": True,
            "all_native_memory_and_observables_finite": True,
            "maximum_segment_relative_error": 0.1,
            "minimum_rod_height_m": -0.01,
            "maximum_attachment_distance_m": 0.01,
            "fixed_post_error_m": 1e-9,
            "all_action_prefix_error_m": 1e-5,
            "duplicate_and_repeat_position_budget_m": POSITION_BUDGET_M,
            "duplicate_and_repeat_reward_budget": REWARD_BUDGET,
            "native_final_reward_reconstruction_atol": 1e-7,
            "native_cumulative_reward_float32_exact": True,
            "numerical_budget_is_not_population_bound": True,
        },
        "source_gates": {
            "minimum_best_fixed_gain_over_prefix_hold": 0.01,
            "minimum_adjusted_oracle_gain": 0.01,
            "minimum_distinct_oracle_actions": 2,
            "minimum_worlds_with_oracle_gain_above_0_01": 3,
            "minimum_adjusted_bayes_gain_over_best_fixed": 0.005,
            "minimum_adjusted_fraction_of_best_fixed_reward_deficit": 0.05,
            "minimum_adjusted_bayes_gain_over_map": 0.002,
            "minimum_adjusted_bayes_gain_over_ignored_bias": 0.0,
            "numeric_pair_margin": PAIR_MARGIN,
        },
        "nominal_value_uses_first_repeat_not_best_or_average": True,
        "tie_break": "lowest_action_or_particle_index",
        "all_worlds_sealed_before_belief_value_analysis": True,
        "fallback": "unchanged_cached_fixed_action_not_bit_exact_native_replay",
        "fresh_method_evaluation_authorized": False,
        "published_controller_comparator_available": False,
        "new_recordings": False,
        "gpu_work": False,
        "protected_data_read": False,
        "earlier_studies_reopened": False,
        "retry_authorized": False,
    }


def native_reward(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    positions, target = np.asarray(positions), np.asarray(target)
    if (
        positions.ndim < 3
        or positions.shape[-2:] != (30, 3)
        or target.shape != (30, 3)
        or not np.isfinite(positions).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("finite native wiring geometry required")
    distance = np.linalg.norm(positions[..., :, None, :] - target, axis=-1)
    chamfer = distance.min(axis=-1).mean(axis=-1) + distance.min(axis=-2).mean(axis=-1)
    height = np.maximum(positions[..., :, 2] - 0.04, 0).mean(axis=-1)
    center = np.maximum(
        np.linalg.norm(positions[..., 15, :2] - [0.198, 0.198], axis=-1) - 0.02,
        0,
    )
    return np.exp(-chamfer - height - 5 * center)


def native_qa(data: dict[str, np.ndarray], native: dict, world: dict) -> dict[str, Any]:
    if set(data) != set(TRACE_SHAPES) | set(MEMORY_NAMES) | {
        "controls",
        "joint_targets",
        "target_pos_m",
    }:
        raise ValueError("complete native trace and memory field set required")
    for key, shape in TRACE_SHAPES.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"native trace changed: {key}")
    if (
        data["controls"].dtype != np.float64
        or array_digest(data["controls"]) != array_digest(action_bank())
        or data["joint_targets"].shape != (8, 91, 9)
        or data["target_pos_m"].shape != (30, 3)
        or any(not np.isfinite(value).all() for value in data.values())
        or native["native_steps"] != NATIVE_STEPS
        or native["world"] != world
    ):
        raise ValueError("native execution/parameter/array contract changed")
    for kind, key in (("bending", "bending_E"), ("twisting", "twisting_G")):
        if native["world_realization"][kind] != [world[key]] * 8:
            raise ValueError("native material binding changed")
    final = native_reward(data["rod_pos_m"][-1], data["target_pos_m"])
    cumulative = np.zeros(8, dtype=np.float32)
    for frame in range(19, NATIVE_STEPS, 20):
        cumulative += native_reward(
            data["rod_pos_m"][frame], data["target_pos_m"]
        ).astype(np.float32)
    prefix = max(
        float(np.max(np.abs(data[k][:PREFIX_STEPS] - data[k][:PREFIX_STEPS, 1:2])))
        for k in POSITION_FIELDS
    )
    duplicate = max(
        float(np.max(np.abs(data[k][:, 1] - data[k][:, 7]))) for k in POSITION_FIELDS
    )
    posts = np.asarray([[0.28, 0.14, 0.02], [0.1, 0.275, 0.02]])
    hidden = np.asarray(
        [
            [[0.28, 0.14, z] for z in (-0.02, 0.0, 0.02)],
            [[0.1, 0.275, z] for z in (-0.02, 0.0, 0.02)],
        ]
    )
    fixed = max(
        float(np.max(np.abs(data["post_pos_m"] - posts))),
        float(np.max(np.abs(data["hidden_post_pos_m"] - hidden))),
    )
    length_error = float(
        np.max(
            np.abs(
                np.linalg.norm(np.diff(data["rod_pos_m"], axis=2), axis=-1) / 0.02 - 1
            )
        )
    )
    attachment = float(
        np.linalg.norm(
            data["rod_pos_m"][:, :, 3] - data["gripper_pos_m"], axis=-1
        ).max()
    )
    final_error = float(np.max(np.abs(final - native["native_final_reward"])))
    checks = {
        "native_final_reward": final_error <= 1e-7,
        "native_cumulative_reward": np.array_equal(
            cumulative, native["native_cumulative_reward"]
        ),
        "ordinary_native_success": bool(
            np.all(np.asarray(native["native_final_reward"]) > 0)
        ),
        "common_prefix": prefix <= 1e-5,
        "duplicate_positions": duplicate <= POSITION_BUDGET_M,
        "duplicate_rewards": abs(final[1] - final[7]) <= REWARD_BUDGET,
        "fixed_posts": fixed <= 1e-9,
        "segment_length": length_error <= 0.1,
        "above_floor": float(data["rod_pos_m"][..., 2].min()) >= -0.01,
        "attached_material_point": attachment <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "maximum_prefix_error_m": prefix,
        "maximum_duplicate_coordinate_error_m": duplicate,
        "fixed_post_error_m": fixed,
        "maximum_segment_relative_error": length_error,
        "maximum_attachment_distance_m": attachment,
        "final_reward_reconstruction_error": final_error,
        "final_rewards": final.tolist(),
    }


def repeat_qa(rows: list[dict[str, np.ndarray]], rewards: np.ndarray) -> dict[str, Any]:
    if len(rows) != 3 or rewards.shape != (3, 8) or not np.isfinite(rewards).all():
        raise ValueError("three complete nominal repetitions required")
    coordinate_span = max(
        float(np.ptp(np.stack([row[k] for row in rows]), axis=0).max())
        for k in POSITION_FIELDS
    )
    reward_span = float(np.ptp(rewards, axis=0).max())
    return {
        "maximum_coordinate_span_m": coordinate_span,
        "maximum_same_action_reward_span": reward_span,
        "passed": coordinate_span <= POSITION_BUDGET_M and reward_span <= REWARD_BUDGET,
        "population_bound_claimed": False,
    }


def prefix_observation(positions: np.ndarray) -> np.ndarray:
    if positions.shape != (PREFIX_STEPS, 8, 30, 3) or not np.isfinite(positions).all():
        raise ValueError("only the complete registered prefix may be observed")
    return positions[list(FRAMES), 1][:, list(NODES)].copy()


def information_value(prefix: np.ndarray, rewards: np.ndarray) -> dict[str, Any]:
    if (
        prefix.shape != (9, 3, 5, 3)
        or rewards.shape != (9, 7)
        or not np.isfinite(prefix).all()
        or not np.isfinite(rewards).all()
        or np.any((rewards <= 0) | (rewards > 1))
    ):
        raise ValueError("complete finite nine-world source bank required")
    count, draws = 15, 8192
    rng = np.random.default_rng(260928)
    noise = rng.normal(0, 0.005, (draws, 1, 3)) + rng.normal(
        0, 0.002, (draws, count, 3)
    )
    signal = prefix.reshape(9, count, 3) - prefix[4].reshape(1, count, 3)
    chol = np.linalg.cholesky(
        0.002**2 * np.eye(count) + 0.005**2 * np.ones((count, count))
    )
    versions = (
        (
            np.linalg.solve(chol, signal).reshape(9, -1),
            np.linalg.solve(chol, noise).reshape(draws, -1),
        ),
        (signal.reshape(9, -1) / 0.002, noise.reshape(draws, -1) / 0.002),
    )
    realized = np.zeros((draws, 3))
    per_world = np.zeros((9, 3))
    selection = np.zeros((3, 7))
    for world in range(9):
        for start in range(0, draws, 256):
            decisions = []
            for white, error in versions:
                delta = white[world] + error[start : start + 256, None] - white
                log_weight = -0.5 * np.sum(delta**2, axis=-1)
                weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
                weight /= weight.sum(axis=1, keepdims=True)
                decisions.append(np.argmax(weight @ rewards, axis=1))
                if len(decisions) == 1:
                    decisions.append(
                        np.argmax(rewards[np.argmax(weight, axis=1)], axis=1)
                    )
            selected = np.stack(decisions, axis=1)
            values = rewards[world, selected]
            realized[start : start + 256] += values / 9
            per_world[world] += values.sum(axis=0) / draws
            for arm in range(3):
                selection[arm] += np.bincount(selected[:, arm], minlength=7) / (
                    9 * draws
                )
    fixed_action = int(np.argmax(rewards.mean(axis=0)))
    fixed = float(rewards[:, fixed_action].mean())
    means = realized.mean(axis=0)
    oracle = rewards.max(axis=1)
    adjusted = float(means[0] - fixed - PAIR_MARGIN)
    checks = {
        "best_fixed_beats_prefix_hold": fixed - float(rewards[:, 0].mean()) >= 0.01,
        "adjusted_oracle_headroom": float(oracle.mean()) - fixed - PAIR_MARGIN >= 0.01,
        "distinct_oracle_actions": len(set(np.argmax(rewards, axis=1))) >= 2,
        "at_least_three_worlds_with_useful_oracle_headroom": int(
            np.sum(oracle - rewards[:, fixed_action] > 0.01)
        )
        >= 3,
        "adjusted_bayes_gain_over_best_fixed": adjusted >= 0.005,
        "adjusted_bayes_gain_fraction_of_fixed_deficit": adjusted >= 0.05 * (1 - fixed),
        "adjusted_bayes_gain_over_map": float(means[0] - means[1] - PAIR_MARGIN)
        >= 0.002,
        "adjusted_bayes_gain_over_ignored_bias": float(
            means[0] - means[2] - PAIR_MARGIN
        )
        >= 0,
    }
    names = ("bias_aware_bayes", "bias_aware_map", "ignored_shared_bias")
    return {
        "arms": {
            name: {
                "expected_native_final_reward": float(means[i]),
                "gain_over_best_fixed": float(means[i] - fixed),
                "monte_carlo_standard_error": float(
                    realized[:, i].std(ddof=1) / np.sqrt(draws)
                ),
                "source_world_expected_rewards": per_world[:, i].tolist(),
                "action_probability": selection[i].tolist(),
            }
            for i, name in enumerate(names)
        },
        "best_fixed_action": fixed_action,
        "best_fixed_reward": fixed,
        "nominal_world_best_action": int(np.argmax(rewards[4])),
        "nominal_world_action_expected_reward": float(
            rewards[:, np.argmax(rewards[4])].mean()
        ),
        "prefix_hold_reward": float(rewards[:, 0].mean()),
        "oracle_reward": float(oracle.mean()),
        "oracle_actions": np.argmax(rewards, axis=1).tolist(),
        "adjusted_bayes_gain": adjusted,
        "checks": {key: bool(value) for key, value in checks.items()},
        "source_gate_passed": all(checks.values()),
        "monte_carlo_only_integrates_assumed_sensor_noise": True,
        "source_worlds_are_prior_support_not_independent_evaluation": True,
        "method_promotion_authorized": False,
    }
