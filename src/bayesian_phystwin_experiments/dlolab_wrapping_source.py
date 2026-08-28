"""Frozen source screen for material-dependent native loop-wrapping decisions."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .deform_state_restart import array_digest
from .dlolab_benchmark import RIGID_FIELDS
from .dlolab_native import STATE_FIELDS

N_ENVS = 9
N_ACTIONS = 8
NATIVE_STEPS = 2200
PREFIX_STEPS = 600
FRAMES = (199, 399, 599)
NODES = (0, 8, 25, 41, 49)
WORLD_ORDER = (4, 4, 4, 0, 1, 2, 3, 5, 6, 7, 8)
POSTS = np.asarray([[0.45, -0.115, 0.02], [0.45, 0.115, 0.02], [0.25, 0, 0.02]])
ACTION_NAMES = (
    "hold_after_prefix",
    "symmetric_pull_then_lower",
    "lower_early",
    "lower_late",
    "wide_finish",
    "narrow_finish",
    "first_gripper_leads",
    "second_gripper_leads",
    "nominal_duplicate",
)
POSITION_BUDGET_M = 0.001
REWARD_BUDGET = 0.001
PAIR_MARGIN = 2 * REWARD_BUDGET
TRACE_SHAPES = {
    "rod_pos_m": (NATIVE_STEPS, N_ENVS, 50, 3),
    "rod_vel_m_s": (NATIVE_STEPS, N_ENVS, 50, 3),
    "post_pos_m": (NATIVE_STEPS, N_ENVS, 3, 3),
    "gripper_pos_m": (NATIVE_STEPS, N_ENVS, 2, 3),
    "robot_qpos": (NATIVE_STEPS, N_ENVS, 18),
}
POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m")
MEMORY_NAMES = tuple(f"memory_RigidSolverState.{key}" for key in RIGID_FIELDS) + tuple(
    f"memory_RODSolverState.{key}" for key in STATE_FIELDS
)


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
            type(world[k]) not in (int, float) for k in ("stretching_K", "bending_E")
        )
        or world != worlds()[world["index"]]
    ):
        raise ValueError("unregistered wrapping material world")


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(len(WORLD_ORDER)):
        raise ValueError("unregistered wrapping source task")
    world_index = WORLD_ORDER[index]
    return {
        "index": index,
        "name": f"batch-{index:02d}-material-{world_index}",
        "world": worlds()[world_index],
        "qualification_repeat": index in (1, 2),
    }


def action_bank() -> np.ndarray:
    # Relative waypoints use the public three-post geometry, not native outcomes.
    x = np.asarray(
        [0, 0, -0.06, -0.14, -0.22, -0.30, -0.37, -0.37, -0.37, -0.37, -0.37, -0.37]
    )
    spread = np.asarray([0, 0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0])
    height = np.asarray(
        [0, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.03, 0.03, 0.02, 0.02, 0.02]
    )
    path = np.zeros((N_ENVS, 12, 2, 3), dtype=np.float64)
    path[..., 0] = x[None, :, None]
    path[:, :, 0, 1] = spread
    path[:, :, 1, 1] = -spread
    path[..., 2] = height[None, :, None]
    path[0, 4:] = path[0, 3]
    path[2, 5:7, :, 2] = 0.03
    path[3, 7:9, :, 2] = 0.06
    path[4, 8:, 0, 1] = 0.035
    path[4, 8:, 1, 1] = -0.035
    path[5, 8:, 0, 1] = -0.04
    path[5, 8:, 1, 1] = 0.04
    path[6, 4:6, 0, 0] -= 0.02
    path[6, 4:6, 1, 0] += 0.02
    path[7, 4:6, 0, 0] += 0.02
    path[7, 4:6, 1, 0] -= 0.02
    controls = np.zeros((N_ENVS, 11, 12), dtype=np.float64)
    controls[..., :6] = np.diff(path, axis=1).reshape(N_ENVS, 11, 6)
    if (
        np.linalg.norm(controls[..., :6].reshape(N_ENVS, 11, 2, 3), axis=-1).max()
        > 0.1000000001
        or not np.all(controls[:, :3] == controls[1, :3])
        or not np.array_equal(controls[1], controls[8])
        or np.unique(controls.reshape(N_ENVS, -1), axis=0).shape[0] != N_ACTIONS
    ):
        raise ValueError("registered wrapping action geometry changed")
    return controls


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-wrapping-belief-source-v1",
        "role": "finite_simulator_source_screen_not_confirmatory_evidence",
        "native_environment": "envs.env_wrapping.Train_Env_Wrapping",
        "native_reward_controller_scene_solver_unchanged": True,
        "only_native_material_randomization_hooks_overridden": [
            "bending",
            "stretching",
        ],
        "native_extensible_closed_loop_preserved": True,
        "twisting_parameter_varied": False,
        "action_names": list(ACTION_NAMES),
        "controls_array_sha256": array_digest(action_bank()),
        "unique_actions": N_ACTIONS,
        "native_batches": 11,
        "native_trajectories": 99,
        "tasks": [task(i) for i in range(11)],
        "n_envs": N_ENVS,
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
        "noise_seed": 260930,
        "primary_reward": "native_final_winding_reward_not_cumulative_reward",
        "qualification": {
            "fresh_process_per_batch": True,
            "first_three_batches_nominal_repeat": True,
            "all_native_memory_and_observables_finite": True,
            "segment_length_ratio_range": [0.25, 3.0],
            "minimum_rod_height_m": -0.01,
            "maximum_attachment_distance_m": 0.01,
            "fixed_post_error_m": 1e-9,
            "all_action_prefix_error_m": 1e-5,
            "duplicate_and_repeat_position_budget_m": POSITION_BUDGET_M,
            "duplicate_and_repeat_reward_budget": REWARD_BUDGET,
            "native_final_reward_reconstruction_atol": 1e-7,
            "native_cumulative_reward_float32_exact": True,
            "native_failure_penalties_cannot_count_as_ordinary_success": True,
            "numerical_budget_is_not_population_bound": True,
        },
        "source_gates": {
            "minimum_best_fixed_gain_over_prefix_hold": 0.05,
            "minimum_adjusted_oracle_gain": 0.05,
            "minimum_distinct_oracle_actions": 2,
            "minimum_worlds_with_oracle_gain_above_0_05": 3,
            "minimum_adjusted_bayes_gain_over_best_fixed": 0.02,
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


def winding_components(
    positions: np.ndarray, posts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions, posts = np.asarray(positions), np.asarray(posts)
    if (
        positions.ndim < 3
        or positions.shape[-2:] != (50, 3)
        or posts.shape != positions.shape[:-2] + (3, 3)
        or not np.isfinite(positions).all()
        or not np.isfinite(posts).all()
    ):
        raise ValueError("aligned finite native wrapping geometry required")
    relative = positions[..., :, None, :2] - posts[..., None, :, :2]
    following = np.roll(relative, -1, axis=-3)
    cross = relative[..., 0] * following[..., 1] - relative[..., 1] * following[..., 0]
    dot = np.sum(relative * following, axis=-1)
    turns = np.sum(np.arctan2(cross, dot), axis=-2) / (2 * np.pi)
    distances = np.linalg.norm(
        positions[..., :, None, :] - posts[..., None, :, :], axis=-1
    )
    penalties = np.maximum(distances.min(axis=-2) - 0.015, 0)
    return turns, penalties


def native_reward(positions: np.ndarray, posts: np.ndarray) -> np.ndarray:
    turns, penalties = winding_components(positions, posts)
    return 1 - np.mean((np.abs(turns) - 1) ** 2, axis=-1) - penalties.sum(axis=-1)


def native_qa(data: dict[str, np.ndarray], native: dict, world: dict) -> dict[str, Any]:
    validate_world(world)
    if set(data) != set(TRACE_SHAPES) | set(MEMORY_NAMES) | {
        "controls",
        "joint_targets",
        "initial_rod_pos_m",
    }:
        raise ValueError("complete native trace and memory field set required")
    for key, shape in TRACE_SHAPES.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"native trace changed: {key}")
    if (
        data["controls"].dtype != np.float64
        or array_digest(data["controls"]) != array_digest(action_bank())
        or data["joint_targets"].shape != (N_ENVS, 111, 18)
        or data["initial_rod_pos_m"].shape != (N_ENVS, 50, 3)
        or any(not np.isfinite(value).all() for value in data.values())
        or native["native_steps"] != NATIVE_STEPS
        or native["world"] != world
        or native["device"] != "cpu"
        or native["twisting_stiffness_zero_preserved"] is not True
        or native["runtime_camera_rendered"] is not False
        or native["native_source_modified"] is not False
        or any(
            np.asarray(native[k]).shape != (N_ENVS,) or not np.isfinite(native[k]).all()
            for k in ("native_final_reward", "native_cumulative_reward")
        )
    ):
        raise ValueError("native execution/parameter/array contract changed")
    for kind, key in (("bending", "bending_E"), ("stretching", "stretching_K")):
        if native["world_realization"][kind] != [world[key]] * N_ENVS:
            raise ValueError("native material binding changed")
    final = native_reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    cumulative = np.zeros(N_ENVS, dtype=np.float32)
    for frame in range(19, NATIVE_STEPS, 20):
        cumulative += native_reward(
            data["rod_pos_m"][frame], data["post_pos_m"][frame]
        ).astype(np.float32) + np.float32(1)
    prefix = max(
        float(np.abs(data[k][:PREFIX_STEPS] - data[k][:PREFIX_STEPS, 1:2]).max())
        for k in POSITION_FIELDS
    )
    duplicate = max(
        float(np.abs(data[k][:, 1] - data[k][:, 8]).max()) for k in POSITION_FIELDS
    )
    initial = data["initial_rod_pos_m"]
    rest = np.linalg.norm(np.roll(initial, -1, axis=1) - initial, axis=-1)
    if np.any(rest <= 0):
        raise ValueError("initial loop contains collapsed segments")
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
    final_error = float(np.abs(final - native["native_final_reward"]).max())
    checks = {
        "native_final_reward": final_error <= 1e-7,
        "native_cumulative_reward": np.array_equal(
            cumulative, native["native_cumulative_reward"]
        ),
        "ordinary_native_success": bool(
            np.all(np.asarray(native["native_final_reward"]) > -98)
        ),
        "common_prefix": prefix <= 1e-5,
        "duplicate_positions": duplicate <= POSITION_BUDGET_M,
        "duplicate_rewards": abs(final[1] - final[8]) <= REWARD_BUDGET,
        "fixed_posts": fixed <= 1e-9,
        "finite_extensible_segments": bool(ratios.min() >= 0.25 and ratios.max() <= 3),
        "above_floor": float(data["rod_pos_m"][..., 2].min()) >= -0.01,
        "attached_material_points": attachment <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "maximum_prefix_error_m": prefix,
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
        float(np.ptp(np.stack([row[k] for row in rows]), axis=0).max())
        for k in POSITION_FIELDS
    )
    reward_span = float(np.ptp(rewards, axis=0).max())
    return {
        "maximum_coordinate_span_m": span,
        "maximum_same_action_reward_span": reward_span,
        "passed": span <= POSITION_BUDGET_M and reward_span <= REWARD_BUDGET,
        "population_bound_claimed": False,
    }


def prefix_observation(positions: np.ndarray) -> np.ndarray:
    if (
        positions.shape != (PREFIX_STEPS, N_ENVS, 50, 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("only the complete registered prefix may be observed")
    return positions[list(FRAMES), 1][:, list(NODES)].copy()


def information_value(prefix: np.ndarray, rewards: np.ndarray) -> dict[str, Any]:
    if (
        prefix.shape != (9, 3, 5, 3)
        or rewards.shape != (9, N_ACTIONS)
        or not np.isfinite(prefix).all()
        or not np.isfinite(rewards).all()
        or np.any((rewards <= -98) | (rewards > 1))
    ):
        raise ValueError("complete ordinary nine-world source bank required")
    count, draws = 15, 8192
    rng = np.random.default_rng(260930)
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
    selection = np.zeros((3, N_ACTIONS))
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
                selection[arm] += np.bincount(selected[:, arm], minlength=N_ACTIONS) / (
                    9 * draws
                )
    fixed_action = int(np.argmax(rewards.mean(axis=0)))
    fixed = float(rewards[:, fixed_action].mean())
    means = realized.mean(axis=0)
    oracle = rewards.max(axis=1)
    adjusted = float(means[0] - fixed - PAIR_MARGIN)
    checks = {
        "best_fixed_beats_prefix_hold": fixed - float(rewards[:, 0].mean()) >= 0.05,
        "adjusted_oracle_headroom": float(oracle.mean()) - fixed - PAIR_MARGIN >= 0.05,
        "distinct_oracle_actions": len(set(np.argmax(rewards, axis=1))) >= 2,
        "three_worlds_with_useful_oracle_headroom": int(
            np.sum(oracle - rewards[:, fixed_action] > 0.05)
        )
        >= 3,
        "adjusted_bayes_gain_over_best_fixed": adjusted >= 0.02,
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
