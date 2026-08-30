"""Bounded public-simulator screen for coiling query competence."""

from __future__ import annotations

import itertools
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .deform_state_restart import array_digest
from .dlolab_benchmark import RIGID_FIELDS
from .dlolab_native import STATE_FIELDS

Array: TypeAlias = NDArray[Any]

ACTION_NAMES = (
    "prefix_hold",
    "clockwise_medium",
    "clockwise_tight",
    "clockwise_loose",
    "clockwise_fast",
    "counterclockwise_medium",
    "radial_in",
    "clockwise_medium_duplicate",
)
UNIQUE_ACTION_COUNT = 7
NATIVE_STEPS = 2000
PREFIX_STEPS = 800
PREFIX_FRAMES = (399, 599, 799)
OBSERVED_NODES = (5, 15, 30, 45, 58)
DRAW_COUNT = 8192
DRAW_SEED = 270901
INDEPENDENT_NOISE_STD_M = 0.002
SHARED_BIAS_STD_M = 0.005
REWARD_BUDGET = 0.001
PAIR_MARGIN = 2 * REWARD_BUDGET
MEMORY_NAMES = tuple(f"memory_RigidSolverState.{key}" for key in RIGID_FIELDS) + tuple(
    f"memory_RODSolverState.{key}" for key in STATE_FIELDS
)


def worlds() -> list[dict[str, int | float]]:
    """Return the complete development material grid in canonical order."""
    values = (500.0, 2000.0, 8000.0)
    return [
        {"index": index, "bending_E": bending, "twisting_G": twisting}
        for index, (bending, twisting) in enumerate(itertools.product(values, values))
    ]


def action_bank() -> Array:
    """Build public-geometry Cartesian controls, without simulator outcomes."""
    initial = np.asarray([-0.58, 0.265, 0.012], dtype=np.float64)
    common = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.09, 0.0, 0.018],
            [0.18, 0.0, 0.036],
            [0.27, -0.02, 0.048],
            [0.35, -0.05, 0.055],
        ],
        dtype=np.float64,
    )

    def arc(radius: float, angles_deg: tuple[float, ...], height: float) -> Array:
        angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
        absolute = np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full(len(angles), height),
            ]
        )
        return absolute - initial

    branch = [
        np.repeat(common[-1][None], 6, axis=0),
        arc(0.29, (127, 112, 97, 82, 67, 52), 0.067),
        arc(0.25, (132, 117, 102, 87, 72, 57), 0.067),
        arc(0.34, (127, 112, 97, 82, 67, 52), 0.067),
        arc(0.29, (124, 106, 88, 70, 52, 34), 0.067),
        arc(0.29, (147, 162, 177, 192, 207, 222), 0.067),
        np.linspace(common[-1], np.asarray([-0.05, 0.05, 0.07]) - initial, 7)[1:],
    ]
    branch.append(branch[1].copy())
    relative_path: Array = np.zeros((len(ACTION_NAMES), 11, 3), dtype=np.float64)
    relative_path[:, 1:5] = common[1:]
    for index, continuation in enumerate(branch):
        relative_path[index, 5:] = continuation
    controls: Array = np.zeros((len(ACTION_NAMES), 10, 6), dtype=np.float64)
    controls[..., :3] = np.diff(relative_path, axis=1)
    translation_norm = np.linalg.norm(controls[..., :3], axis=-1)
    if (
        controls.dtype != np.float64
        or not np.isfinite(controls).all()
        or float(translation_norm.max()) > 0.1
        or not np.array_equal(controls[:, :4], np.repeat(controls[:1, :4], 8, 0))
        or not np.array_equal(controls[1], controls[7])
        or np.any(controls[..., 3:] != 0)
    ):
        raise ValueError("registered coiling action geometry changed")
    return controls


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(len(worlds())):
        raise ValueError("unregistered coiling development world")
    return {
        "index": index,
        "name": f"material-{index:02d}",
        "world": worlds()[index],
    }


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-coiling-query-competence-development-v1",
        "role": "bounded_public_simulator_development_screen_not_scientific_evidence",
        "task": "coiling",
        "native_environment": "envs.env_coiling.Train_Env_Coiling",
        "public_geometry_only_action_construction": True,
        "action_names": list(ACTION_NAMES),
        "unique_actions": UNIQUE_ACTION_COUNT,
        "controls_sha256": array_digest(action_bank()),
        "worlds": worlds(),
        "source_prior": [1 / len(worlds())] * len(worlds()),
        "native_steps": NATIVE_STEPS,
        "native_steps_per_macro": 200,
        "pink_micro_controls_per_macro": 10,
        "branch_native_step": PREFIX_STEPS,
        "prefix_frames_zero_based": list(PREFIX_FRAMES),
        "observed_material_nodes": list(OBSERVED_NODES),
        "observation_units": "world_frame_metres",
        "independent_noise_std_m": INDEPENDENT_NOISE_STD_M,
        "shared_translation_bias_std_m": SHARED_BIAS_STD_M,
        "noise_is_assumed_not_sensor_calibrated": True,
        "draw_count": DRAW_COUNT,
        "draw_seed": DRAW_SEED,
        "primary_metric": "unchanged_native_final_reward",
        "qualification": {
            "all_native_observables_and_memory_finite": True,
            "maximum_common_prefix_error_m": 1e-5,
            "maximum_duplicate_coordinate_error_m": 0.001,
            "maximum_duplicate_reward_error": REWARD_BUDGET,
            "maximum_segment_relative_error": 0.1,
            "minimum_rod_height_m": -0.01,
            "maximum_attachment_distance_m": 0.02,
            "maximum_fixed_cone_error_m": 1e-9,
            "native_final_reward_reconstruction_atol": 1e-7,
        },
        "development_gates": {
            "minimum_best_fixed_gain_over_prefix_hold": 0.01,
            "minimum_adjusted_oracle_gain_over_best_fixed": 0.005,
            "minimum_distinct_oracle_actions": 2,
            "minimum_worlds_with_oracle_gain_at_least_0_005": 3,
            "minimum_adjusted_bayes_gain_over_best_fixed": 0.002,
            "minimum_fraction_of_oracle_headroom": 0.25,
            "minimum_bayes_gain_over_map": 0.0,
            "numeric_pair_margin": PAIR_MARGIN,
        },
        "tie_break": "lowest_action_or_world_index",
        "all_worlds_sealed_before_value_analysis": True,
        "prospective_replication_automatically_authorized": False,
        "fallback": "unchanged_best_fixed_action",
        "retry_authorized": False,
        "new_recordings": False,
        "gpu_work": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
    }


def native_reward(positions_m: Array) -> Array:
    positions = np.asarray(positions_m, dtype=np.float64)
    if (
        positions.ndim < 2
        or positions.shape[-2:] != (60, 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("complete finite 60-node coiling geometry required")
    target = np.asarray([0.0, 0.0, 0.15], dtype=np.float64)
    return np.exp(-0.1 * np.linalg.norm(positions - target, axis=-1).sum(axis=-1))


def native_qa(
    arrays: dict[str, Array], native: dict[str, Any], world: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "controls",
        "prefix_positions_m",
        "final_positions_m",
        "final_velocities_m_s",
        "final_gripper_positions_m",
        "joint_targets",
        *MEMORY_NAMES,
    }
    if set(arrays) != required or any(
        not np.isfinite(x).all() for x in arrays.values()
    ):
        raise ValueError("complete finite native coiling bundle required")
    if (
        arrays["controls"].shape != (8, 10, 6)
        or arrays["controls"].dtype != np.float64
        or array_digest(arrays["controls"]) != array_digest(action_bank())
        or arrays["prefix_positions_m"].shape != (3, 8, 5, 3)
        or arrays["final_positions_m"].shape != (8, 60, 3)
        or arrays["final_velocities_m_s"].shape != (8, 60, 3)
        or arrays["final_gripper_positions_m"].shape != (8, 3)
        or arrays["joint_targets"].shape != (8, 101, 9)
        or native.get("native_steps") != NATIVE_STEPS
        or native.get("world") != world
    ):
        raise ValueError("native coiling execution layout changed")
    realized = native.get("world_realization")
    if realized != {
        "bending": [world["bending_E"]] * 8,
        "twisting": [world["twisting_G"]] * 8,
    }:
        raise ValueError("native coiling material realization changed")
    reconstructed = native_reward(arrays["final_positions_m"])
    reported = np.asarray(native.get("native_final_reward"), dtype=np.float64)
    if reported.shape != (8,) or not np.isfinite(reported).all():
        raise ValueError("complete native final rewards required")
    measurements = native.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("native qualification measurements missing")
    reward_error = float(np.max(np.abs(reconstructed - reported)))
    checks = {
        "ordinary_native_success": bool(np.all(reported > 0)),
        "native_final_reward": reward_error <= 1e-7,
        "common_prefix": measurements.get("maximum_common_prefix_error_m", np.inf)
        <= 1e-5,
        "duplicate_positions": measurements.get(
            "maximum_duplicate_coordinate_error_m", np.inf
        )
        <= 0.001,
        "duplicate_rewards": abs(reported[1] - reported[7]) <= REWARD_BUDGET,
        "segment_length": measurements.get("maximum_segment_relative_error", np.inf)
        <= 0.1,
        "above_floor": measurements.get("minimum_rod_height_m", -np.inf) >= -0.01,
        "attached_material_point": measurements.get(
            "maximum_attachment_distance_m", np.inf
        )
        <= 0.02,
        "fixed_cone": measurements.get("maximum_fixed_cone_error_m", np.inf) <= 1e-9,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "native_final_reward_reconstruction_error": reward_error,
        "measurements": measurements,
        "final_rewards": reconstructed.tolist(),
    }


def _posterior(observations: Array, particles: Array) -> Array:
    observed = np.asarray(observations, dtype=np.float64)
    predicted = np.asarray(particles, dtype=np.float64)
    if (
        observed.ndim != 4
        or predicted.shape != (len(worlds()), 3, 5, 3)
        or observed.shape[1:] != predicted.shape[1:]
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("aligned finite coiling prefix observations required")
    residual = (observed[:, None] - predicted[None]).reshape(
        len(observed), len(worlds()), -1, 3
    )
    mean = residual.mean(axis=2)
    centered = residual - mean[:, :, None]
    independent_var = INDEPENDENT_NOISE_STD_M**2
    shared_var = SHARED_BIAS_STD_M**2
    count = residual.shape[2]
    distance = np.sum(centered**2, axis=(2, 3)) / independent_var
    distance += count * np.sum(mean**2, axis=2) / (independent_var + count * shared_var)
    log_weight = -np.log(len(worlds())) - 0.5 * distance
    log_weight -= log_weight.max(axis=1, keepdims=True)
    posterior = np.exp(log_weight)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return np.asarray(posterior, dtype=np.float64)


def source_value(prefix: Array, rewards: Array) -> dict[str, Any]:
    feature = np.asarray(prefix, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    if (
        feature.shape != (len(worlds()), 3, 5, 3)
        or reward.shape != (len(worlds()), UNIQUE_ACTION_COUNT)
        or not np.isfinite(feature).all()
        or not np.isfinite(reward).all()
        or np.any((reward <= 0) | (reward > 1))
    ):
        raise ValueError("complete finite coiling source bank required")
    prior: Array = np.full(len(worlds()), 1 / len(worlds()), dtype=np.float64)
    expected = prior @ reward
    best_fixed_action = int(np.argmax(expected))
    best_fixed_reward = float(expected[best_fixed_action])
    oracle_actions = np.argmax(reward, axis=1)
    oracle_reward = float(prior @ np.max(reward, axis=1))
    rng = np.random.default_rng(DRAW_SEED)
    shared = rng.normal(0, SHARED_BIAS_STD_M, (DRAW_COUNT, 1, 1, 3))
    independent = rng.normal(0, INDEPENDENT_NOISE_STD_M, (DRAW_COUNT, 3, 5, 3))
    bayes_reward = 0.0
    map_reward = 0.0
    world_bayes_reward: Array = np.zeros(len(worlds()), dtype=np.float64)
    for world in range(len(worlds())):
        total_bayes = 0.0
        total_map = 0.0
        for start in range(0, DRAW_COUNT, 256):
            stop = min(start + 256, DRAW_COUNT)
            observations = feature[world] + shared[start:stop] + independent[start:stop]
            posterior = _posterior(observations, feature)
            bayes_action = np.argmax(posterior @ reward, axis=1)
            map_world = np.argmax(posterior, axis=1)
            map_action = np.argmax(reward[map_world], axis=1)
            total_bayes += float(np.sum(reward[world, bayes_action]))
            total_map += float(np.sum(reward[world, map_action]))
        world_bayes_reward[world] = total_bayes / DRAW_COUNT
        bayes_reward += prior[world] * world_bayes_reward[world]
        map_reward += prior[world] * total_map / DRAW_COUNT
    oracle_headroom = oracle_reward - best_fixed_reward
    bayes_gain = float(bayes_reward - best_fixed_reward)
    per_world_oracle_gain = np.max(reward, axis=1) - reward[:, best_fixed_action]
    checks = {
        "best_fixed_gain_over_hold_at_least_0_01": bool(
            best_fixed_reward - float(expected[0]) >= 0.01
        ),
        "adjusted_oracle_gain_at_least_0_005": bool(
            oracle_headroom - PAIR_MARGIN >= 0.005
        ),
        "at_least_two_distinct_oracle_actions": bool(
            len(set(oracle_actions.tolist())) >= 2
        ),
        "at_least_three_worlds_gain_0_005": bool(
            np.count_nonzero(per_world_oracle_gain >= 0.005) >= 3
        ),
        "adjusted_bayes_gain_at_least_0_002": bool(bayes_gain - PAIR_MARGIN >= 0.002),
        "captures_at_least_25pct_oracle_headroom": bool(
            oracle_headroom > 0 and bayes_gain >= 0.25 * oracle_headroom
        ),
        "bayes_not_worse_than_map": bool(bayes_reward >= map_reward),
    }
    return {
        "best_fixed_action": best_fixed_action,
        "best_fixed_action_name": ACTION_NAMES[best_fixed_action],
        "best_fixed_reward": best_fixed_reward,
        "prefix_hold_reward": float(expected[0]),
        "oracle_reward": oracle_reward,
        "oracle_headroom": float(oracle_headroom),
        "oracle_actions": oracle_actions.tolist(),
        "distinct_oracle_actions": len(set(oracle_actions.tolist())),
        "worlds_with_oracle_gain_at_least_0_005": int(
            np.count_nonzero(per_world_oracle_gain >= 0.005)
        ),
        "bayes_reward": float(bayes_reward),
        "bayes_gain_over_best_fixed": bayes_gain,
        "bayes_fraction_of_oracle_headroom": float(
            bayes_gain / oracle_headroom if oracle_headroom > 0 else 0.0
        ),
        "map_reward": float(map_reward),
        "bayes_gain_over_map": float(bayes_reward - map_reward),
        "per_world_bayes_reward": world_bayes_reward.tolist(),
        "per_world_oracle_gain": per_world_oracle_gain.tolist(),
        "checks": checks,
        "development_gate_passed": bool(all(checks.values())),
        "prospective_replication_automatically_authorized": False,
    }
