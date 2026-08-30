"""Bounded public-simulator headroom screen for DLO-Lab separation."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .deform_state_restart import array_digest
from .dlolab_benchmark import RIGID_FIELDS
from .dlolab_native import STATE_FIELDS

Array: TypeAlias = NDArray[Any]

ACTION_ANGLES_DEG = (-35.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0, 35.0)
WORLD_ANGLES_DEG = (-35.0, -26.25, -17.5, -8.75, 0.0, 8.75, 17.5, 26.25, 35.0)
ACTION_NAMES = (
    ("prefix_hold",)
    + tuple(f"pull_{angle:+05.1f}_deg" for angle in ACTION_ANGLES_DEG)
    + (
        "duplicate_negative_extreme",
        "duplicate_positive_extreme",
    )
)
UNIQUE_ACTION_COUNT = 1 + len(ACTION_ANGLES_DEG)
NATIVE_MACROS = 6
NATIVE_STEPS_PER_MACRO = 200
NATIVE_STEPS = NATIVE_MACROS * NATIVE_STEPS_PER_MACRO
PREFIX_MACROS = 2
PREFIX_STEPS = PREFIX_MACROS * NATIVE_STEPS_PER_MACRO
PREFIX_FRAMES = (199, 399)
OBSERVED_NODES = (2, 8, 15, 21, 27)
NUMERIC_REWARD_MARGIN_M = 0.001
MEMORY_NAMES = tuple(f"memory_RigidSolverState.{key}" for key in RIGID_FIELDS) + tuple(
    f"memory_RODSolverState.{key}" for key in STATE_FIELDS
)


def worlds() -> list[dict[str, int | float]]:
    """Return the complete geometry-only development roster."""
    return [
        {"index": index, "rotation_deg": angle}
        for index, angle in enumerate(WORLD_ANGLES_DEG)
    ]


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(len(WORLD_ANGLES_DEG)):
        raise ValueError("unregistered separation development world")
    return {
        "index": index,
        "name": f"rotation-{index:02d}",
        "world": worlds()[index],
    }


def action_bank() -> Array:
    """Build a symmetric pull bank from public scene geometry only."""
    unique: Array = np.zeros((UNIQUE_ACTION_COUNT, NATIVE_MACROS, 12), dtype=np.float64)
    unique[:, :PREFIX_MACROS, 2] = 0.015
    unique[:, :PREFIX_MACROS, 5] = 0.015
    for index, angle_deg in enumerate(ACTION_ANGLES_DEG):
        angle = np.deg2rad(angle_deg)
        direction = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        unique[index + 1, PREFIX_MACROS:, :3] = 0.03 * direction
        unique[index + 1, PREFIX_MACROS:, 3:6] = -0.03 * direction
    actions = np.concatenate([unique, unique[[1, -1]]], axis=0)
    translation = actions[..., :6].reshape(len(actions), NATIVE_MACROS, 2, 3)
    if (
        actions.shape != (11, NATIVE_MACROS, 12)
        or actions.dtype != np.float64
        or not np.isfinite(actions).all()
        or np.any(actions[..., 6:] != 0)
        or float(np.linalg.norm(translation, axis=-1).max()) > 0.031
        or not np.array_equal(
            actions[:, :PREFIX_MACROS],
            np.repeat(actions[:1, :PREFIX_MACROS], len(actions), axis=0),
        )
        or not np.array_equal(actions[9], actions[1])
        or not np.array_equal(actions[10], actions[8])
    ):
        raise ValueError("registered separation action geometry changed")
    return actions


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-separation-headroom-development-v1",
        "role": "bounded_public_simulator_development_screen_not_scientific_evidence",
        "task": "separation",
        "native_environment": "envs.env_separation.Train_Env_Separation",
        "public_initial_geometry_rotation_only": True,
        "worlds": worlds(),
        "world_count": len(WORLD_ANGLES_DEG),
        "world_distribution": "fixed rotations of both released rope geometries about their shared centroid",
        "action_names": list(ACTION_NAMES),
        "action_angles_deg": list(ACTION_ANGLES_DEG),
        "unique_actions": UNIQUE_ACTION_COUNT,
        "controls_sha256": array_digest(action_bank()),
        "native_macros": NATIVE_MACROS,
        "native_steps_per_macro": NATIVE_STEPS_PER_MACRO,
        "pink_micro_controls_per_macro": 10,
        "native_steps": NATIVE_STEPS,
        "branch_native_step": PREFIX_STEPS,
        "prefix_frames_zero_based": list(PREFIX_FRAMES),
        "observed_nodes_per_rope": list(OBSERVED_NODES),
        "primary_metric": "unchanged_native_final_symmetric_nearest_point_distance_m",
        "qualification": {
            "all_native_observables_and_memory_finite": True,
            "maximum_common_prefix_error_m": 1e-5,
            "maximum_duplicate_coordinate_error_m": 0.001,
            "maximum_duplicate_reward_error_m": NUMERIC_REWARD_MARGIN_M,
            "maximum_rotation_realization_error_m": 1e-10,
            "maximum_segment_relative_error": 0.10,
            "minimum_rod_height_m": -0.01,
            "maximum_attachment_distance_m": 0.02,
            "native_final_reward_reconstruction_atol_m": 1e-7,
        },
        "development_gates": {
            "minimum_best_fixed_gain_over_shared_prefix_hold_m": 0.02,
            "minimum_adjusted_oracle_headroom_m": 0.01,
            "minimum_distinct_oracle_actions": 3,
            "minimum_worlds_with_oracle_gain_at_least_0_01_m": 4,
            "numeric_reward_margin_m": NUMERIC_REWARD_MARGIN_M,
        },
        "all_worlds_sealed_before_value_analysis": True,
        "source_transfer_automatically_authorized": False,
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


def native_reward(rope_a_m: Array, rope_b_m: Array) -> Array:
    a = np.asarray(rope_a_m, dtype=np.float64)
    b = np.asarray(rope_b_m, dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim < 3
        or a.shape[-2:] != (30, 3)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("aligned finite 30-node rope geometries required")
    distance = np.linalg.norm(a[..., :, None, :] - b[..., None, :, :], axis=-1)
    return distance.min(axis=-1).mean(axis=-1) + distance.min(axis=-2).mean(axis=-1)


def native_qa(
    arrays: dict[str, Array], native: dict[str, Any], world: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "controls",
        "prefix_rope_a_m",
        "prefix_rope_b_m",
        "final_rope_a_m",
        "final_rope_b_m",
        "final_rope_a_velocity_m_s",
        "final_rope_b_velocity_m_s",
        "final_gripper_a_m",
        "final_gripper_b_m",
        "joint_targets",
        *MEMORY_NAMES,
    }
    if set(arrays) != required or any(
        not np.isfinite(value).all() for value in arrays.values()
    ):
        raise ValueError("complete finite native separation bundle required")
    if (
        arrays["controls"].shape != (11, NATIVE_MACROS, 12)
        or arrays["controls"].dtype != np.float64
        or array_digest(arrays["controls"]) != array_digest(action_bank())
        or arrays["prefix_rope_a_m"].shape != (2, 11, 5, 3)
        or arrays["prefix_rope_b_m"].shape != (2, 11, 5, 3)
        or arrays["final_rope_a_m"].shape != (11, 30, 3)
        or arrays["final_rope_b_m"].shape != (11, 30, 3)
        or arrays["final_rope_a_velocity_m_s"].shape != (11, 30, 3)
        or arrays["final_rope_b_velocity_m_s"].shape != (11, 30, 3)
        or arrays["final_gripper_a_m"].shape != (11, 3)
        or arrays["final_gripper_b_m"].shape != (11, 3)
        or arrays["joint_targets"].shape != (11, 61, 18)
        or native.get("native_steps") != NATIVE_STEPS
        or native.get("world") != world
    ):
        raise ValueError("native separation execution layout changed")
    reported = np.asarray(native.get("native_final_reward_m"), dtype=np.float64)
    reconstructed = native_reward(arrays["final_rope_a_m"], arrays["final_rope_b_m"])
    if reported.shape != (11,) or not np.isfinite(reported).all():
        raise ValueError("complete finite native separation rewards required")
    measurements = native.get("measurements")
    realization = native.get("world_realization")
    if not isinstance(measurements, dict) or not isinstance(realization, dict):
        raise ValueError("native separation qualification measurements missing")
    reward_error = float(np.max(np.abs(reconstructed - reported)))
    duplicate_reward_error = float(
        max(abs(reported[1] - reported[9]), abs(reported[8] - reported[10]))
    )
    checks = {
        "ordinary_native_success": bool(np.all(reported >= 0)),
        "native_final_reward": reward_error <= 1e-7,
        "common_prefix": measurements.get("maximum_common_prefix_error_m", np.inf)
        <= 1e-5,
        "duplicate_positions": measurements.get(
            "maximum_duplicate_coordinate_error_m", np.inf
        )
        <= 0.001,
        "duplicate_rewards": duplicate_reward_error <= NUMERIC_REWARD_MARGIN_M,
        "registered_world_rotation": realization.get("rotation_deg")
        == world["rotation_deg"],
        "rotation_realization": realization.get(
            "maximum_rotation_realization_error_m", np.inf
        )
        <= 1e-10,
        "segment_length": measurements.get("maximum_segment_relative_error", np.inf)
        <= 0.10,
        "rod_height": measurements.get("minimum_rod_height_m", -np.inf) >= -0.01,
        "material_attachment": measurements.get("maximum_attachment_distance_m", np.inf)
        <= 0.02,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "final_rewards_m": reported.tolist(),
        "reward_reconstruction_error_m": reward_error,
        "duplicate_reward_error_m": duplicate_reward_error,
        "measurements": measurements,
        "world_realization": realization,
    }


def development_metrics(rewards: Array) -> dict[str, Any]:
    value = np.asarray(rewards, dtype=np.float64)
    if (
        value.shape != (len(WORLD_ANGLES_DEG), UNIQUE_ACTION_COUNT)
        or not np.isfinite(value).all()
        or np.any(value < 0)
    ):
        raise ValueError("complete finite separation reward bank required")
    expected = value.mean(axis=0)
    best_fixed_action = int(np.argmax(expected))
    best_fixed_reward = float(expected[best_fixed_action])
    oracle_actions = np.argmax(value, axis=1)
    oracle_per_world = np.max(value, axis=1)
    oracle_reward = float(oracle_per_world.mean())
    gains = oracle_per_world - value[:, best_fixed_action]
    adjusted_headroom = oracle_reward - best_fixed_reward - NUMERIC_REWARD_MARGIN_M
    checks = {
        "best_fixed_gain_over_shared_prefix_hold_at_least_0_02_m": bool(
            best_fixed_reward - float(value[:, 0].mean()) >= 0.02
        ),
        "adjusted_oracle_headroom_at_least_0_01_m": bool(adjusted_headroom >= 0.01),
        "at_least_three_distinct_oracle_actions": bool(
            len(set(oracle_actions.tolist())) >= 3
        ),
        "at_least_four_worlds_gain_0_01_m": bool(np.count_nonzero(gains >= 0.01) >= 4),
    }
    return {
        "best_fixed_action": best_fixed_action,
        "best_fixed_action_name": ACTION_NAMES[best_fixed_action],
        "best_fixed_reward_m": best_fixed_reward,
        "oracle_reward_m": oracle_reward,
        "oracle_headroom_m": oracle_reward - best_fixed_reward,
        "adjusted_oracle_headroom_m": adjusted_headroom,
        "distinct_oracle_actions": len(set(oracle_actions.tolist())),
        "oracle_actions": oracle_actions.tolist(),
        "world_oracle_gains_m": gains.tolist(),
        "worlds_with_oracle_gain_at_least_0_01_m": int(np.count_nonzero(gains >= 0.01)),
        "checks": checks,
        "development_gate_passed": bool(all(checks.values())),
        "source_transfer_automatically_authorized": False,
    }
