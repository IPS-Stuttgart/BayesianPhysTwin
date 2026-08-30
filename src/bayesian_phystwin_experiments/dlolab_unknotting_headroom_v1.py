"""Bounded public-simulator headroom screen for DLO-Lab unknotting."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .deform_state_restart import array_digest
from .dlolab_benchmark import RIGID_FIELDS
from .dlolab_native import STATE_FIELDS

Array: TypeAlias = NDArray[Any]

BASE_PULL_ANGLE_DEG = -85.51756252062847
ACTION_OFFSETS_DEG = (-35.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0, 35.0)
WORLD_ROTATIONS_DEG = (-35.0, -26.25, -17.5, -8.75, 0.0, 8.75, 17.5, 26.25, 35.0)
ACTION_NAMES = (
    ("prefix_hold",)
    + tuple(f"pull_{offset:+05.1f}_deg" for offset in ACTION_OFFSETS_DEG)
    + ("duplicate_negative_extreme", "duplicate_positive_extreme")
)
UNIQUE_ACTION_COUNT = 1 + len(ACTION_OFFSETS_DEG)
NATIVE_MACROS = 6
NATIVE_STEPS_PER_MACRO = 200
NATIVE_STEPS = NATIVE_MACROS * NATIVE_STEPS_PER_MACRO
PREFIX_MACROS = 2
PREFIX_STEPS = PREFIX_MACROS * NATIVE_STEPS_PER_MACRO
PREFIX_FRAMES = (199, 399)
OBSERVED_NODES = (2, 10, 20, 30, 37)
CONTROL_NODES = (2, 37)
NUMERIC_REWARD_MARGIN = 0.002
MEMORY_NAMES = tuple(f"memory_RigidSolverState.{key}" for key in RIGID_FIELDS) + tuple(
    f"memory_RODSolverState.{key}" for key in STATE_FIELDS
)


def worlds() -> list[dict[str, int | float]]:
    """Return the complete geometry-only development roster."""
    return [
        {"index": index, "rotation_deg": angle}
        for index, angle in enumerate(WORLD_ROTATIONS_DEG)
    ]


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(len(WORLD_ROTATIONS_DEG)):
        raise ValueError("unregistered unknotting development world")
    return {
        "index": index,
        "name": f"rotation-{index:02d}",
        "world": worlds()[index],
    }


def action_bank() -> Array:
    """Build symmetric pulls around the public control-point axis."""
    unique: Array = np.zeros((UNIQUE_ACTION_COUNT, NATIVE_MACROS, 12), dtype=np.float64)
    unique[:, :PREFIX_MACROS, 2] = 0.015
    unique[:, :PREFIX_MACROS, 5] = 0.015
    for index, offset_deg in enumerate(ACTION_OFFSETS_DEG):
        angle = np.deg2rad(BASE_PULL_ANGLE_DEG + offset_deg)
        direction = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        unique[index + 1, PREFIX_MACROS:, :3] = 0.025 * direction
        unique[index + 1, PREFIX_MACROS:, 3:6] = -0.025 * direction
    actions = np.concatenate([unique, unique[[1, -1]]], axis=0)
    translation = actions[..., :6].reshape(len(actions), NATIVE_MACROS, 2, 3)
    if (
        actions.shape != (11, NATIVE_MACROS, 12)
        or actions.dtype != np.float64
        or not np.isfinite(actions).all()
        or np.any(actions[..., 6:] != 0)
        or float(np.linalg.norm(translation, axis=-1).max()) > 0.026
        or not np.array_equal(
            actions[:, :PREFIX_MACROS],
            np.repeat(actions[:1, :PREFIX_MACROS], len(actions), axis=0),
        )
        or not np.array_equal(actions[9], actions[1])
        or not np.array_equal(actions[10], actions[8])
    ):
        raise ValueError("registered unknotting action geometry changed")
    return actions


def native_reward(rope_m: Array, interval_m: float = 0.02) -> Array:
    """Reconstruct the unchanged native unknotting reward in NumPy."""
    points = np.asarray(rope_m, dtype=np.float64)
    if (
        points.ndim < 3
        or points.shape[-2:] != (50, 3)
        or not np.isfinite(points).all()
        or interval_m != 0.02
    ):
        raise ValueError("finite 50-node rope geometries and native interval required")
    edges = points[..., 1:, :] - points[..., :-1, :]
    midpoints = (points[..., 1:, :] + points[..., :-1, :]) / 2
    displacement = midpoints[..., :, None, :] - midpoints[..., None, :, :]
    numerator = np.abs(
        np.sum(
            displacement * np.cross(edges[..., :, None, :], edges[..., None, :, :]),
            axis=-1,
        )
    )
    distance = np.linalg.norm(displacement, axis=-1)
    denominator = (distance**2 + 0.02**2) ** 1.5
    acn_matrix = numerator / denominator
    edge_index = np.arange(49)
    neighbor_mask = np.abs(edge_index[:, None] - edge_index[None, :]) <= 1
    acn_matrix[..., neighbor_mask] = 0
    acn_value = acn_matrix.sum(axis=(-2, -1)) / (4 * np.pi)
    penetration = np.maximum(2 * interval_m - distance, 0)
    penetration[..., neighbor_mask] = 0
    penalty = acn_value + 100 * np.square(penetration).sum(axis=(-2, -1))
    return np.exp(-penalty)


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-unknotting-headroom-development-v1",
        "role": "bounded_public_simulator_development_screen_not_scientific_evidence",
        "task": "unknotting",
        "native_environment": "envs.env_unknotting.Train_Env_Unknotting",
        "public_initial_geometry_rotation_only": True,
        "public_geometry_control_axis_angle_deg": BASE_PULL_ANGLE_DEG,
        "worlds": worlds(),
        "world_count": len(WORLD_ROTATIONS_DEG),
        "world_distribution": "fixed rotations of the released knot about its centroid",
        "action_names": list(ACTION_NAMES),
        "action_offsets_deg": list(ACTION_OFFSETS_DEG),
        "unique_actions": UNIQUE_ACTION_COUNT,
        "controls_sha256": array_digest(action_bank()),
        "native_macros": NATIVE_MACROS,
        "native_steps_per_macro": NATIVE_STEPS_PER_MACRO,
        "pink_micro_controls_per_macro": 10,
        "native_steps": NATIVE_STEPS,
        "branch_native_step": PREFIX_STEPS,
        "prefix_frames_zero_based": list(PREFIX_FRAMES),
        "observed_nodes": list(OBSERVED_NODES),
        "control_nodes": list(CONTROL_NODES),
        "primary_metric": "unchanged_native_final_exp_negative_unknotting_penalty",
        "qualification": {
            "all_native_observables_and_memory_finite": True,
            "maximum_common_prefix_error_m": 1e-5,
            "maximum_duplicate_coordinate_error_m": 0.001,
            "maximum_duplicate_reward_error": NUMERIC_REWARD_MARGIN,
            "maximum_rotation_realization_error_m": 1e-10,
            "maximum_segment_relative_error": 0.10,
            "minimum_rod_height_m": -0.01,
            "maximum_attachment_offset_drift_m": 0.001,
            "native_final_reward_reconstruction_atol": 1e-7,
        },
        "development_gates": {
            "minimum_best_fixed_gain_over_shared_prefix_hold": 0.05,
            "minimum_adjusted_oracle_headroom": 0.03,
            "minimum_distinct_oracle_actions": 3,
            "minimum_worlds_with_oracle_gain_at_least_0_03": 4,
            "numeric_reward_margin": NUMERIC_REWARD_MARGIN,
        },
        "all_worlds_sealed_before_value_analysis": True,
        "runtime_preflight_before_attempt_consumption": True,
        "external_write_once_attempt_ledger": True,
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


def native_qa(
    arrays: dict[str, Array], native: dict[str, Any], world: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "controls",
        "prefix_rope_m",
        "final_rope_m",
        "final_rope_velocity_m_s",
        "final_gripper_a_m",
        "final_gripper_b_m",
        "joint_targets",
        *MEMORY_NAMES,
    }
    if set(arrays) != required or any(
        not np.isfinite(value).all() for value in arrays.values()
    ):
        raise ValueError("complete finite native unknotting bundle required")
    if (
        arrays["controls"].shape != (11, NATIVE_MACROS, 12)
        or arrays["controls"].dtype != np.float64
        or array_digest(arrays["controls"]) != array_digest(action_bank())
        or arrays["prefix_rope_m"].shape != (2, 11, 5, 3)
        or arrays["final_rope_m"].shape != (11, 50, 3)
        or arrays["final_rope_velocity_m_s"].shape != (11, 50, 3)
        or arrays["final_gripper_a_m"].shape != (11, 3)
        or arrays["final_gripper_b_m"].shape != (11, 3)
        or arrays["joint_targets"].shape != (11, 61, 18)
        or native.get("native_steps") != NATIVE_STEPS
        or native.get("world") != world
    ):
        raise ValueError("native unknotting execution layout changed")
    reported = np.asarray(native.get("native_final_reward"), dtype=np.float64)
    reconstructed = native_reward(arrays["final_rope_m"])
    if reported.shape != (11,) or not np.isfinite(reported).all():
        raise ValueError("complete finite native unknotting rewards required")
    measurements = native.get("measurements")
    realization = native.get("world_realization")
    if not isinstance(measurements, dict) or not isinstance(realization, dict):
        raise ValueError("native unknotting qualification measurements missing")
    reward_error = float(np.max(np.abs(reconstructed - reported)))
    duplicate_reward_error = float(
        max(abs(reported[1] - reported[9]), abs(reported[8] - reported[10]))
    )
    checks = {
        "ordinary_native_success": bool(np.all((reported >= 0) & (reported <= 1))),
        "native_final_reward": reward_error <= 1e-7,
        "common_prefix": measurements.get("maximum_common_prefix_error_m", np.inf)
        <= 1e-5,
        "duplicate_positions": measurements.get(
            "maximum_duplicate_coordinate_error_m", np.inf
        )
        <= 0.001,
        "duplicate_rewards": duplicate_reward_error <= NUMERIC_REWARD_MARGIN,
        "registered_world_rotation": realization.get("rotation_deg")
        == world["rotation_deg"],
        "rotation_realization": realization.get(
            "maximum_rotation_realization_error_m", np.inf
        )
        <= 1e-10,
        "segment_length": measurements.get("maximum_segment_relative_error", np.inf)
        <= 0.10,
        "rod_height": measurements.get("minimum_rod_height_m", -np.inf) >= -0.01,
        "material_attachment": measurements.get(
            "maximum_attachment_offset_drift_m", np.inf
        )
        <= 0.001,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "final_rewards": reported.tolist(),
        "reward_reconstruction_error": reward_error,
        "duplicate_reward_error": duplicate_reward_error,
        "measurements": measurements,
        "world_realization": realization,
    }


def development_metrics(rewards: Array) -> dict[str, Any]:
    value = np.asarray(rewards, dtype=np.float64)
    if (
        value.shape != (len(WORLD_ROTATIONS_DEG), UNIQUE_ACTION_COUNT)
        or not np.isfinite(value).all()
        or np.any((value < 0) | (value > 1))
    ):
        raise ValueError("complete finite unknotting reward bank required")
    expected = value.mean(axis=0)
    best_fixed_action = int(np.argmax(expected))
    best_fixed_reward = float(expected[best_fixed_action])
    oracle_actions = np.argmax(value, axis=1)
    oracle_per_world = np.max(value, axis=1)
    oracle_reward = float(oracle_per_world.mean())
    gains = oracle_per_world - value[:, best_fixed_action]
    adjusted_headroom = oracle_reward - best_fixed_reward - NUMERIC_REWARD_MARGIN
    checks = {
        "best_fixed_gain_over_shared_prefix_hold_at_least_0_05": bool(
            best_fixed_reward - float(value[:, 0].mean()) >= 0.05
        ),
        "adjusted_oracle_headroom_at_least_0_03": bool(adjusted_headroom >= 0.03),
        "at_least_three_distinct_oracle_actions": bool(
            len(set(oracle_actions.tolist())) >= 3
        ),
        "at_least_four_worlds_gain_0_03": bool(np.count_nonzero(gains >= 0.03) >= 4),
    }
    return {
        "best_fixed_action": best_fixed_action,
        "best_fixed_action_name": ACTION_NAMES[best_fixed_action],
        "best_fixed_reward": best_fixed_reward,
        "oracle_reward": oracle_reward,
        "oracle_headroom": oracle_reward - best_fixed_reward,
        "adjusted_oracle_headroom": adjusted_headroom,
        "distinct_oracle_actions": len(set(oracle_actions.tolist())),
        "oracle_actions": oracle_actions.tolist(),
        "world_oracle_gains": gains.tolist(),
        "worlds_with_oracle_gain_at_least_0_03": int(np.count_nonzero(gains >= 0.03)),
        "checks": checks,
        "development_gate_passed": bool(all(checks.values())),
        "source_transfer_automatically_authorized": False,
    }
