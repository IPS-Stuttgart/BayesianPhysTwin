"""Post-prefix vertical recovery paths with native actuator-force custody."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from . import dlolab_slingshot_grip as grip
from .dlolab_slingshot_contact import POSITION_FIELDS

VERTICAL_DETOURS_M = (-0.02, -0.01, 0.01, 0.02)
FORCES_N = (-24.0, -24.0, -24.0, -24.0, -24.0, -3.0, -24.0, -3.0)
RETAINED_GRIP_INDICES = {4: 2, 5: 5, 6: 6, 7: 7}


def protocol() -> dict[str, Any]:
    result = grip.protocol()
    result.update(
        schema="dlolab-slingshot-contact-path-source-v1",
        only_new_control_is_post_prefix_grip_force=False,
        post_prefix_finger_forces_N=list(FORCES_N),
        cartesian_source_indices=[6, 6, 6, 6, 6, 6, 5, 6],
        vertical_detours_m=list(VERTICAL_DETOURS_M),
        detour_actions=[0, 1, 2, 3],
        vertical_detour_added_to_macro_index=1,
        vertical_detour_subtracted_from_macro_index=2,
        planar_limit_policy="scale_xy_only_if_translation_norm_exceeds_0.1_m",
        final_cartesian_endpoint_not_claimed_identical=True,
        retained_grip_action_pairs=[
            list(pair) for pair in RETAINED_GRIP_INDICES.items()
        ],
        retained_grip_position_atol_m=1e-6,
        minimum_gain_over_previous_posterior_policy=0.002,
        earlier_failed_gates_unchanged=True,
    )
    return result


def controls(source: np.ndarray) -> np.ndarray:
    previous = grip.controls(source)
    result = previous[[2, 2, 2, 2, 2, 5, 6, 7]].copy()
    for index, detour in enumerate(VERTICAL_DETOURS_M):
        for macro, sign in ((1, 1), (2, -1)):
            value = result[index, macro, :3]
            value[2] += sign * detour
            if abs(value[2]) >= 0.1:
                raise ValueError("vertical recovery exceeds the native action limit")
            planar = float(np.linalg.norm(value[:2]))
            allowed = float(np.sqrt(0.1**2 - value[2] ** 2))
            if planar > allowed:
                value[:2] *= allowed / planar
    if np.max(np.linalg.norm(result[:, :, :3], axis=-1)) > 0.1 + 1e-12:
        raise ValueError("native translation limit exceeded")
    if not np.array_equal(result[:, 0], previous[:, 0]):
        raise ValueError("causal prefix changed")
    return result


def validate_force_record(record: dict[str, Any]) -> None:
    with patch.object(grip, "FORCES_N", FORCES_N):
        grip.validate_force_record(record)


def run_path_world(
    upstream: Path, output: Path, commands: np.ndarray, index: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with patch.object(grip, "FORCES_N", FORCES_N):
        return grip.run_grip_world(upstream, output, commands, index)


def reference_checks(
    candidate: dict[str, np.ndarray],
    contact: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    rewards: list[float],
    contact_rewards: list[float],
    previous_rewards: list[float],
) -> dict[str, Any]:
    original = grip.reference_checks(candidate, contact, rewards, contact_rewards)
    for current, old in RETAINED_GRIP_INDICES.items():
        if not np.array_equal(
            candidate["controls"][current], previous["controls"][old]
        ):
            raise ValueError("retained strong Cartesian control changed")
    error = max(
        float(np.max(np.abs(candidate[name][:, current] - previous[name][:, old])))
        for name in POSITION_FIELDS
        for current, old in RETAINED_GRIP_INDICES.items()
    )
    checks = {
        **original["checks"],
        "retained_grip_positions_within_1um": error <= 1e-6,
        "retained_grip_rewards_exact": all(
            rewards[current] == previous_rewards[old]
            for current, old in RETAINED_GRIP_INDICES.items()
        ),
    }
    return {
        **original,
        "checks": checks,
        "passed": all(checks.values()),
        "retained_grip_error_m": error,
    }


def compare_previous_policy(
    metrics: dict[str, Any], previous: dict[str, Any]
) -> dict[str, Any]:
    name = "bias_aware_posterior_mean"
    current_reward = metrics["arms"][name]["expected_native_reward"]
    previous_reward = previous["metrics"]["arms"][name]["expected_native_reward"]
    gain = float(current_reward - previous_reward)
    checks = {
        **metrics["checks"],
        "gain_over_previous_posterior_at_least_0_002": gain >= 0.002,
    }
    return {
        **metrics,
        "previous_grip_posterior_reward": previous_reward,
        "gain_over_previous_posterior_policy": gain,
        "checks": checks,
        "source_information_value_passed": all(checks.values()),
    }
