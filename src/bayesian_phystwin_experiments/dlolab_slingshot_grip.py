"""Causal native grip-force choices with unchanged prefix and release."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .dlolab_slingshot_contact import POSITION_FIELDS, run_contact_world
from .dlolab_slingshot_contact import protocol as contact_protocol

CARTESIAN_INDICES = (6, 6, 6, 5, 5, 6, 5, 6)
FORCES_N = (-6.0, -12.0, -24.0, -6.0, -12.0, -3.0, -24.0, -3.0)
FORCE_FRAMES = (0, *range(100, 700, 20))


def protocol() -> dict[str, Any]:
    result = contact_protocol()
    result.update(
        schema="dlolab-slingshot-grip-recovery-source-v1",
        native_actions_reward_controller_and_release_unchanged=False,
        cartesian_source_indices=list(CARTESIAN_INDICES),
        post_prefix_finger_forces_N=list(FORCES_N),
        force_branch_native_step=300,
        force_call_native_steps=list(FORCE_FRAMES),
        release_native_step=700,
        release_position_m=0.08,
        reset_force_N=-1.0,
        prefix_force_N=-3.0,
        native_physics_reward_arm_controller_and_release_unchanged=True,
        fallback_action_index=5,
        duplicate_fallback_index=7,
        fallback_reference_cartesian_index=6,
        fallback_reference_force_N=-3.0,
        fallback_replay_position_atol_m=1e-6,
        source_noise_integration_reused_not_independent_confirmation=True,
        only_new_control_is_post_prefix_grip_force=True,
    )
    return result


def controls(source: np.ndarray) -> np.ndarray:
    if (
        source.shape != (8, 3, 6)
        or source.dtype != np.float64
        or not np.isfinite(source).all()
        or not np.all(source[:, 0] == source[5, 0])
        or not np.array_equal(source[5], source[7])
    ):
        raise ValueError("complete shared-prefix source command bank required")
    result = source[list(CARTESIAN_INDICES)].copy()
    if (
        np.max(np.linalg.norm(result[:, :, :3], axis=-1)) > 0.1 + 1e-12
        or np.max(np.abs(result[:, :, 3:])) > 1
    ):
        raise ValueError("native Cartesian action limits exceeded")
    return result


def as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def force_at(frame: int) -> np.ndarray:
    if type(frame) is not int or frame not in FORCE_FRAMES:
        raise ValueError("unregistered native force-command frame")
    values = (
        np.full(8, -1.0 if frame == 0 else -3.0)
        if frame < 300
        else np.asarray(FORCES_N)
    )
    return np.repeat(values[:, None], 2, axis=1)


def validate_force_record(record: dict[str, Any]) -> None:
    if record["native_steps"] != 900 or record["initializations"] != 1:
        raise ValueError("incomplete native force schedule")
    calls = record["force_calls"]
    if [call["native_step"] for call in calls] != list(FORCE_FRAMES):
        raise ValueError("missing, repeated or shifted force command")
    for call in calls:
        expected = force_at(call["native_step"])
        if not np.array_equal(call["command_N"], expected) or not np.array_equal(
            call["solver_control_force_N"], expected
        ):
            raise ValueError("force schedule did not reach native actuator")
    if record["release_calls"] != [
        {"native_step": 700, "finger_position_m": [[0.08, 0.08]] * 8}
    ]:
        raise ValueError("native release changed")
    for name, value in (("lower_force_limit_N", -30.0), ("upper_force_limit_N", 30.0)):
        if not np.array_equal(record[name], [value, value]):
            raise ValueError("native actuator limits changed")


@contextmanager
def grip_adapter(environment_class: Any) -> Iterator[dict[str, Any]]:
    original_initialize = environment_class.init_cmaes_env
    captured: dict[str, Any] = {
        "native_steps": 0,
        "initializations": 0,
        "force_calls": [],
        "release_calls": [],
    }
    with ExitStack() as hooks:

        def initialize(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_initialize(self, *args, **kwargs)
            captured["initializations"] += 1
            if captured["initializations"] != 1:
                raise ValueError("exactly one native initialization required")
            robot, fingers = self.franka1, self.c1.fingers_dof
            if as_array(fingers).tolist() != [7, 8]:
                raise ValueError("registered two-finger actuator layout changed")
            lower, upper = robot.get_dofs_force_range(fingers)
            captured["lower_force_limit_N"] = as_array(lower).tolist()
            captured["upper_force_limit_N"] = as_array(upper).tolist()
            if not np.array_equal(as_array(lower), [-30.0] * 2) or not np.array_equal(
                as_array(upper), [30.0] * 2
            ):
                raise ValueError("registered native force limits unavailable")
            original_step = self.scene.step
            original_force = robot.control_dofs_force
            original_position = robot.control_dofs_position

            def step(*args: Any, **kwargs: Any) -> Any:
                value = original_step(*args, **kwargs)
                captured["native_steps"] += 1
                if captured["native_steps"] > 900:
                    raise ValueError("native horizon extended")
                return value

            def force_command(
                force: Any, dofs_idx_local: Any = None, envs_idx: Any = None
            ) -> Any:
                frame = captured["native_steps"]
                expected_input = -1.0 if frame == 0 else -3.0
                if (
                    envs_idx is not None
                    or not np.array_equal(as_array(dofs_idx_local), [7, 8])
                    or as_array(force).shape != (8, 2)
                    or not np.all(as_array(force) == expected_input)
                    or len(captured["force_calls"]) >= len(FORCE_FRAMES)
                    or frame != FORCE_FRAMES[len(captured["force_calls"])]
                ):
                    raise ValueError("native force-call contract changed")
                expected = force_at(frame)
                value = original_force(
                    force if frame < 300 else expected, dofs_idx_local, envs_idx
                )
                actual = as_array(robot.get_dofs_control_force(fingers))
                if not np.array_equal(actual, expected):
                    raise ValueError("native actuator clipped or changed force")
                captured["force_calls"].append(
                    {
                        "native_step": frame,
                        "command_N": expected.tolist(),
                        "solver_control_force_N": actual.tolist(),
                    }
                )
                return value

            def position_command(
                position: Any, dofs_idx_local: Any = None, envs_idx: Any = None
            ) -> Any:
                if np.array_equal(as_array(dofs_idx_local), [7, 8]):
                    if (
                        captured["native_steps"] != 700
                        or captured["release_calls"]
                        or not np.array_equal(as_array(position), [[0.08, 0.08]] * 8)
                        or not np.array_equal(as_array(envs_idx), np.arange(8))
                    ):
                        raise ValueError("native release contract changed")
                    captured["release_calls"].append(
                        {
                            "native_step": 700,
                            "finger_position_m": as_array(position).tolist(),
                        }
                    )
                return original_position(position, dofs_idx_local, envs_idx)

            hooks.enter_context(patch.object(self.scene, "step", step))
            hooks.enter_context(
                patch.object(robot, "control_dofs_force", force_command)
            )
            hooks.enter_context(
                patch.object(robot, "control_dofs_position", position_command)
            )
            return result

        hooks.enter_context(
            patch.object(environment_class, "init_cmaes_env", initialize)
        )
        yield captured
        validate_force_record(captured)


def run_grip_world(
    upstream: Path, output: Path, commands: np.ndarray, index: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    native = importlib.import_module("envs.env_slingshot")
    with grip_adapter(native.Train_Env_Slingshot) as record:
        arrays, report = run_contact_world(upstream, output, commands, index)
    return arrays, {**report, "grip_schedule": record}


def reference_checks(
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    rewards: list[float],
    reference_rewards: list[float],
) -> dict[str, Any]:
    if arrays["rod_pos_m"].shape != (900, 8, 12, 3) or reference["rod_pos_m"].shape != (
        900,
        8,
        12,
        3,
    ):
        raise ValueError("complete matched native reference required")
    for index in (5, 7):
        if not np.array_equal(arrays["controls"][index], reference["controls"][6]):
            raise ValueError("fallback Cartesian command changed")
    error = max(
        float(np.max(np.abs(arrays[k][:, index] - reference[k][:, 6])))
        for k in POSITION_FIELDS
        for index in (5, 7)
    )
    prefix_error = max(
        float(np.max(np.abs(arrays[k][:300] - reference[k][:300])))
        for k in POSITION_FIELDS
    )
    checks = {
        "fallback_positions_within_1um": error <= 1e-6,
        "fallback_reward_exact": rewards[5] == rewards[7] == reference_rewards[6],
        "prefix_replay_within_1um": prefix_error <= 1e-6,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "fallback_error_m": error,
        "prefix_error_m": prefix_error,
    }
