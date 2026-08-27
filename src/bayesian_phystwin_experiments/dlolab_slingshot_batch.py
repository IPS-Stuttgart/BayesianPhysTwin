"""Qualification of native eight-environment execution against isolated runs."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dlolab_benchmark import RIGID_FIELDS, native_memory, slingshot_actions
from .dlolab_native import STATE_FIELDS, array_digest
from .dlolab_slingshot_controls import native_reward_from_trace
from .dlolab_slingshot_process import observe

BATCH_INDICES = (0, 1, 1, 0, 1, 0, 1, 1)
TRACE_NAMES = (
    "rod_pos_m",
    "rod_vel_m_s",
    "sphere_pos_m",
    "sphere_vel_m_s",
    "cube_pos_m",
    "cube_vel_m_s",
    "gripper_pos_m",
    "robot_qpos",
)
MEMORY_NAMES = tuple(f"memory_RigidSolverState.{x}" for x in RIGID_FIELDS) + tuple(
    f"memory_RODSolverState.{x}" for x in STATE_FIELDS
)


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-batch-qualification-v1",
        "environment_count": 8,
        "action_indices": list(BATCH_INDICES),
        "native_steps": 900,
        "fresh_process_per_batch": True,
        "native_physics_reward_controller_unchanged": True,
        "reference": "frozen_fresh_process_zero_and_pull",
        "position_atol_m": 1e-6,
        "memory_rtol": 1e-6,
        "memory_atol": 1e-9,
        "fixed_endpoint_atol_m": 1e-9,
        "exact_native_reward_required": True,
        "retry_authorized": False,
        "method_evaluation_authorized": False,
        "protected_data_read": False,
        "gpu_work": False,
    }


def split_batch(
    arrays: dict[str, np.ndarray], count: int
) -> list[dict[str, np.ndarray]]:
    expected = set(TRACE_NAMES + MEMORY_NAMES + ("controls", "joint_targets"))
    if set(arrays) != expected or count != 8:
        raise ValueError("complete registered batch layout required")
    rows: list[dict[str, np.ndarray]] = [dict() for _ in range(count)]
    for name, value in arrays.items():
        axis = 1 if name in TRACE_NAMES else 0
        if (
            value.ndim <= axis
            or value.shape[axis] != count
            or (axis == 1 and value.shape[0] != 900)
        ):
            raise ValueError("batch environment/time axis changed")
        if value.dtype.kind not in "bifu" or not np.isfinite(value).all():
            raise ValueError("invalid native batch array")
        for index in range(count):
            rows[index][name] = np.take(value, [index], axis=axis)
    return rows


def run_batch(
    upstream: Path, output: Path, controls: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if (
        controls.shape != (8, 3, 6)
        or controls.dtype != np.float64
        or not np.isfinite(controls).all()
    ):
        raise ValueError("native batch control must have shape (8,3,6) float64")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    gs: Any = importlib.import_module("genesis")
    import torch
    from envs.env_slingshot import Train_Env_Slingshot
    from omegaconf import DictConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    started = time.monotonic()
    try:
        gs.init(
            seed=0,
            precision="64",
            logging_level="error",
            backend=gs.cpu,
            performance_mode=True,
            theme="dumb",
        )
        env = Train_Env_Slingshot(
            DictConfig(
                {
                    "task": "slingshot",
                    "log_dir": str(output / "native-log"),
                    "n_envs": 8,
                    "GUI": False,
                    "camera": False,
                    "raytracer": False,
                    "requires_grad": False,
                }
            )
        )
        env.init_cmaes_env(n_steps_sub=10)
        trace: list[dict[str, np.ndarray]] = []
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            value = original_step(*args, **kwargs)
            trace.append(observe(env))
            return value

        env.scene.step = step
        native = env.eval_traj(controls.copy())
        env.scene.step = original_step
        if len(trace) != 900:
            raise ValueError("native step count changed")
        arrays = {name: np.stack([row[name] for row in trace]) for name in trace[0]}
        arrays.update(controls=controls.copy(), joint_targets=env.qpos_seq.copy())
        arrays.update(
            {
                f"memory_{name}": value
                for name, value in native_memory(env.scene.get_state()).items()
            }
        )
        return arrays, {
            "native_cumulative_reward": np.asarray(native["cum_reward"]).tolist(),
            "native_forward_seconds": float(native["forward_time"]),
            "wall_seconds": time.monotonic() - started,
            "native_steps": len(trace),
        }
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()


def compare(
    rows: list[dict[str, np.ndarray]],
    references: list[dict[str, np.ndarray]],
    rewards: list[float],
) -> dict[str, Any]:
    if len(rows) != 8 or len(references) != 2 or len(rewards) != 8:
        raise ValueError("the complete registered comparison is required")
    values = []
    for index, row in enumerate(rows):
        reference = references[BATCH_INDICES[index]]
        if set(row) != set(reference):
            raise ValueError("scalar/batch member mismatch")
        if array_digest(row["controls"]) != array_digest(
            slingshot_actions()[BATCH_INDICES[index]][None]
        ):
            raise ValueError("batch qualification controls changed")
        for name in row:
            if (
                row[name].shape != reference[name].shape
                or row[name].dtype != reference[name].dtype
            ):
                raise ValueError("scalar/batch layout mismatch")
        position_error = max(
            float(np.max(np.abs(row[name] - reference[name])))
            for name in ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
        )
        memory_ok = all(
            np.allclose(row[name], reference[name], rtol=1e-6, atol=1e-9)
            for name in MEMORY_NAMES
        )
        failed_memory = [
            name
            for name in MEMORY_NAMES
            if not np.allclose(row[name], reference[name], rtol=1e-6, atol=1e-9)
        ]
        fixed_error = max(
            float(
                np.max(
                    np.abs(
                        row["rod_pos_m"][:, 0, node]
                        - reference["rod_pos_m"][0, 0, node]
                    )
                )
            )
            for node in (0, 1, 10, 11)
        )
        reward_ok = (
            rewards[index]
            == native_reward_from_trace(row["cube_pos_m"])
            == native_reward_from_trace(reference["cube_pos_m"])
        )
        values.append(
            {
                "index": index,
                "action_index": BATCH_INDICES[index],
                "maximum_position_error_m": position_error,
                "memory_within_tolerance": bool(memory_ok),
                "failed_memory_fields": failed_memory,
                "maximum_fixed_endpoint_error_m": fixed_error,
                "exact_native_reward": reward_ok,
                "passed": position_error <= 1e-6
                and bool(memory_ok)
                and fixed_error <= 1e-9
                and reward_ok,
            }
        )
    return {
        "rows": values,
        "batch_qualification_passed": all(row["passed"] for row in values),
        "method_evaluation_authorized": False,
        "protected_data_read": False,
    }
