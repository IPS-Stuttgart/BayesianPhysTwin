"""Native Slingshot execution with a new interpreter for every rollout."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dlolab_benchmark import (
    fixed_endpoint_error,
    memory_comparison,
    native_memory,
    slingshot_actions,
)
from .dlolab_benchmark import (
    protocol as reused_reset_protocol,
)
from .dlolab_native import array_digest, file_digest
from .dlolab_regret_artifacts import runtime_identity


def protocol() -> dict[str, Any]:
    value = reused_reset_protocol()
    value.update(
        schema="dlolab-slingshot-fresh-process-qualification-v2",
        reset_contract="new_python_process_per_rollout",
        retained_parent_result_id="d3631c30851c0efe8436a1bfcdf388b9ce7e4d46df46534c3a60ba0351a3daae",
        automatic_method_evaluation_authorized=False,
    )
    return value


def runtime() -> dict[str, Any]:
    value = runtime_identity()
    value["benchmark_packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "pin",
            "pin-pink",
            "qpsolvers",
            "proxsuite",
            "quadprog",
            "mushroom-rl",
            "omegaconf",
        )
    }
    return value


def load_native_bundle(
    directory: Path, manifest: dict[str, Any]
) -> dict[str, np.ndarray]:
    if (
        set(manifest) != {"file", "file_sha256", "arrays"}
        or manifest["file"] != "arrays.npz"
    ):
        raise ValueError("invalid native bundle manifest")
    path = directory / "arrays.npz"
    if path.is_symlink() or file_digest(path) != manifest["file_sha256"]:
        raise ValueError("native bundle changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["arrays"]) or len(archive.files) != len(
            set(archive.files)
        ):
            raise ValueError("native bundle member set changed")
        result = {
            name: np.array(archive[name], order="C", copy=True)
            for name in archive.files
        }
    for name, value in result.items():
        if (
            value.dtype.kind not in "bifu"
            or not np.isfinite(value).all()
            or array_digest(value) != manifest["arrays"][name]
        ):
            raise ValueError("invalid or changed native array")
    return result


def observe(env: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in {
        "rod_pos_m": env.rope.get_all_verts_tc(),
        "rod_vel_m_s": env.rope.get_all_vels_tc(),
        "sphere_pos_m": env.sphere.get_pos(),
        "sphere_vel_m_s": env.sphere.get_vel(),
        "cube_pos_m": env.cube.get_pos(),
        "cube_vel_m_s": env.cube.get_vel(),
        "gripper_pos_m": env._ef1.get_pos(),
        "robot_qpos": env.franka1.get_qpos(),
    }.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result[name] = np.array(value, order="C", copy=True)
        if not np.isfinite(result[name]).all():
            raise ValueError(f"nonfinite native observation: {name}")
    return result


def run_native(
    upstream: Path, output: Path, controls: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Call the official task/controller/reward once, with read-only tracing."""
    if (
        controls.shape != (1, 3, 6)
        or controls.dtype != np.float64
        or not np.isfinite(controls).all()
    ):
        raise ValueError("native control must be a finite float64 (1,3,6) array")
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
                    "n_envs": 1,
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
        arrays["controls"] = controls.copy()
        arrays["joint_targets"] = env.qpos_seq.copy()
        arrays.update(
            {
                f"memory_{name}": value
                for name, value in native_memory(env.scene.get_state()).items()
            }
        )
        return arrays, {
            "native_steps": len(trace),
            "native_cumulative_reward": np.asarray(native["cum_reward"]).tolist(),
            "native_forward_seconds": float(native["forward_time"]),
            "wall_seconds": time.monotonic() - started,
        }
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()


def qualify(arrays: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if len(arrays) != 3:
        raise ValueError("all three registered rollouts are required")
    for index, row in enumerate(arrays):
        if array_digest(row["controls"]) != array_digest(
            slingshot_actions()[[0, 1, 1][index]][None]
        ):
            raise ValueError("registered action changed")
        if row["rod_pos_m"].shape != (900, 1, 12, 3):
            raise ValueError("native rollout layout changed")
    replay_error = max(
        float(np.max(np.abs(arrays[1][k] - arrays[2][k])))
        for k in (
            "rod_pos_m",
            "sphere_pos_m",
            "cube_pos_m",
            "gripper_pos_m",
        )
    )
    memory = memory_comparison(
        {k: v for k, v in arrays[1].items() if k.startswith("memory_")},
        {k: v for k, v in arrays[2].items() if k.startswith("memory_")},
    )
    if memory["field_count"] != 23:
        raise ValueError("complete native memory is required")
    fixed_error = fixed_endpoint_error([row["rod_pos_m"] for row in arrays])
    gripper_motion = float(
        np.linalg.norm(arrays[1]["gripper_pos_m"][699] - arrays[1]["gripper_pos_m"][99])
    )
    band_motion = float(
        np.max(
            np.linalg.norm(
                arrays[1]["rod_pos_m"][699] - arrays[1]["rod_pos_m"][99], axis=-1
            )
        )
    )
    checks = {
        "all_three_native_rollouts_complete": True,
        "finite_native_arrays": all(
            np.isfinite(v).all() for row in arrays for v in row.values()
        ),
        "gripper_motion_at_least_10mm": gripper_motion >= 0.01,
        "band_motion_at_least_10mm": band_motion >= 0.01,
        "fixed_endpoints_unchanged": fixed_error <= 1e-9,
        "position_replay_within_1um": replay_error <= 1e-6,
        "memory_replay_within_tolerance": memory["within_tolerance"],
    }
    return {
        "checks": checks,
        "native_qualification_passed": all(checks.values()),
        "maximum_position_replay_error_m": replay_error,
        "maximum_fixed_endpoint_error_m": fixed_error,
        "gripper_motion_m": gripper_motion,
        "band_motion_m": band_motion,
        "memory_replay": memory,
        "method_evaluation_authorized": False,
        "protected_data_read": False,
    }
