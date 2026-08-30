"""CPU execution of registered DLO-Lab unknotting development worlds."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_benchmark import native_memory
from .dlolab_unknotting_headroom_v1 import (
    CONTROL_NODES,
    NATIVE_STEPS,
    OBSERVED_NODES,
    PREFIX_FRAMES,
    PREFIX_STEPS,
    action_bank,
    worlds,
)

Array: TypeAlias = NDArray[Any]


def _host(value: Any) -> Array:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result: Array = np.array(value, order="C", copy=True)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite native unknotting observable")
    return result


def _rotate_xy(points: Array, center: Array, angle_deg: float) -> Array:
    value = np.asarray(points, dtype=np.float64)
    pivot = np.asarray(center, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 3 or pivot.shape != (len(value), 1, 3):
        raise ValueError("batched xyz points and centers required")
    angle = np.deg2rad(angle_deg)
    offset = value - pivot
    result = value.copy()
    result[..., 0] = (
        pivot[..., 0] + np.cos(angle) * offset[..., 0] - np.sin(angle) * offset[..., 1]
    )
    result[..., 1] = (
        pivot[..., 1] + np.sin(angle) * offset[..., 0] + np.cos(angle) * offset[..., 1]
    )
    return result


def run_world(
    upstream: Path, output: Path, world: dict[str, Any]
) -> tuple[dict[str, Array], dict[str, Any]]:
    registered = worlds()
    if world not in registered or world != registered[int(world["index"])]:
        raise ValueError("unregistered unknotting development world")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    import genesis as gs
    import torch
    from envs.env_unknotting import Train_Env_Unknotting
    from omegaconf import DictConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    realization: dict[str, Any] = {}

    class RegisteredWorld(Train_Env_Unknotting):  # type: ignore[misc]
        def _randomize_positions(
            self, objects: list[Any], envs_idx: Any = None
        ) -> None:
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full single-rope native reset required")
            before = _host(self.rope.get_all_verts())
            if before.shape != (11, 50, 3):
                raise ValueError("native unknotting geometry changed")
            center = before.mean(axis=1, keepdims=True)
            expected = _rotate_xy(before, center, float(world["rotation_deg"]))
            self.rope.set_pos(0, torch.as_tensor(expected, dtype=gs.tc_float))
            actual = _host(self.rope.get_all_verts())
            realization.update(
                {
                    "rotation_deg": float(world["rotation_deg"]),
                    "maximum_rotation_realization_error_m": float(
                        np.max(np.abs(actual - expected))
                    ),
                    "centroid_before_m": center[0, 0].tolist(),
                    "centroid_after_m": actual.mean(axis=1)[0].tolist(),
                }
            )

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
        env = RegisteredWorld(
            DictConfig(
                {
                    "task": "unknotting",
                    "log_dir": str(output / "native-log"),
                    "n_envs": 11,
                    "GUI": False,
                    "camera": False,
                    "raytracer": False,
                    "requires_grad": False,
                    "n_substeps_per_step": 200,
                }
            )
        )
        env.init_cmaes_env(n_steps_sub=10)
        if tuple(env.control_idx) != CONTROL_NODES:
            raise RuntimeError("native unknotting control-node contract changed")
        prefix: list[Array] = []
        measurements = {
            "maximum_common_prefix_error_m": 0.0,
            "maximum_duplicate_coordinate_error_m": 0.0,
            "maximum_segment_relative_error": 0.0,
            "minimum_rod_height_m": float("inf"),
            "maximum_attachment_offset_drift_m": 0.0,
        }
        initial_segments: Array | None = None
        initial_attachment_offsets: tuple[Array, Array] | None = None
        step_index = 0
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            nonlocal initial_attachment_offsets, initial_segments, step_index
            if step_index >= NATIVE_STEPS:
                raise RuntimeError("native unknotting horizon exceeded")
            if initial_segments is None or initial_attachment_offsets is None:
                initial_rope = _host(env.rope.get_all_verts())
                initial_segments = np.linalg.norm(
                    np.diff(initial_rope, axis=1), axis=-1
                )
                initial_attachment_offsets = (
                    initial_rope[:, CONTROL_NODES[0]] - _host(env.c1.ef.get_pos()),
                    initial_rope[:, CONTROL_NODES[1]] - _host(env.c2.ef.get_pos()),
                )
            result = original_step(*args, **kwargs)
            rope = _host(env.rope.get_all_verts())
            gripper_a = _host(env.c1.ef.get_pos())
            gripper_b = _host(env.c2.ef.get_pos())
            if rope.shape != (11, 50, 3):
                raise RuntimeError("native unknotting observable shape changed")
            segments = np.linalg.norm(np.diff(rope, axis=1), axis=-1)
            relative = np.abs(segments / initial_segments - 1)
            measurements["maximum_segment_relative_error"] = max(
                measurements["maximum_segment_relative_error"],
                float(np.max(relative)),
            )
            if step_index < PREFIX_STEPS:
                measurements["maximum_common_prefix_error_m"] = max(
                    measurements["maximum_common_prefix_error_m"],
                    float(np.max(np.abs(rope - rope[:1]))),
                )
            measurements["maximum_duplicate_coordinate_error_m"] = max(
                measurements["maximum_duplicate_coordinate_error_m"],
                float(np.max(np.abs(rope[1] - rope[9]))),
                float(np.max(np.abs(rope[8] - rope[10]))),
            )
            measurements["minimum_rod_height_m"] = min(
                measurements["minimum_rod_height_m"],
                float(rope[..., 2].min()),
            )
            offsets = (
                rope[:, CONTROL_NODES[0]] - gripper_a,
                rope[:, CONTROL_NODES[1]] - gripper_b,
            )
            drift_a = float(
                np.linalg.norm(
                    offsets[0] - initial_attachment_offsets[0], axis=-1
                ).max()
            )
            drift_b = float(
                np.linalg.norm(
                    offsets[1] - initial_attachment_offsets[1], axis=-1
                ).max()
            )
            measurements["maximum_attachment_offset_drift_m"] = max(
                measurements["maximum_attachment_offset_drift_m"],
                drift_a,
                drift_b,
            )
            if step_index in PREFIX_FRAMES:
                prefix.append(rope[:, list(OBSERVED_NODES)].copy())
            step_index += 1
            return result

        env.scene.step = step
        try:
            native = env.eval_traj(action_bank())
        finally:
            env.scene.step = original_step
        if step_index != NATIVE_STEPS or len(prefix) != len(PREFIX_FRAMES):
            raise RuntimeError("incomplete native unknotting execution")
        arrays = {
            "controls": action_bank(),
            "prefix_rope_m": np.stack(prefix),
            "final_rope_m": _host(env.rope.get_all_verts()),
            "final_rope_velocity_m_s": _host(env.rope.get_all_vels()),
            "final_gripper_a_m": _host(env.c1.ef.get_pos()),
            "final_gripper_b_m": _host(env.c2.ef.get_pos()),
            "joint_targets": np.asarray(env.qpos_seq, dtype=np.float32),
        }
        arrays.update(
            {
                f"memory_{name}": value
                for name, value in native_memory(env.scene.get_state()).items()
            }
        )
        return arrays, {
            "native_steps": step_index,
            "native_final_reward": np.asarray(native["final_reward"]).tolist(),
            "native_cumulative_reward": np.asarray(native["cum_reward"]).tolist(),
            "native_forward_seconds": float(native["forward_time"]),
            "wall_seconds": time.monotonic() - started,
            "world": world,
            "world_realization": realization,
            "measurements": measurements,
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
        }
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()
