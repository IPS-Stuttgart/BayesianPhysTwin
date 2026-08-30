"""CPU execution of registered DLO-Lab separation development worlds."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_benchmark import native_memory
from .dlolab_separation_headroom_v1 import (
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
        raise RuntimeError("nonfinite native separation observable")
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
        raise ValueError("unregistered separation development world")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    import genesis as gs
    import torch
    from envs.env_separation import Train_Env_Separation
    from omegaconf import DictConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    realization: dict[str, Any] = {}

    class RegisteredWorld(Train_Env_Separation):  # type: ignore[misc]
        def _randomize_positions(
            self, objects: list[Any], envs_idx: Any = None
        ) -> None:
            if envs_idx is not None or objects != [self.rope, self.rope2]:
                raise ValueError("full two-rope native reset required")
            before_a = _host(self.rope.get_all_verts())
            before_b = _host(self.rope2.get_all_verts())
            if before_a.shape != (11, 30, 3) or before_b.shape != (11, 30, 3):
                raise ValueError("native separation geometry changed")
            center = np.concatenate([before_a, before_b], axis=1).mean(
                axis=1, keepdims=True
            )
            expected_a = _rotate_xy(before_a, center, float(world["rotation_deg"]))
            expected_b = _rotate_xy(before_b, center, float(world["rotation_deg"]))
            self.rope.set_pos(0, torch.as_tensor(expected_a, dtype=gs.tc_float))
            self.rope2.set_pos(0, torch.as_tensor(expected_b, dtype=gs.tc_float))
            actual_a = _host(self.rope.get_all_verts())
            actual_b = _host(self.rope2.get_all_verts())
            error = float(
                max(
                    np.max(np.abs(actual_a - expected_a)),
                    np.max(np.abs(actual_b - expected_b)),
                )
            )
            realization.update(
                {
                    "rotation_deg": float(world["rotation_deg"]),
                    "maximum_rotation_realization_error_m": error,
                    "shared_centroid_before_m": center[0, 0].tolist(),
                    "shared_centroid_after_m": np.concatenate(
                        [actual_a, actual_b], axis=1
                    )
                    .mean(axis=1)[0]
                    .tolist(),
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
                    "task": "separation",
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
        prefix_a: list[Array] = []
        prefix_b: list[Array] = []
        measurements = {
            "maximum_common_prefix_error_m": 0.0,
            "maximum_duplicate_coordinate_error_m": 0.0,
            "maximum_segment_relative_error": 0.0,
            "minimum_rod_height_m": float("inf"),
            "maximum_attachment_distance_m": 0.0,
        }
        initial_segments: dict[str, Array] = {}
        step_index = 0
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            nonlocal step_index
            if step_index >= NATIVE_STEPS:
                raise RuntimeError("native separation horizon exceeded")
            result = original_step(*args, **kwargs)
            a = _host(env.rope.get_all_verts())
            b = _host(env.rope2.get_all_verts())
            gripper_a = _host(env.c1.ef.get_pos())
            gripper_b = _host(env.c2.ef.get_pos())
            if a.shape != (11, 30, 3) or b.shape != (11, 30, 3):
                raise RuntimeError("native separation observable shape changed")
            if not initial_segments:
                initial_segments["a"] = np.linalg.norm(np.diff(a, axis=1), axis=-1)
                initial_segments["b"] = np.linalg.norm(np.diff(b, axis=1), axis=-1)
            if step_index < PREFIX_STEPS:
                measurements["maximum_common_prefix_error_m"] = max(
                    measurements["maximum_common_prefix_error_m"],
                    float(np.max(np.abs(a - a[:1]))),
                    float(np.max(np.abs(b - b[:1]))),
                )
            measurements["maximum_duplicate_coordinate_error_m"] = max(
                measurements["maximum_duplicate_coordinate_error_m"],
                float(np.max(np.abs(a[1] - a[9]))),
                float(np.max(np.abs(b[1] - b[9]))),
                float(np.max(np.abs(a[8] - a[10]))),
                float(np.max(np.abs(b[8] - b[10]))),
            )
            for key, points in (("a", a), ("b", b)):
                segment = np.linalg.norm(np.diff(points, axis=1), axis=-1)
                relative = np.abs(segment / initial_segments[key] - 1)
                measurements["maximum_segment_relative_error"] = max(
                    measurements["maximum_segment_relative_error"],
                    float(np.max(relative)),
                )
            measurements["minimum_rod_height_m"] = min(
                measurements["minimum_rod_height_m"],
                float(min(a[..., 2].min(), b[..., 2].min())),
            )
            measurements["maximum_attachment_distance_m"] = max(
                measurements["maximum_attachment_distance_m"],
                float(np.linalg.norm(a[:, 27] - gripper_a, axis=-1).max()),
                float(np.linalg.norm(b[:, 2] - gripper_b, axis=-1).max()),
            )
            if step_index in PREFIX_FRAMES:
                prefix_a.append(a[:, list(OBSERVED_NODES)].copy())
                prefix_b.append(b[:, list(OBSERVED_NODES)].copy())
            step_index += 1
            return result

        env.scene.step = step
        try:
            native = env.eval_traj(action_bank())
        finally:
            env.scene.step = original_step
        if step_index != NATIVE_STEPS or len(prefix_a) != len(PREFIX_FRAMES):
            raise RuntimeError("incomplete native separation execution")
        arrays = {
            "controls": action_bank(),
            "prefix_rope_a_m": np.stack(prefix_a),
            "prefix_rope_b_m": np.stack(prefix_b),
            "final_rope_a_m": _host(env.rope.get_all_verts()),
            "final_rope_b_m": _host(env.rope2.get_all_verts()),
            "final_rope_a_velocity_m_s": _host(env.rope.get_all_vels()),
            "final_rope_b_velocity_m_s": _host(env.rope2.get_all_vels()),
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
            "native_final_reward_m": np.asarray(native["final_reward"]).tolist(),
            "native_cumulative_reward_m": np.asarray(native["cum_reward"]).tolist(),
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
