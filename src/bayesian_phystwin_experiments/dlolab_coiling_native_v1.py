"""CPU execution of the unchanged public DLO-Lab coiling environment."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_benchmark import native_memory
from .dlolab_coiling_query_competence_v1 import (
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
        raise RuntimeError("nonfinite native coiling observable")
    return result


def run_world(
    upstream: Path, output: Path, world: dict[str, Any]
) -> tuple[dict[str, Array], dict[str, Any]]:
    if world not in worlds() or world != worlds()[int(world["index"])]:
        raise ValueError("unregistered coiling material world")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    import genesis as gs
    import torch
    from envs.env_coiling import Train_Env_Coiling
    from omegaconf import DictConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    captured: dict[str, list[float]] = {}

    class RegisteredWorld(Train_Env_Coiling):  # type: ignore[misc]
        def _set_parameter(
            self,
            objects: list[Any],
            envs_idx: Any,
            name: str,
            key: str,
        ) -> None:
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full native coiling reset required")
            field = getattr(self.scene.sim.rod_solver, f"rods_{name}_stiffness")
            before = field.to_numpy().copy()
            values = torch.full((8,), float(world[key]), dtype=gs.tc_float)
            getattr(self.rope, f"set_{name}_stiffness")(values)
            actual = field.to_numpy()
            if (
                actual.shape != (3, 8)
                or not np.all(actual[0] == world[key])
                or not np.array_equal(actual[1:], before[1:])
            ):
                raise ValueError("native coiling material realization changed")
            captured[name] = actual[0].tolist()

        def _randomize_bending_stiffness(
            self, objects: list[Any], envs_idx: Any = None
        ) -> None:
            self._set_parameter(objects, envs_idx, "bending", "bending_E")

        def _randomize_twisting_stiffness(
            self, objects: list[Any], envs_idx: Any = None
        ) -> None:
            self._set_parameter(objects, envs_idx, "twisting", "twisting_G")

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
                    "task": "coiling",
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
        prefix: list[Array] = []
        measurements = {
            "maximum_common_prefix_error_m": 0.0,
            "maximum_duplicate_coordinate_error_m": 0.0,
            "maximum_segment_relative_error": 0.0,
            "minimum_rod_height_m": float("inf"),
            "maximum_attachment_distance_m": 0.0,
            "maximum_fixed_cone_error_m": 0.0,
        }
        step_index = 0
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            nonlocal step_index
            if step_index >= NATIVE_STEPS:
                raise RuntimeError("native coiling horizon exceeded")
            result = original_step(*args, **kwargs)
            positions = _host(env.rope.get_all_verts())
            gripper = _host(env.c1.ef.get_pos())
            cone = _host(env.cone1.get_pos())
            if positions.shape != (8, 60, 3) or gripper.shape != (8, 3):
                raise RuntimeError("native coiling observable shape changed")
            if step_index < PREFIX_STEPS:
                measurements["maximum_common_prefix_error_m"] = max(
                    measurements["maximum_common_prefix_error_m"],
                    float(np.max(np.abs(positions - positions[:1]))),
                )
            measurements["maximum_duplicate_coordinate_error_m"] = max(
                measurements["maximum_duplicate_coordinate_error_m"],
                float(np.max(np.abs(positions[1] - positions[7]))),
            )
            segment = np.linalg.norm(np.diff(positions, axis=1), axis=-1) / 0.02
            measurements["maximum_segment_relative_error"] = max(
                measurements["maximum_segment_relative_error"],
                float(np.max(np.abs(segment - 1))),
            )
            measurements["minimum_rod_height_m"] = min(
                measurements["minimum_rod_height_m"],
                float(positions[..., 2].min()),
            )
            measurements["maximum_attachment_distance_m"] = max(
                measurements["maximum_attachment_distance_m"],
                float(np.linalg.norm(positions[:, 1] - gripper, axis=-1).max()),
            )
            measurements["maximum_fixed_cone_error_m"] = max(
                measurements["maximum_fixed_cone_error_m"],
                float(np.max(np.abs(cone - [0.0, 0.0, 0.15]))),
            )
            if step_index in PREFIX_FRAMES:
                prefix.append(positions[:, list(OBSERVED_NODES)].copy())
            step_index += 1
            return result

        env.scene.step = step
        try:
            native = env.eval_traj(action_bank())
        finally:
            env.scene.step = original_step
        if (
            step_index != NATIVE_STEPS
            or len(prefix) != len(PREFIX_FRAMES)
            or set(captured) != {"bending", "twisting"}
        ):
            raise RuntimeError("incomplete native coiling execution")
        final_positions = _host(env.rope.get_all_verts())
        arrays = {
            "controls": action_bank(),
            "prefix_positions_m": np.stack(prefix),
            "final_positions_m": final_positions,
            "final_velocities_m_s": _host(env.rope.get_all_vels()),
            "final_gripper_positions_m": _host(env.c1.ef.get_pos()),
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
            "world_realization": captured,
            "measurements": measurements,
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
        }
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()
