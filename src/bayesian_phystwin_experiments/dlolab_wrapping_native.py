"""CPU-only observation of the unchanged native loop-wrapping environment."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dlolab_benchmark import native_memory
from .dlolab_wrapping_source import N_ENVS, NATIVE_STEPS, action_bank, validate_world


def observe(env: Any) -> dict[str, np.ndarray]:
    def host(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.array(value, order="C", copy=True)
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite native wrapping observable")
        return result

    return {
        "rod_pos_m": host(env.rope.get_all_verts()),
        "rod_vel_m_s": host(env.rope.get_all_vels()),
        "post_pos_m": np.stack(
            [host(p.get_pos()) for p in (env.post1, env.post2, env.post3)], axis=1
        ),
        "gripper_pos_m": np.stack(
            [host(c.ef.get_pos()) for c in (env.c1, env.c2)], axis=1
        ),
        "robot_qpos": np.concatenate(
            [host(f.get_qpos()) for f in (env.franka1, env.franka2)], axis=1
        ),
    }


def run_world(
    upstream: Path, output: Path, world: dict
) -> tuple[dict[str, np.ndarray], dict]:
    validate_world(world)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    import genesis as gs
    import torch
    from envs.env_wrapping import Train_Env_Wrapping
    from omegaconf import DictConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    captured: dict[str, list[float]] = {}

    class RegisteredWorld(Train_Env_Wrapping):
        def _set_parameter(self, objects, envs_idx, name, key):
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full native reset required")
            field = getattr(self.scene.sim.rod_solver, f"rods_{name}_stiffness")
            values = torch.full((N_ENVS,), float(world[key]), dtype=gs.tc_float)
            getattr(self.rope, f"set_{name}_stiffness")(values)
            actual = field.to_numpy()
            if actual.shape != (1, N_ENVS) or not np.all(actual[0] == world[key]):
                raise ValueError("native material realization changed")
            captured[name] = actual[0].tolist()

        def _randomize_bending_stiffness(self, objects, envs_idx=None):
            self._set_parameter(objects, envs_idx, "bending", "bending_E")

        def _randomize_stretching_stiffness(self, objects, envs_idx=None):
            self._set_parameter(objects, envs_idx, "stretching", "stretching_K")

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
                    "task": "wrapping",
                    "log_dir": str(output / "native-log"),
                    "n_envs": N_ENVS,
                    "GUI": False,
                    "camera": False,
                    "raytracer": False,
                    "requires_grad": False,
                }
            )
        )
        if (
            env.rope.n_vertices != 50
            or env.rope.material.use_inextensible is not False
            or env.cameras
        ):
            raise ValueError("native extensible-loop configuration changed")
        env.init_cmaes_env(n_steps_sub=10)
        trace: list[dict[str, np.ndarray]] = []
        initial: list[np.ndarray] = []
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            if len(trace) >= NATIVE_STEPS:
                raise RuntimeError("native horizon exceeded")
            if not trace:
                initial.append(np.array(env.rope.get_all_verts(), order="C", copy=True))
            result = original_step(*args, **kwargs)
            trace.append(observe(env))
            return result

        env.scene.step = step
        try:
            native = env.eval_traj(action_bank())
        finally:
            env.scene.step = original_step
        if (
            len(trace) != NATIVE_STEPS
            or set(captured) != {"bending", "stretching"}
            or len(initial) != 1
        ):
            raise RuntimeError("incomplete native wrapping execution")
        if not np.all(env.scene.sim.rod_solver.rods_twisting_stiffness.to_numpy() == 0):
            raise ValueError("native zero twisting stiffness changed")
        arrays = {name: np.stack([row[name] for row in trace]) for name in trace[0]}
        arrays.update(
            controls=action_bank(),
            joint_targets=env.qpos_seq.copy(),
            initial_rod_pos_m=initial[0],
        )
        arrays.update(
            {
                f"memory_{name}": value
                for name, value in native_memory(env.scene.get_state()).items()
            }
        )
        return arrays, {
            "native_steps": len(trace),
            "native_final_reward": np.asarray(native["final_reward"]).tolist(),
            "native_cumulative_reward": np.asarray(native["cum_reward"]).tolist(),
            "native_forward_seconds": float(native["forward_time"]),
            "wall_seconds": time.monotonic() - started,
            "world": world,
            "world_realization": captured,
            "twisting_stiffness_zero_preserved": True,
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
        }
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()
