"""CPU-only execution of the unchanged public wiring-post environment."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dlolab_benchmark import native_memory
from .dlolab_wiring_source import NATIVE_STEPS, action_bank, worlds


def observe(env: Any) -> dict[str, np.ndarray]:
    def host(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.array(value, order="C", copy=True)
        if not np.isfinite(array).all():
            raise RuntimeError("nonfinite native wiring observable")
        return array

    return {
        "rod_pos_m": host(env.rope.get_all_verts()),
        "rod_vel_m_s": host(env.rope.get_all_vels()),
        "post_pos_m": np.stack(
            [host(env.stick1.get_pos()), host(env.stick2.get_pos())], axis=1
        ),
        "hidden_post_pos_m": np.stack(
            [
                host(env.stick1_hidden.get_all_verts()),
                host(env.stick2_hidden.get_all_verts()),
            ],
            axis=1,
        ),
        "gripper_pos_m": host(env.c1.ef.get_pos()),
        "robot_qpos": host(env.franka1.get_qpos()),
    }


def run_world(
    upstream: Path, output: Path, world: dict
) -> tuple[dict[str, np.ndarray], dict]:
    if world not in worlds() or world != worlds()[int(world["index"])]:
        raise ValueError("unregistered wiring material world")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    import genesis as gs
    import torch
    from envs.env_wiring_post import Train_Env_Wiring_post
    from omegaconf import DictConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    captured: dict[str, list[float]] = {}

    class RegisteredWorld(Train_Env_Wiring_post):
        def _set_parameter(self, objects, envs_idx, name, key):
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full native reset required")
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
                raise ValueError("native material realization changed")
            captured[name] = actual[0].tolist()

        def _randomize_bending_stiffness(self, objects, envs_idx=None):
            self._set_parameter(objects, envs_idx, "bending", "bending_E")

        def _randomize_twisting_stiffness(self, objects, envs_idx=None):
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
                    "task": "wiring_post",
                    "log_dir": str(output / "native-log"),
                    "n_envs": 8,
                    "GUI": False,
                    "camera": False,
                    "raytracer": False,
                    "requires_grad": False,
                }
            )
        )
        if env.rope.material.use_inextensible is not True:
            raise ValueError("native inextensible configuration changed")
        env.init_cmaes_env(n_steps_sub=10)
        trace: list[dict[str, np.ndarray]] = []
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            if len(trace) >= NATIVE_STEPS:
                raise RuntimeError("native horizon exceeded")
            result = original_step(*args, **kwargs)
            trace.append(observe(env))
            return result

        env.scene.step = step
        try:
            native = env.eval_traj(action_bank())
        finally:
            env.scene.step = original_step
        if len(trace) != NATIVE_STEPS or set(captured) != {"bending", "twisting"}:
            raise RuntimeError("incomplete native wiring execution")
        arrays = {name: np.stack([row[name] for row in trace]) for name in trace[0]}
        arrays.update(
            controls=action_bank(),
            joint_targets=env.qpos_seq.copy(),
            target_pos_m=env.target_pos.copy(),
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
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
        }
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()
