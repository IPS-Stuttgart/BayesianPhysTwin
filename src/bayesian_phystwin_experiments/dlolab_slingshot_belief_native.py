"""Fresh native Slingshot worlds and a hard causal-prefix stop, CPU only."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .dlolab_slingshot_batch import run_batch
from .dlolab_slingshot_process import observe

PREFIX_STEPS = 300


class PrefixComplete(Exception):
    """Internal control flow: the allowed prefix has completed, not a failure."""


class PrefixTrace:
    def __init__(self) -> None:
        self.rows: list[dict[str, np.ndarray]] = []

    def append(self, row: dict[str, np.ndarray]) -> None:
        if len(self.rows) >= PREFIX_STEPS:
            raise ValueError("native execution crossed the prefix boundary")
        self.rows.append(row)
        if len(self.rows) == PREFIX_STEPS:
            raise PrefixComplete

    def arrays(self) -> dict[str, np.ndarray]:
        if len(self.rows) != PREFIX_STEPS:
            raise ValueError("incomplete causal prefix")
        return {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}


def run_prefix(upstream: Path, output: Path, controls: np.ndarray):
    """Stop after frame 299, before even entering the second native action."""
    if controls.shape != (8, 3, 6) or controls.dtype != np.float64:
        raise ValueError("exact native batch controls required")
    import genesis as gs
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
        trace = PrefixTrace()
        original = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            trace.append(observe(env))
            return result

        env.scene.step = step
        try:
            env.eval_traj(controls.copy())
        except PrefixComplete:
            pass
        finally:
            env.scene.step = original
        arrays = trace.arrays()
        arrays["controls"] = controls.copy()
        return arrays, {
            "native_steps": PREFIX_STEPS,
            "future_simulated": False,
            "reward_scored": False,
            "hidden_state_restart": False,
            "wall_seconds": time.monotonic() - started,
        }
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()


def run_registered_worlds(
    upstream: Path,
    output: Path,
    controls: np.ndarray,
    worlds: list[dict],
    *,
    prefix_only: bool,
):
    if len(worlds) != 8 or controls.shape != (8, 3, 6):
        raise ValueError("exactly eight native slots required")
    parameters = np.asarray(
        [
            [row[k] for k in ("x_offset_m", "bending_E", "stretching_K")]
            for row in worlds
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(parameters).all()
        or np.any(np.abs(parameters[:, 0]) > 0.02)
        or np.any(parameters[:, 1:] <= 0)
        or controls.dtype != np.float64
        or not np.isfinite(controls).all()
    ):
        raise ValueError("invalid registered native parameters")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    module = importlib.import_module("envs.env_slingshot")
    import genesis as gs
    import torch
    from envs.env_slingshot import Train_Env_Slingshot

    captured: dict[str, Any] = {}

    class RegisteredWorldBatch(Train_Env_Slingshot):
        def _set_parameter(self, objects, envs_idx, name, column):
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full native reset required")
            values = torch.tensor(parameters[:, column].copy(), dtype=gs.tc_float)
            getattr(self.rope, f"set_{name}_stiffness")(values)
            actual = getattr(
                self.scene.sim.rod_solver, f"rods_{name}_stiffness"
            ).to_numpy()
            if actual.shape != (1, 8) or not np.array_equal(
                actual[0], parameters[:, column]
            ):
                raise ValueError("native material parameter binding failed")
            captured[name] = actual.tolist()

        def _randomize_bending_stiffness(self, objects, envs_idx=None):
            self._set_parameter(objects, envs_idx, "bending", 1)

        def _randomize_stretching_stiffness(self, objects, envs_idx=None):
            self._set_parameter(objects, envs_idx, "stretching", 2)

        def _randomize_sphere_and_cube_positions(self, envs_idx=None):
            if envs_idx is not None:
                raise ValueError("full native reset required")
            offsets = np.zeros((8, 3), dtype=np.float64)
            offsets[:, 0] = parameters[:, 0]
            displacement = torch.tensor(offsets, dtype=gs.tc_float)
            for name in ("sphere", "cube"):
                entity = getattr(self, name)
                before = entity.get_pos().detach().cpu().numpy().copy()
                entity.set_pos(displacement, relative=True)
                after = entity.get_pos().detach().cpu().numpy().copy()
                if not np.allclose(after - before, offsets, rtol=0, atol=1e-15):
                    raise ValueError("native placement binding failed")
                captured[f"{name}_initial_position_m"] = after.tolist()

    with patch.object(module, "Train_Env_Slingshot", RegisteredWorldBatch):
        execute = run_prefix if prefix_only else run_batch
        values, native = execute(upstream, output, controls)
    if set(captured) != {
        "bending",
        "stretching",
        "sphere_initial_position_m",
        "cube_initial_position_m",
    }:
        raise ValueError("incomplete native world realization")
    return values, {**native, "world_realization": captured}
