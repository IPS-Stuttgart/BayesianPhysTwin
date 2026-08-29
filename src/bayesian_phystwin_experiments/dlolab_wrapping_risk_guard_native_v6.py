"""Corrected runtime qualification for the wrapping chance-guard lineage."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_benchmark import native_memory
from .dlolab_native import array_digest
from .dlolab_wrapping_risk_guard_v4 import validate_continuous_world
from .dlolab_wrapping_source import N_ENVS, action_bank

PREFIX_MACRO_STEPS = 3
FULL_MACRO_STEPS = 11
MICRO_STEPS_PER_MACRO = 200
Array: TypeAlias = NDArray[Any]


def _observe(env: Any) -> dict[str, Array]:
    def host(value: Any) -> Array:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result: Array = np.array(value, order="C", copy=True)
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite native wrapping observable")
        return result

    return {
        "rod_pos_m": host(env.rope.get_all_verts()),
        "rod_vel_m_s": host(env.rope.get_all_vels()),
        "post_pos_m": np.stack(
            [host(post.get_pos()) for post in (env.post1, env.post2, env.post3)],
            axis=1,
        ),
        "gripper_pos_m": np.stack(
            [host(controller.ef.get_pos()) for controller in (env.c1, env.c2)],
            axis=1,
        ),
        "robot_qpos": np.concatenate(
            [host(robot.get_qpos()) for robot in (env.franka1, env.franka2)],
            axis=1,
        ),
    }


def _runtime_modules(upstream: Path) -> tuple[Any, Any, Any, Any]:
    for path in (upstream / "experiments", upstream):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import genesis as gs
    import torch
    from envs.env_wrapping import Train_Env_Wrapping
    from omegaconf import DictConfig

    return gs, torch, Train_Env_Wrapping, DictConfig


def _initialize(
    upstream: Path,
    output: Path,
    worlds: list[dict[str, Any]],
) -> tuple[Any, Any, dict[str, list[float]]]:
    if len(worlds) != N_ENVS:
        raise ValueError("exactly nine registered native worlds required")
    for world in worlds:
        validate_continuous_world(world)
    gs, torch, train_env, dict_config = _runtime_modules(upstream)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    captured: dict[str, list[float]] = {}

    class RegisteredWorld(train_env):  # type: ignore[misc,valid-type]
        def _set_parameter(
            self,
            objects: list[Any],
            envs_idx: Any | None,
            name: str,
            key: str,
        ) -> None:
            if envs_idx is not None or objects != [self.rope]:
                raise ValueError("full native reset required")
            expected = [float(world[key]) for world in worlds]
            field = getattr(self.scene.sim.rod_solver, f"rods_{name}_stiffness")
            values = torch.tensor(expected, dtype=gs.tc_float)
            getattr(self.rope, f"set_{name}_stiffness")(values)
            actual = field.to_numpy()
            if actual.shape != (1, N_ENVS) or not np.array_equal(
                actual[0], np.asarray(expected)
            ):
                raise ValueError("native continuous material realization changed")
            captured[name] = actual[0].tolist()

        def _randomize_bending_stiffness(
            self, objects: list[Any], envs_idx: Any | None = None
        ) -> None:
            self._set_parameter(objects, envs_idx, "bending", "bending_E")

        def _randomize_stretching_stiffness(
            self, objects: list[Any], envs_idx: Any | None = None
        ) -> None:
            self._set_parameter(objects, envs_idx, "stretching", "stretching_K")

    gs.init(
        seed=0,
        precision="64",
        logging_level="error",
        backend=gs.cpu,
        performance_mode=True,
        theme="dumb",
    )
    env = RegisteredWorld(
        dict_config(
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
    if captured:
        raise RuntimeError("native material randomization was not deferred")
    if not np.all(env.scene.sim.rod_solver.rods_twisting_stiffness.to_numpy() == 0):
        raise ValueError("native zero twisting stiffness changed")
    return gs, env, captured


def run_constructor_probe(
    upstream: Path,
    output: Path,
    worlds: list[dict[str, Any]],
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Construct through ``init_cmaes_env`` and serialize the untouched state."""
    started = time.monotonic()
    gs: Any | None = None
    try:
        gs, env, captured = _initialize(upstream, output, worlds)
        arrays = _observe(env)
        arrays.update(
            {
                f"memory_{name}": value
                for name, value in native_memory(env.scene.get_state()).items()
            }
        )
        return arrays, {
            "native_steps": 0,
            "wall_seconds": time.monotonic() - started,
            "worlds": worlds,
            "world_realization": captured,
            "parameter_randomization_deferred": True,
            "state_sha256": {
                name: array_digest(value) for name, value in sorted(arrays.items())
            },
            "constructor_completed": True,
            "init_cmaes_env_completed": True,
            "future_simulated": False,
            "reward_exposed": False,
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
        }
    finally:
        if gs is not None and getattr(gs, "_initialized", False):
            gs.destroy()


def run_worlds(
    upstream: Path,
    output: Path,
    worlds: list[dict[str, Any]],
    *,
    prefix_only: bool,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Run the v4 physical workload through the runtime-hardened constructor."""
    controls = action_bank()[:, :PREFIX_MACRO_STEPS] if prefix_only else action_bank()
    expected_steps = (
        PREFIX_MACRO_STEPS if prefix_only else FULL_MACRO_STEPS
    ) * MICRO_STEPS_PER_MACRO
    started = time.monotonic()
    gs: Any | None = None
    try:
        gs, env, captured = _initialize(upstream, output, worlds)
        trace: list[dict[str, Array]] = []
        initial: list[Array] = []
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            if len(trace) >= expected_steps:
                raise RuntimeError("native continuous wrapping horizon exceeded")
            if not trace:
                initial.append(np.array(env.rope.get_all_verts(), order="C", copy=True))
            result = original_step(*args, **kwargs)
            trace.append(_observe(env))
            return result

        env.scene.step = step
        try:
            native_result = env.eval_traj(controls)
        finally:
            env.scene.step = original_step
        if (
            len(trace) != expected_steps
            or len(initial) != 1
            or set(captured) != {"bending", "stretching"}
        ):
            raise RuntimeError("incomplete native continuous wrapping execution")
        if not np.all(env.scene.sim.rod_solver.rods_twisting_stiffness.to_numpy() == 0):
            raise ValueError("native zero twisting stiffness changed")
        arrays = {name: np.stack([row[name] for row in trace]) for name in trace[0]}
        arrays.update(
            controls=controls,
            joint_targets=env.qpos_seq.copy(),
            initial_rod_pos_m=initial[0],
        )
        arrays.update(
            {
                f"memory_{name}": value
                for name, value in native_memory(env.scene.get_state()).items()
            }
        )
        metadata: dict[str, Any] = {
            "native_steps": len(trace),
            "native_forward_seconds": float(native_result["forward_time"]),
            "wall_seconds": time.monotonic() - started,
            "worlds": worlds,
            "world_realization": captured,
            "prefix_only": prefix_only,
            "future_simulated": not prefix_only,
            "reward_exposed": not prefix_only,
            "prefix_reward_excluded": prefix_only,
            "twisting_stiffness_zero_preserved": True,
            "device": "cpu",
            "runtime_camera_rendered": False,
            "native_source_modified": False,
        }
        if not prefix_only:
            metadata.update(
                native_final_reward=np.asarray(native_result["final_reward"]).tolist(),
                native_cumulative_reward=np.asarray(
                    native_result["cum_reward"]
                ).tolist(),
            )
        return arrays, metadata
    finally:
        if gs is not None and getattr(gs, "_initialized", False):
            gs.destroy()
