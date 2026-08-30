"""Independent-process native Slingshot execution and development qualification."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_benchmark import native_memory
from .dlolab_native import array_digest
from .dlolab_slingshot_batch import MEMORY_NAMES, TRACE_NAMES
from .dlolab_slingshot_belief import native_qa
from .dlolab_slingshot_policy_certificate_source_v2 import (
    continuous_worlds as policy_v2_worlds,
)
from .dlolab_slingshot_policy_certificate_source_v2 import opened_world_keys
from .dlolab_slingshot_process import observe

Array: TypeAlias = NDArray[Any]
WORLD_SEED = 262060
WORLD_COUNT = 8
ACTION_COUNT = 8
PROCESS_COUNT = WORLD_COUNT * ACTION_COUNT
POSITION_ENVELOPE_M = 0.0005
REWARD_ENVELOPE = 0.001
FIXED_ENDPOINT_ENVELOPE_M = 1e-9


def _world_key(world: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(world["x_offset_m"]),
        float(world["bending_E"]),
        float(world["stretching_K"]),
    )


def qualification_worlds() -> list[dict[str, Any]]:
    """Return a fixed development-only roster disjoint from all prior worlds."""

    rng = np.random.default_rng(WORLD_SEED)
    x = rng.uniform(-0.02, 0.02, WORLD_COUNT)
    bending = 1e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), WORLD_COUNT))
    stretching = 8e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), WORLD_COUNT))
    return [
        {
            "index": index,
            "x_offset_m": float(x[index]),
            "bending_E": float(bending[index]),
            "stretching_K": float(stretching[index]),
        }
        for index in range(WORLD_COUNT)
    ]


def validate_world(world: dict[str, Any]) -> None:
    if set(world) != {"index", "x_offset_m", "bending_E", "stretching_K"}:
        raise ValueError("exact registered Slingshot world schema required")
    parameters = np.asarray(
        [world["x_offset_m"], world["bending_E"], world["stretching_K"]],
        dtype=np.float64,
    )
    if (
        type(world["index"]) is not int
        or world["index"] not in range(WORLD_COUNT)
        or not np.isfinite(parameters).all()
        or abs(parameters[0]) > 0.02
        or np.any(parameters[1:] <= 0)
    ):
        raise ValueError("invalid registered Slingshot world")


def validate_roster() -> None:
    roster = qualification_worlds()
    for world in roster:
        validate_world(world)
    current = {_world_key(world) for world in roster}
    prior = opened_world_keys() | {
        _world_key(world)
        for role in ("calibration", "evaluation")
        for world in policy_v2_worlds(role)
    }
    if len(current) != WORLD_COUNT or current & prior:
        raise ValueError("fresh disjoint executor-qualification worlds required")


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(PROCESS_COUNT):
        raise ValueError("registered independent native task required")
    world_index, action_index = divmod(index, ACTION_COUNT)
    return {
        "kind": "independent_action_rollout",
        "name": f"world-{world_index:02d}-action-{action_index:02d}",
        "index": index,
        "world_index": world_index,
        "action_index": action_index,
    }


def protocol() -> dict[str, Any]:
    validate_roster()
    return {
        "schema": "dlolab-slingshot-independent-native-qualification-v1",
        "purpose": "qualify_one_world_one_action_fresh_process_execution",
        "world_seed": WORLD_SEED,
        "worlds": qualification_worlds(),
        "world_count": WORLD_COUNT,
        "action_slots_per_world": ACTION_COUNT,
        "process_count": PROCESS_COUNT,
        "fresh_python_process_per_world_action": True,
        "duplicate_action_slots": [5, 7],
        "gate": {
            "ordinary_processes_required": PROCESS_COUNT,
            "qualified_worlds_required": WORLD_COUNT,
            "common_prefix_at_most_m": POSITION_ENVELOPE_M,
            "duplicate_position_error_at_most_m": POSITION_ENVELOPE_M,
            "duplicate_reward_error_at_most": REWARD_ENVELOPE,
            "fixed_endpoint_error_at_most_m": FIXED_ENDPOINT_ENVELOPE_M,
            "exact_world_realization_required": True,
        },
        "retry_authorized": False,
        "replacement_authorized": False,
        "v2_world_retry_authorized": False,
        "v2_partial_outcome_scoring_authorized": False,
        "v3_protocol_freeze_authorized_only_after_pass": True,
        "v3_scientific_execution_automatically_authorized": False,
        "scientific_policy_value_scored": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "new_recordings": False,
        "gpu_work": False,
        "push_or_merge": False,
    }


def _expected_realization(world: dict[str, Any]) -> dict[str, Any]:
    return {
        "bending": [[world["bending_E"]]],
        "stretching": [[world["stretching_K"]]],
        "sphere_initial_position_m": [[0.12 + world["x_offset_m"], 0.06, 0.2]],
        "cube_initial_position_m": [[0.12 + world["x_offset_m"], 0.23, 0.22]],
    }


def validate_world_realization(native: dict[str, Any], world: dict[str, Any]) -> None:
    validate_world(world)
    actual = native.get("world_realization")
    expected = _expected_realization(world)
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("complete native world realization required")
    for name in ("bending", "stretching"):
        if actual[name] != expected[name]:
            raise ValueError("native material realization changed")
    for name in ("sphere_initial_position_m", "cube_initial_position_m"):
        observed = np.asarray(actual[name], dtype=np.float64)
        target = np.asarray(expected[name], dtype=np.float64)
        if observed.shape != (1, 3) or not np.allclose(
            observed, target, rtol=0.0, atol=1e-15
        ):
            raise ValueError("native placement realization changed")


def run_registered_world(
    upstream: Path,
    output: Path,
    control: Array,
    world: dict[str, Any],
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Run exactly one world/action in the current fresh interpreter."""

    validate_world(world)
    command = np.asarray(control)
    if (
        command.shape != (1, 3, 6)
        or command.dtype != np.float64
        or not np.isfinite(command).all()
    ):
        raise ValueError("native control must be a finite float64 (1,3,6) array")
    for path in (upstream / "experiments", upstream):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    gs: Any = importlib.import_module("genesis")
    import torch
    from envs.env_slingshot import Train_Env_Slingshot
    from omegaconf import DictConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    started = time.monotonic()
    captured: dict[str, Any] = {}

    class RegisteredWorld(Train_Env_Slingshot):  # type: ignore[misc]
        def _set_parameter(
            self, objects: list[Any], envs_idx: Any | None, name: str, key: str
        ) -> None:
            if envs_idx is not None or len(objects) != 1 or objects[0] is not self.rope:
                raise ValueError("full single-world native reset required")
            expected = np.asarray([world[key]], dtype=np.float64)
            getattr(self.rope, f"set_{name}_stiffness")(
                torch.tensor(expected, dtype=gs.tc_float)
            )
            actual = getattr(
                self.scene.sim.rod_solver, f"rods_{name}_stiffness"
            ).to_numpy()
            if actual.shape != (1, 1) or not np.array_equal(actual[0], expected):
                raise ValueError("native material parameter binding failed")
            captured[name] = actual.tolist()

        def _randomize_bending_stiffness(
            self, objects: list[Any], envs_idx: Any | None = None
        ) -> None:
            self._set_parameter(objects, envs_idx, "bending", "bending_E")

        def _randomize_stretching_stiffness(
            self, objects: list[Any], envs_idx: Any | None = None
        ) -> None:
            self._set_parameter(objects, envs_idx, "stretching", "stretching_K")

        def _randomize_sphere_and_cube_positions(
            self, envs_idx: Any | None = None
        ) -> None:
            if envs_idx is not None:
                raise ValueError("full single-world native reset required")
            offset = np.asarray([[world["x_offset_m"], 0.0, 0.0]], dtype=np.float64)
            displacement = torch.tensor(offset, dtype=gs.tc_float)
            for name in ("sphere", "cube"):
                entity = getattr(self, name)
                before = entity.get_pos().detach().cpu().numpy().copy()
                entity.set_pos(displacement, relative=True)
                after = entity.get_pos().detach().cpu().numpy().copy()
                if after.shape != (1, 3) or not np.allclose(
                    after - before, offset, rtol=0.0, atol=1e-15
                ):
                    raise ValueError("native placement binding failed")
                captured[f"{name}_initial_position_m"] = after.tolist()

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
        trace: list[dict[str, Array]] = []
        original_step = env.scene.step

        def step(*args: Any, **kwargs: Any) -> Any:
            value = original_step(*args, **kwargs)
            trace.append(observe(env))
            return value

        env.scene.step = step
        try:
            native = env.eval_traj(command.copy())
        finally:
            env.scene.step = original_step
        if len(trace) != 900:
            raise ValueError("native step count changed")
        arrays = {name: np.stack([row[name] for row in trace]) for name in trace[0]}
        arrays.update(controls=command.copy(), joint_targets=env.qpos_seq.copy())
        arrays.update(
            {
                f"memory_{name}": value
                for name, value in native_memory(env.scene.get_state()).items()
            }
        )
        report = {
            "native_cumulative_reward": np.asarray(native["cum_reward"]).tolist(),
            "native_forward_seconds": float(native["forward_time"]),
            "wall_seconds": time.monotonic() - started,
            "native_steps": len(trace),
            "environment_count": 1,
            "fresh_python_process": True,
            "world": world,
            "world_realization": captured,
        }
        validate_world_realization(report, world)
        return arrays, report
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()


def combine_singletons(rows: list[dict[str, Array]]) -> dict[str, Array]:
    if len(rows) != ACTION_COUNT:
        raise ValueError("all eight independent actions are required")
    result: dict[str, Array] = {}
    for index, row in enumerate(rows):
        validate_singleton_arrays(row, index=index)
    expected = set(TRACE_NAMES + MEMORY_NAMES + ("controls", "joint_targets"))
    for name in sorted(expected):
        axis = 1 if name in TRACE_NAMES else 0
        result[name] = np.ascontiguousarray(
            np.concatenate([row[name] for row in rows], axis=axis)
        )
    return result


def validate_singleton_arrays(
    row: dict[str, Array], *, index: int | None = None
) -> None:
    expected = set(TRACE_NAMES + MEMORY_NAMES + ("controls", "joint_targets"))
    if set(row) != expected:
        raise ValueError("independent native array layout changed")
    for name, value in row.items():
        axis = 1 if name in TRACE_NAMES else 0
        if (
            value.ndim <= axis
            or value.shape[axis] != 1
            or (axis == 1 and value.shape[0] != 900)
            or value.dtype.kind not in "bifu"
            or not np.isfinite(value).all()
        ):
            label = "" if index is None else f" at action {index}"
            raise ValueError(f"invalid singleton native array{label}: {name}")


def independent_world_qa(
    rows: list[dict[str, Array]],
    reports: list[dict[str, Any]],
    expected_controls: Array,
    world: dict[str, Any],
) -> dict[str, Any]:
    validate_world(world)
    controls = np.asarray(expected_controls)
    if (
        len(reports) != ACTION_COUNT
        or controls.shape != (ACTION_COUNT, 3, 6)
        or controls.dtype != np.float64
        or not np.isfinite(controls).all()
    ):
        raise ValueError("complete independent-action evidence required")
    for report in reports:
        validate_world_realization(report, world)
        if (
            report.get("native_steps") != 900
            or report.get("environment_count") != 1
            or report.get("fresh_python_process") is not True
            or report.get("world") != world
            or not isinstance(report.get("native_cumulative_reward"), list)
            or len(report["native_cumulative_reward"]) != 1
        ):
            raise ValueError("fresh-process native report changed")
    combined = combine_singletons(rows)
    if array_digest(combined["controls"]) != array_digest(controls):
        raise ValueError("independent action controls changed")
    native = {
        "native_cumulative_reward": [
            report["native_cumulative_reward"][0] for report in reports
        ]
    }
    qa = native_qa(combined, native, controls)
    checks = {
        "all_eight_fresh_processes": True,
        "exact_world_realization": True,
        **qa["checks"],
    }
    return {
        **qa,
        "checks": checks,
        "qa_passed": bool(all(checks.values())),
        "independent_process_count": ACTION_COUNT,
    }
