"""Pure contracts for independent-process native Slingshot execution."""

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_benchmark import RIGID_FIELDS
from bayesian_phystwin_experiments.dlolab_native import STATE_FIELDS
from bayesian_phystwin_experiments.dlolab_slingshot_batch import (
    MEMORY_NAMES,
    TRACE_NAMES,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import task_metrics
from bayesian_phystwin_experiments.dlolab_slingshot_independent_native_v3 import (
    ACTION_COUNT,
    PROCESS_COUNT,
    WORLD_COUNT,
    combine_singletons,
    independent_world_qa,
    protocol,
    qualification_worlds,
    run_registered_world,
    task,
    validate_roster,
    validate_singleton_arrays,
    validate_world_realization,
)


def _row(control: np.ndarray) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in TRACE_NAMES:
        tail = (
            (12, 3)
            if name.startswith("rod_")
            else (9,)
            if name == "robot_qpos"
            else (3,)
        )
        arrays[name] = np.zeros((900, 1, *tail), dtype=np.float64)
    arrays["cube_pos_m"][:, :, 1] = 0.23
    arrays.update(
        {
            f"memory_RigidSolverState.{name}": np.ones((1, 2), dtype=np.float64)
            for name in RIGID_FIELDS
        }
    )
    arrays.update(
        {
            f"memory_RODSolverState.{name}": np.ones(
                (1, 2), dtype=bool if name == "fixed" else np.float64
            )
            for name in STATE_FIELDS
        }
    )
    arrays["controls"] = np.array(control, dtype=np.float64, order="C", copy=True)
    arrays["joint_targets"] = np.zeros((1, 30, 9), dtype=np.float32)
    assert set(arrays) == set(
        TRACE_NAMES + MEMORY_NAMES + ("controls", "joint_targets")
    )
    return arrays


def _realization(world: dict) -> dict:
    return {
        "bending": [[world["bending_E"]]],
        "stretching": [[world["stretching_K"]]],
        "sphere_initial_position_m": [[0.12 + world["x_offset_m"], 0.06, 0.2]],
        "cube_initial_position_m": [[0.12 + world["x_offset_m"], 0.23, 0.22]],
    }


def _evidence():
    world = qualification_worlds()[0]
    controls = np.zeros((ACTION_COUNT, 3, 6), dtype=np.float64)
    controls[:, 1, 0] = np.arange(ACTION_COUNT)
    controls[7] = controls[5]
    rows = [_row(controls[index : index + 1]) for index in range(ACTION_COUNT)]
    reports = [
        {
            "native_steps": 900,
            "environment_count": 1,
            "fresh_python_process": True,
            "world": world,
            "world_realization": _realization(world),
            "native_cumulative_reward": [task_metrics(row)["native_reward"]],
        }
        for row in rows
    ]
    return world, controls, rows, reports


def test_development_roster_is_fixed_fresh_and_execution_only():
    validate_roster()
    value = protocol()
    assert len(value["worlds"]) == WORLD_COUNT == 8
    assert value["process_count"] == PROCESS_COUNT == 64
    assert value["fresh_python_process_per_world_action"] is True
    assert value["v3_scientific_execution_automatically_authorized"] is False
    assert value["scientific_policy_value_scored"] is False
    assert value["protected_data_read"] is False
    assert task(63)["world_index"] == 7
    assert task(63)["action_index"] == 7
    with pytest.raises(ValueError, match="registered independent"):
        task(64)


def test_singleton_combination_preserves_axes_dtypes_and_controls():
    _, controls, rows, _ = _evidence()
    for row in rows:
        validate_singleton_arrays(row)
    combined = combine_singletons(rows)
    assert combined["rod_pos_m"].shape == (900, 8, 12, 3)
    assert combined["controls"].shape == (8, 3, 6)
    assert combined["joint_targets"].shape == (8, 30, 9)
    assert combined["memory_RODSolverState.fixed"].dtype == bool
    assert combined["controls"].tobytes() == controls.tobytes()
    changed = deepcopy(rows)
    changed[0].pop("joint_targets")
    with pytest.raises(ValueError, match="layout"):
        combine_singletons(changed)


def test_independent_world_qa_rederives_prefix_duplicate_and_reward_checks():
    world, controls, rows, reports = _evidence()
    result = independent_world_qa(rows, reports, controls, world)
    assert result["qa_passed"]
    assert result["independent_process_count"] == 8
    assert all(result["checks"].values())
    changed = deepcopy(rows)
    changed[7]["rod_pos_m"][500, 0, 6, 0] = 0.001
    assert not independent_world_qa(changed, reports, controls, world)["qa_passed"]
    changed = deepcopy(reports)
    changed[3]["native_cumulative_reward"] = [0.0]
    with pytest.raises(ValueError, match="reward mismatch"):
        independent_world_qa(rows, changed, controls, world)


def test_world_realization_and_fresh_process_metadata_are_fail_closed():
    world, controls, rows, reports = _evidence()
    changed = deepcopy(reports)
    changed[0]["world_realization"]["bending"][0][0] += 1.0
    with pytest.raises(ValueError, match="material realization"):
        independent_world_qa(rows, changed, controls, world)
    changed = deepcopy(reports)
    changed[0]["fresh_python_process"] = False
    with pytest.raises(ValueError, match="fresh-process"):
        independent_world_qa(rows, changed, controls, world)
    validate_world_realization(reports[0], world)


def test_invalid_control_is_rejected_before_native_import(tmp_path):
    world = qualification_worlds()[0]
    with pytest.raises(ValueError, match="native control"):
        run_registered_world(
            tmp_path,
            tmp_path,
            np.zeros((3, 6), dtype=np.float64),
            world,
        )
