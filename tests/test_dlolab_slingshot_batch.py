"""Batch-axis and full-memory parity, without simulator initialization."""

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_benchmark import slingshot_actions
from bayesian_phystwin_experiments.dlolab_slingshot_batch import (
    BATCH_INDICES,
    MEMORY_NAMES,
    TRACE_NAMES,
    compare,
    protocol,
    run_batch,
    split_batch,
)
from bayesian_phystwin_experiments.dlolab_slingshot_controls import (
    native_reward_from_trace,
)


def _bank():
    arrays = {}
    for name in TRACE_NAMES:
        tail = (
            (12, 3)
            if name.startswith("rod_")
            else (9,)
            if name == "robot_qpos"
            else (3,)
        )
        arrays[name] = np.zeros((900, 8, *tail))
    arrays["cube_pos_m"][:, :, 1] = 0.23
    for name in MEMORY_NAMES:
        arrays[name] = np.ones((8, 2), dtype=bool if name.endswith(".fixed") else float)
    arrays["controls"] = np.stack([slingshot_actions()[i] for i in BATCH_INDICES])
    arrays["joint_targets"] = np.zeros((8, 30, 9), dtype=np.float32)
    return arrays


def test_batch_splitting_preserves_singleton_identity_axis_and_dtypes():
    arrays = _bank()
    rows = split_batch(arrays, 8)
    assert len(rows) == 8
    for index, row in enumerate(rows):
        assert row["rod_pos_m"].shape == (900, 1, 12, 3)
        assert row["memory_RODSolverState.fixed"].dtype == bool
        assert (
            row["controls"].tobytes() == arrays["controls"][index : index + 1].tobytes()
        )
    with pytest.raises(ValueError, match="layout"):
        split_batch(arrays, 7)
    arrays["gripper_pos_m"] = np.zeros((8, 900, 3))
    with pytest.raises(ValueError, match="axis"):
        split_batch(arrays, 8)


def test_batch_qualification_uses_same_strict_memory_and_position_gates():
    rows = split_batch(_bank(), 8)
    references = [
        {name: value.copy() for name, value in rows[i].items()} for i in range(2)
    ]
    rewards = [native_reward_from_trace(row["cube_pos_m"]) for row in rows]
    assert compare(rows, references, rewards)["batch_qualification_passed"]
    rows[7]["memory_RODSolverState.vel"][0, 0] = 0.5
    result = compare(rows, references, rewards)
    assert not result["batch_qualification_passed"]
    assert result["rows"][7]["failed_memory_fields"] == ["memory_RODSolverState.vel"]
    assert protocol()["memory_atol"] == 1e-9
    assert not protocol()["method_evaluation_authorized"]


def test_incomplete_or_nonfinite_native_batch_is_rejected():
    arrays = _bank()
    arrays.pop(MEMORY_NAMES[0])
    with pytest.raises(ValueError, match="layout"):
        split_batch(arrays, 8)
    arrays = _bank()
    arrays["rod_vel_m_s"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        split_batch(arrays, 8)


def test_bad_batch_control_fails_before_import(tmp_path):
    with pytest.raises(ValueError, match="batch control"):
        run_batch(tmp_path, tmp_path, np.zeros((1, 3, 6)))
