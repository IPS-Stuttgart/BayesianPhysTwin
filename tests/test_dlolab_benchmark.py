"""Pure contracts, without importing or running the native task."""

from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_benchmark import (
    RIGID_FIELDS,
    fixed_endpoint_error,
    memory_comparison,
    native_memory,
    protocol,
    slingshot_actions,
)
from bayesian_phystwin_experiments.dlolab_native import STATE_FIELDS


def test_actions_preserve_official_three_stage_control_contract():
    actions = slingshot_actions()
    assert actions.shape == (2, 3, 6)
    assert actions.dtype == np.float64
    assert np.count_nonzero(actions[0]) == 0
    assert np.count_nonzero(actions[1]) == 3
    np.testing.assert_array_equal(actions[1, :, 1], [-0.04] * 3)
    assert np.max(np.linalg.norm(actions[:, :, :3], axis=-1)) < 0.1
    assert protocol()["automatic_method_evaluation_authorized"] is False


def test_exactly_two_native_solvers_with_all_memory_are_required():
    rigid = type("RigidSolverState", (), {})()
    rod = type("RODSolverState", (), {})()
    for value, fields in ((rigid, RIGID_FIELDS), (rod, STATE_FIELDS)):
        for name in fields:
            setattr(value, name, np.ones((1, 2)))
    arrays = native_memory(SimpleNamespace(solvers_state=[None, rigid, rod]))
    assert len(arrays) == 23
    assert memory_comparison(arrays, arrays)["byte_identical"] is True
    with pytest.raises(ValueError, match="exactly"):
        native_memory(SimpleNamespace(solvers_state=[rod]))
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["RODSolverState.pos"] += 0.001
    assert not memory_comparison(arrays, changed)["within_tolerance"]
    assert not memory_comparison(arrays, changed)["byte_identical"]
    changed["RODSolverState.pos"] = np.ones((1, 3))
    with pytest.raises(ValueError, match="layout"):
        memory_comparison(arrays, changed)


def test_native_arrays_are_copied_and_nonfinite_memory_is_rejected():
    rigid = type("RigidSolverState", (), {})()
    rod = type("RODSolverState", (), {})()
    for value, fields in ((rigid, RIGID_FIELDS), (rod, STATE_FIELDS)):
        for name in fields:
            setattr(value, name, np.ones((1, 2)))
    state = SimpleNamespace(solvers_state=[rigid, rod])
    arrays = native_memory(state)
    rod.pos[0, 0] = 2
    assert arrays["RODSolverState.pos"][0, 0] == 1
    rigid.qpos[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        native_memory(state)


def test_fixed_endpoint_check_does_not_mix_identity_or_world_axes():
    trace = np.zeros((3, 1, 12, 3))
    trace[:, :, :, 0] = np.arange(12) * 0.02
    changed = trace.copy()
    changed[:, :, 6] += 1
    assert fixed_endpoint_error([trace, changed]) == 0.0
    changed[-1, 0, 11, 2] = 0.001
    assert fixed_endpoint_error([trace, changed]) == 0.001
    with pytest.raises(ValueError):
        fixed_endpoint_error([trace[0]])
