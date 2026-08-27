from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_native import (
    STATE_FIELDS,
    DloLabConfig,
    NativeSnapshot,
    clamp_only_state,
    native_state_arrays,
    native_state_digests,
)


def _state():
    fields = {name: np.full((1, 6, 3), 0.1) for name in STATE_FIELDS}
    fields["fixed"] = np.zeros((1, 6), dtype=bool)
    return SimpleNamespace(solvers_state=[None, SimpleNamespace(**fields)])


def test_only_commands_change_clamps_not_free_positions_or_velocities():
    positions = np.arange(48, dtype=float).reshape(2, 8, 3) / 100
    velocities = positions * 0.1
    commands = np.full((2, 2, 3), 0.04)
    updated, velocity = clamp_only_state(positions, velocities, commands)
    np.testing.assert_array_equal(updated[:, :2], commands)
    np.testing.assert_array_equal(velocity[:, :2], 0.0)
    np.testing.assert_array_equal(updated[:, 2:], positions[:, 2:])
    np.testing.assert_array_equal(velocity[:, 2:], velocities[:, 2:])
    assert not np.shares_memory(updated, positions)
    assert not np.shares_memory(velocity, velocities)


@pytest.mark.parametrize("shape", [(1, 6, 3), (6, 3), (1, 3, 3), (1, 2, 4)])
def test_future_free_geometry_cannot_enter_control_api(shape):
    with pytest.raises(ValueError, match="two-clamp"):
        clamp_only_state(np.zeros((1, 6, 3)), np.zeros((1, 6, 3)), np.zeros(shape))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_nonfinite_control_or_state_is_rejected(index):
    args = [np.zeros((1, 6, 3)), np.zeros((1, 6, 3)), np.zeros((1, 2, 3))]
    args[index].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        clamp_only_state(*args)


def test_snapshot_requires_every_native_field_and_model_identity():
    config = DloLabConfig()
    state = _state()
    snapshot = NativeSnapshot(3, config.identity, state, native_state_digests(state))
    snapshot.validate(config)
    with pytest.raises(ValueError, match="configuration"):
        snapshot.validate(dataclasses.replace(config, dt_s=0.001))
    state.solvers_state[1].theta.flat[0] += 0.01
    with pytest.raises(ValueError, match="mutated"):
        snapshot.validate(config)
    del state.solvers_state[1].theta
    with pytest.raises(ValueError, match="exactly one"):
        native_state_digests(state)


def test_coupled_solver_not_silently_discarded():
    state = _state()
    state.solvers_state[0] = SimpleNamespace()
    with pytest.raises(ValueError, match="exactly one"):
        native_state_digests(state)


def test_array_snapshots_are_owned_and_dtype_preserving():
    state = _state()
    arrays = native_state_arrays(state)
    assert arrays["fixed"].dtype == np.bool_
    assert arrays["pos"].dtype == np.float64
    for name, array in arrays.items():
        assert not np.shares_memory(array, getattr(state.solvers_state[1], name))


@pytest.mark.parametrize(
    "changes",
    [
        {"node_count": True},
        {"node_count": 5},
        {"dt_s": float("nan")},
        {"substeps": 0},
        {"bending_modulus": -1.0},
        {"seed": False},
        {"schema": "unknown"},
    ],
)
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(DloLabConfig(), **changes)
