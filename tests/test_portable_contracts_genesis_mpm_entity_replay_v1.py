from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    GenesisMPMEntityReplayV1,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


class _DeviceArray:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.events: list[str] = []

    def detach(self) -> _DeviceArray:
        self.events.append("detach")
        return self

    def cpu(self) -> _DeviceArray:
        self.events.append("cpu")
        return self

    def numpy(self) -> np.ndarray:
        self.events.append("numpy")
        return self.value


class _GenesisState:
    def __init__(self, positions: np.ndarray, active: np.ndarray) -> None:
        self.pos = _DeviceArray(positions)
        self.active = _DeviceArray(active)


class _GenesisEntity:
    def __init__(self, positions: np.ndarray, active: np.ndarray) -> None:
        self.positions = np.ascontiguousarray(positions).copy()
        self.active = np.ascontiguousarray(active).copy()
        self.pending = np.zeros_like(self.positions)
        self.last_state: _GenesisState | None = None

    def get_state(self) -> _GenesisState:
        self.last_state = _GenesisState(self.positions, self.active)
        return self.last_state

    def step(self) -> None:
        self.positions += self.pending
        self.pending.fill(0.0)


def _positions() -> np.ndarray:
    return np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.03, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [1.01, 0.0, 0.0],
                [1.02, 0.0, 0.0],
                [1.03, 0.0, 0.0],
            ],
        ],
        dtype=np.float64,
    )


def _active() -> np.ndarray:
    return np.array(
        [[1, 1, 0, 1], [1, 0, 1, 1]],
        dtype=np.int32,
    )


def test_genesis_replay_selects_environment_and_preserves_host_ownership() -> None:
    entity = _GenesisEntity(_positions(), _active())
    events: list[str] = []
    replay = GenesisMPMEntityReplayV1(
        entity=entity,
        environment_index=np.int64(1),
        step_callback=entity.step,
        synchronize_callback=lambda: events.append("sync"),
        context=entity,
    )

    replay.synchronize()
    positions = replay.get_material_positions_m()
    np.testing.assert_array_equal(positions, _positions()[1, [0, 2, 3]])
    positions[0, 0] = 100.0
    assert entity.positions[1, 0, 0] == 1.0
    assert entity.last_state is not None
    assert entity.last_state.pos.events == ["detach", "cpu", "numpy"]
    assert entity.last_state.active.events == ["detach", "cpu", "numpy"]

    entity.pending[1, :, 2] = 0.002
    replay.step()
    np.testing.assert_allclose(
        replay.get_material_positions_m()[:, 2],
        np.full(3, 0.002),
    )
    assert replay.context is entity
    assert events == ["sync"]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"entity": object()}, TypeError, r"get_state\(\)"),
        ({"step_callback": None}, TypeError, "step_callback"),
        ({"synchronize_callback": None}, TypeError, "synchronize_callback"),
        ({"environment_index": -1}, ValueError, "nonnegative integer"),
        ({"environment_index": 2}, ValueError, "exceeds"),
    ),
)
def test_genesis_replay_rejects_invalid_configuration(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    parameters: dict[str, Any] = {
        "entity": _GenesisEntity(_positions(), _active()),
        "step_callback": lambda: None,
    }
    parameters.update(kwargs)
    with pytest.raises(error, match=message):
        GenesisMPMEntityReplayV1(**parameters)


class _InvalidStateEntity:
    def __init__(self, state: object) -> None:
        self.state = state

    def get_state(self) -> object:
        return self.state


@pytest.mark.parametrize(
    ("state", "error", "message"),
    (
        (None, ValueError, "no active state"),
        (object(), TypeError, "pos and active"),
        (
            _GenesisState(
                np.zeros((4, 3), dtype=np.float64),
                np.ones((1, 4), dtype=np.int32),
            ),
            ValueError,
            r"shape \(B,N,3\)",
        ),
        (
            _GenesisState(
                np.zeros((1, 4, 3), dtype=np.int64),
                np.ones((1, 4), dtype=np.int32),
            ),
            ValueError,
            "floating point",
        ),
        (
            _GenesisState(
                np.zeros((1, 4, 3), dtype=np.float64),
                np.ones((1, 3), dtype=np.int32),
            ),
            ValueError,
            r"shape \(B,N\)",
        ),
        (
            _GenesisState(
                np.zeros((1, 4, 3), dtype=np.float64),
                np.full((1, 4), 0.5, dtype=np.float64),
            ),
            ValueError,
            "boolean or integer",
        ),
        (
            _GenesisState(
                np.zeros((1, 4, 3), dtype=np.float64),
                np.array([[1, 2, 0, 1]], dtype=np.int32),
            ),
            ValueError,
            "zero or one",
        ),
        (
            _GenesisState(
                np.zeros((1, 4, 3), dtype=np.float64),
                np.zeros((1, 4), dtype=np.int32),
            ),
            ValueError,
            "at least one active particle",
        ),
    ),
)
def test_genesis_replay_rejects_invalid_state(
    state: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        GenesisMPMEntityReplayV1(
            entity=_InvalidStateEntity(state),
            step_callback=lambda: None,
        )


def test_genesis_replay_rejects_nonfinite_active_positions() -> None:
    positions = _positions()[:1]
    positions[0, 1, 0] = np.nan
    active = np.array([[1, 1, 0, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="non-finite"):
        GenesisMPMEntityReplayV1(
            entity=_GenesisEntity(positions, active),
            step_callback=lambda: None,
        )


def test_genesis_replay_rejects_particle_or_active_roster_changes() -> None:
    entity = _GenesisEntity(_positions()[:1], _active()[:1])
    replay = GenesisMPMEntityReplayV1(
        entity=entity,
        step_callback=entity.step,
    )

    entity.active[0, 2] = 1
    with pytest.raises(ValueError, match="active-particle roster changed"):
        replay.get_material_positions_m()

    entity.active = _active()[:1].copy()
    entity.positions = entity.positions[:, :3].copy()
    entity.active = entity.active[:, :3].copy()
    with pytest.raises(ValueError, match="particle count changed"):
        replay.get_material_positions_m()


def _producer_kwargs(
    tmp_path: Path,
    replay_factory: Any,
) -> dict[str, Any]:
    return {
        "output_dir": tmp_path / "genesis-mpm-v1",
        "backend_kind": "genesis-mpm-v1",
        "replay_factory": replay_factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_control,
        "frame_count": 4,
        "material_query_indices": np.array([0, 1, 2], dtype=np.int64),
        "action_support": np.array([0.0, 0.7, 1.0], dtype=np.float64),
        "engine_revision": "a" * 40,
        "engine_version": "genesis-contract-test-v1",
        "producer_repository": "example/genesis-mpm-replay-producer",
        "producer_revision": "b" * 40,
        "producer_version": "producer-v1",
        "producer_artifacts": {"scene.py": "c" * 64},
        "topology_sha256": "d" * 64,
        "device": "cpu",
        "device_name": "contract-test-cpu",
        "time_step_s": 0.01,
        "scene_id": "genesis-mpm-contract-test",
        "model_kind": "deformable-solid",
        "constitutive_model": "neo-hookean",
        "integrator": "mpm-grid-particle",
        "solver": "genesis-mpm",
        "substeps": 1,
    }


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    entity = replay.context
    assert isinstance(entity, _GenesisEntity)
    entity.pending[0, :, 2] = 0.0005 * (transition_index + 1)


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    entity = replay.context
    assert isinstance(entity, _GenesisEntity)
    entity.pending.fill(0.0)


def test_genesis_adapter_runs_through_portable_material_producer(
    tmp_path: Path,
) -> None:
    def replay_factory() -> GenesisMPMEntityReplayV1:
        entity = _GenesisEntity(
            _positions()[:1],
            np.ones((1, 4), dtype=np.int32),
        )
        return GenesisMPMEntityReplayV1(
            entity=entity,
            step_callback=entity.step,
            context=entity,
        )

    produce_material_trajectory_backend(**_producer_kwargs(tmp_path, replay_factory))
    physical = load_physical_rollout_archive(
        tmp_path / "genesis-mpm-v1" / "physical-prediction.npz"
    )
    assert np.max(physical["prediction_m"][-1, :, 2]) > 0.0
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(physical["frame_zero_points_m"][None], 4, axis=0),
    )
