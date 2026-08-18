from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.genesis_mpm_replay_v1 import GenesisMPMEntityReplayV1
from bayesian_phystwin.material_trajectory_producer_v1 import (
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


class _TensorView:
    def __init__(self, value: np.ndarray, events: list[str]) -> None:
        self._value = value
        self._events = events

    def block_until_ready(self) -> _TensorView:
        self._events.append("ready")
        return self

    def detach(self) -> _TensorView:
        self._events.append("detach")
        return self

    def cpu(self) -> _TensorView:
        self._events.append("cpu")
        return self

    def numpy(self) -> np.ndarray:
        self._events.append("numpy")
        return self._value


class _Scene:
    def __init__(self, positions: np.ndarray, *, tensor_output: bool = False) -> None:
        self.positions = np.ascontiguousarray(positions).copy()
        self.pending = np.zeros_like(self.positions)
        self.events: list[str] = []
        self.entity = _Entity(self, tensor_output=tensor_output)

    def step(self) -> None:
        self.events.append("step")
        self.positions += self.pending
        self.pending.fill(0.0)


class _Entity:
    def __init__(self, scene: _Scene, *, tensor_output: bool) -> None:
        self._scene = scene
        self._tensor_output = tensor_output
        self.n_particles = int(scene.positions.shape[-2])

    def get_particles_pos(self, envs_idx: object | None = None) -> object:
        value = self._scene.positions
        if envs_idx is not None:
            indices = np.asarray(envs_idx, dtype=np.int64)
            value = value[indices]
        copied = np.ascontiguousarray(value).copy()
        if self._tensor_output:
            return _TensorView(copied, self._scene.events)
        return copied


def _frame_zero() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.1],
            [0.01, 0.0, 0.1],
            [0.02, 0.0, 0.1],
            [0.03, 0.0, 0.1],
        ],
        dtype=np.float64,
    )


def test_replay_reads_public_particle_state_and_steps() -> None:
    scene = _Scene(_frame_zero(), tensor_output=True)
    replay = GenesisMPMEntityReplayV1(
        scene=scene,
        entity=scene.entity,
        synchronize_callback=lambda: scene.events.append("sync"),
    )

    replay.synchronize()
    positions = replay.get_material_positions_m()
    positions[0, 0] = 10.0
    assert scene.positions[0, 0] == 0.0
    scene.pending[:, 2] = 0.001
    replay.step()

    assert replay.context is scene.entity
    assert scene.events[:5] == ["sync", "ready", "detach", "cpu", "numpy"]
    assert scene.events[-1] == "step"
    np.testing.assert_allclose(
        replay.get_material_positions_m()[:, 2],
        np.full(4, 0.101),
    )


def test_replay_selects_one_environment_from_batched_state() -> None:
    positions = np.stack([_frame_zero(), _frame_zero() + 1.0], axis=0)
    scene = _Scene(positions)
    replay = GenesisMPMEntityReplayV1(
        scene=scene,
        entity=scene.entity,
        env_index=np.int64(1),
    )
    np.testing.assert_array_equal(
        replay.get_material_positions_m(),
        _frame_zero() + 1.0,
    )
    assert replay.env_index == 1


def test_singleton_batch_is_accepted_without_environment_selection() -> None:
    scene = _Scene(_frame_zero()[None])
    replay = GenesisMPMEntityReplayV1(scene=scene, entity=scene.entity)
    np.testing.assert_array_equal(replay.get_material_positions_m(), _frame_zero())


def test_multi_environment_state_requires_exact_selection() -> None:
    scene = _Scene(np.stack([_frame_zero(), _frame_zero() + 1.0], axis=0))
    replay = GenesisMPMEntityReplayV1(scene=scene, entity=scene.entity)
    with pytest.raises(ValueError, match="exact env_index"):
        replay.get_material_positions_m()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"scene": object()}, TypeError, r"step\(\)"),
        ({"entity": object()}, TypeError, "get_particles_pos"),
        ({"synchronize_callback": None}, TypeError, "synchronize_callback"),
        ({"env_index": -1}, ValueError, "env_index"),
        ({"env_index": True}, ValueError, "env_index"),
        ({"env_index": 1.5}, ValueError, "env_index"),
    ),
)
def test_replay_rejects_invalid_configuration(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    scene = _Scene(_frame_zero())
    parameters: dict[str, Any] = {"scene": scene, "entity": scene.entity}
    parameters.update(kwargs)
    with pytest.raises(error, match=message):
        GenesisMPMEntityReplayV1(**parameters)


@pytest.mark.parametrize(
    ("positions", "message"),
    (
        (np.zeros((4, 2), dtype=np.float64), r"shape \(N,3\)"),
        (np.zeros((4, 3), dtype=np.int64), "floating point"),
        (
            np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
            "non-finite",
        ),
    ),
)
def test_replay_rejects_invalid_particle_positions(
    positions: np.ndarray,
    message: str,
) -> None:
    scene = _Scene(positions)
    replay = GenesisMPMEntityReplayV1(scene=scene, entity=scene.entity)
    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()


def test_replay_checks_reported_particle_count() -> None:
    scene = _Scene(_frame_zero())
    scene.entity.n_particles += 1
    replay = GenesisMPMEntityReplayV1(scene=scene, entity=scene.entity)
    with pytest.raises(ValueError, match="particle count"):
        replay.get_material_positions_m()

    scene.entity.n_particles = True
    with pytest.raises(TypeError, match="n_particles"):
        replay.get_material_positions_m()


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    concrete = cast(GenesisMPMEntityReplayV1, replay)
    scene = cast(_Scene, concrete.scene)
    scene.pending[:, 0] = 0.0005 * (transition_index + 1)


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    concrete = cast(GenesisMPMEntityReplayV1, replay)
    cast(_Scene, concrete.scene).pending.fill(0.0)


def test_genesis_replay_runs_through_portable_material_producer(
    tmp_path: Path,
) -> None:
    def replay_factory() -> GenesisMPMEntityReplayV1:
        scene = _Scene(_frame_zero())
        return GenesisMPMEntityReplayV1(scene=scene, entity=scene.entity)

    output = tmp_path / "genesis-mpm"
    artifact = produce_material_trajectory_backend(
        output_dir=output,
        backend_kind="genesis-mpm-v1",
        replay_factory=replay_factory,
        driven_control=_driven_control,
        zero_action_control=_zero_control,
        frame_count=4,
        material_query_indices=np.array([0, 2, 3], dtype=np.int64),
        action_support=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        engine_revision="a" * 40,
        engine_version="native-test-v1",
        producer_repository="example/genesis-producer",
        producer_revision="b" * 40,
        producer_version="producer-v1",
        producer_artifacts={"scene.py": "c" * 64},
        topology_sha256="d" * 64,
        device="cpu",
        device_name="contract-test-cpu",
        time_step_s=0.01,
        scene_id="genesis-replay-contract-test",
        model_kind="mpm-elastic-box",
        constitutive_model="elastic",
        integrator="genesis-mpm",
        solver="genesis-mpm",
        substeps=1,
    )

    assert artifact["backend_kind"] == "genesis-mpm-v1"
    physical = load_physical_rollout_archive(output / "physical-prediction.npz")
    assert np.max(physical["prediction_m"][-1, :, 0]) > 0.03
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(physical["frame_zero_points_m"][None], 4, axis=0),
    )
