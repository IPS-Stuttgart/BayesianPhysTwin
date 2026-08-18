from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    PositionBasedDynamicsReplayV1,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


class _ParticleData:
    def __init__(self, value: np.ndarray) -> None:
        self.vertices = np.ascontiguousarray(value).copy()
        self.pending = np.zeros_like(self.vertices)

    def getVertices(self) -> np.ndarray:
        return self.vertices


class _SimulationModel:
    def __init__(self, value: np.ndarray) -> None:
        self.particles = _ParticleData(value)

    def getParticles(self) -> _ParticleData:
        return self.particles


class _TimeStep:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, model: _SimulationModel) -> None:
        model.particles.vertices += model.particles.pending
        model.particles.pending.fill(0.0)
        self.steps += 1


def _frame_zero() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.03, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_pbd_replay_reads_global_particle_state_and_steps_model() -> None:
    model = _SimulationModel(_frame_zero())
    time_step = _TimeStep()
    events: list[str] = []
    replay = PositionBasedDynamicsReplayV1(
        simulation_model=model,
        time_step=time_step,
        synchronize_callback=lambda: events.append("sync"),
        context=model,
    )

    replay.synchronize()
    positions = replay.get_material_positions_m()
    positions[0, 0] = 10.0
    assert model.particles.vertices[0, 0] == 0.0

    model.particles.pending[:, 2] = 0.001
    replay.step()

    assert replay.context is model
    assert events == ["sync"]
    assert time_step.steps == 1
    np.testing.assert_allclose(
        replay.get_material_positions_m()[:, 2],
        np.full(4, 0.001),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"simulation_model": object()}, "getParticles"),
        ({"time_step": object()}, "step"),
        ({"synchronize_callback": None}, "synchronize_callback"),
    ),
)
def test_pbd_replay_rejects_invalid_surface_configuration(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    parameters: dict[str, Any] = {
        "simulation_model": _SimulationModel(_frame_zero()),
        "time_step": _TimeStep(),
    }
    parameters.update(kwargs)
    with pytest.raises(TypeError, match=message):
        PositionBasedDynamicsReplayV1(**parameters)


def test_pbd_replay_rejects_missing_particle_vertex_surface() -> None:
    class Model:
        def getParticles(self) -> object:
            return object()

    replay = PositionBasedDynamicsReplayV1(
        simulation_model=Model(),
        time_step=_TimeStep(),
    )
    with pytest.raises(TypeError, match="getVertices"):
        replay.get_material_positions_m()


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (np.zeros((4, 2), dtype=np.float64), r"shape \(N,3\)"),
        (np.zeros((4, 3), dtype=np.int64), "floating point"),
        (
            np.array(
                [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
                dtype=np.float64,
            ),
            "non-finite",
        ),
    ),
)
def test_pbd_replay_rejects_invalid_particle_positions(
    value: np.ndarray,
    message: str,
) -> None:
    replay = PositionBasedDynamicsReplayV1(
        simulation_model=_SimulationModel(value),
        time_step=_TimeStep(),
    )
    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    model = replay.context
    if not isinstance(model, _SimulationModel):
        raise AssertionError("unexpected replay context")
    model.particles.pending[:, 2] = 0.0005 * (transition_index + 1)


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    model = replay.context
    if not isinstance(model, _SimulationModel):
        raise AssertionError("unexpected replay context")
    model.particles.pending.fill(0.0)


def _producer_kwargs(
    tmp_path: Path,
    replay_factory: Any,
) -> dict[str, Any]:
    return {
        "output_dir": tmp_path / "position-based-dynamics-v1",
        "backend_kind": "position-based-dynamics-v1",
        "replay_factory": replay_factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_control,
        "frame_count": 4,
        "material_query_indices": np.array([0, 2, 3], dtype=np.int64),
        "action_support": np.array([0.0, 0.7, 1.0], dtype=np.float64),
        "engine_revision": "a" * 40,
        "engine_version": "pyPBD-native-test-v1",
        "producer_repository": "example/pypbd-replay-producer",
        "producer_revision": "b" * 40,
        "producer_version": "producer-v1",
        "producer_artifacts": {"scene.py": "c" * 64},
        "topology_sha256": "d" * 64,
        "device": "cpu",
        "device_name": "contract-test-cpu",
        "time_step_s": 0.01,
        "scene_id": "pypbd-replay-contract-test",
        "model_kind": "deformable-particles",
        "constitutive_model": "xpbd-test-model",
        "integrator": "pbd-time-step-controller",
        "solver": "xpbd",
        "substeps": 1,
    }


def test_pbd_adapter_runs_through_portable_material_producer(tmp_path: Path) -> None:
    def replay_factory() -> PositionBasedDynamicsReplayV1:
        model = _SimulationModel(_frame_zero())
        return PositionBasedDynamicsReplayV1(
            simulation_model=model,
            time_step=_TimeStep(),
            context=model,
        )

    artifact = produce_material_trajectory_backend(
        **_producer_kwargs(tmp_path, replay_factory)
    )
    assert artifact["backend_kind"] == "position-based-dynamics-v1"
    physical = load_physical_rollout_archive(
        tmp_path
        / "position-based-dynamics-v1"
        / "physical-prediction.npz"
    )
    assert np.max(physical["prediction_m"][-1, :, 2]) > 0.0
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(physical["frame_zero_points_m"][None], 4, axis=0),
    )
