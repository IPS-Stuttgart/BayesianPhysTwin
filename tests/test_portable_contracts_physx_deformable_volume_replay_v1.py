from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.material_trajectory_producer_v1 import (
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive
from bayesian_phystwin.physx_deformable_volume_replay_v1 import (
    PhysXDeformableVolumeReplayV1,
)


def _frame_zero_buffer() -> np.ndarray:
    return np.array(
        [
            [0.00, 0.0, 0.0, 1.0],
            [0.01, 0.0, 0.0, 1.0],
            [0.02, 0.0, 0.0, 0.5],
            [0.03, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


class _PhysXState:
    def __init__(self) -> None:
        self.buffer = _frame_zero_buffer()
        self.pending = np.zeros((len(self.buffer), 3), dtype=np.float32)
        self.events: list[str] = []

    def synchronize(self) -> None:
        self.events.append("sync")

    def read_buffer(self) -> np.ndarray:
        self.events.append("read")
        return self.buffer

    def step(self) -> None:
        self.events.append("step")
        self.buffer[:, :3] += self.pending
        self.pending.fill(0.0)


def _replay(state: _PhysXState) -> PhysXDeformableVolumeReplayV1:
    return PhysXDeformableVolumeReplayV1(
        simulation_vertex_count=len(state.buffer),
        read_sim_position_inv_mass_callback=state.read_buffer,
        advance_callback=state.step,
        synchronize_callback=state.synchronize,
        context=state,
    )


def test_physx_replay_reads_simulation_mesh_xyz_and_owns_result() -> None:
    state = _PhysXState()
    replay = _replay(state)

    replay.synchronize()
    positions = replay.get_material_positions_m()
    positions[0, 0] = 100.0
    state.pending[:, 2] = 0.001
    replay.step()

    assert replay.context is state
    assert state.events == ["sync", "read", "step"]
    assert state.buffer[0, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(
        replay.get_material_positions_m(),
        _frame_zero_buffer()[:, :3] + np.array([0.0, 0.0, 0.001]),
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"simulation_vertex_count": 0}, ValueError, "simulation_vertex_count"),
        ({"simulation_vertex_count": True}, ValueError, "simulation_vertex_count"),
        (
            {"read_sim_position_inv_mass_callback": None},
            TypeError,
            "read_sim_position_inv_mass_callback",
        ),
        ({"advance_callback": None}, TypeError, "advance_callback"),
        ({"synchronize_callback": None}, TypeError, "synchronize_callback"),
    ),
)
def test_physx_replay_rejects_invalid_surface_configuration(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    state = _PhysXState()
    parameters: dict[str, Any] = {
        "simulation_vertex_count": len(state.buffer),
        "read_sim_position_inv_mass_callback": state.read_buffer,
        "advance_callback": state.step,
        "synchronize_callback": state.synchronize,
    }
    parameters.update(kwargs)
    with pytest.raises(error, match=message):
        PhysXDeformableVolumeReplayV1(**parameters)


@pytest.mark.parametrize(
    ("buffer", "vertex_count", "message"),
    (
        (np.zeros((4, 3), dtype=np.float32), 4, "shape"),
        (np.zeros((3, 4), dtype=np.float32), 4, "shape"),
        (np.zeros((4, 4), dtype=np.int64), 4, "floating point"),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [np.nan, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            2,
            "non-finite",
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.01, 0.0, 0.0, -1.0],
                ],
                dtype=np.float32,
            ),
            2,
            "nonnegative",
        ),
    ),
)
def test_physx_replay_rejects_invalid_simulation_buffer(
    buffer: np.ndarray,
    vertex_count: int,
    message: str,
) -> None:
    replay = PhysXDeformableVolumeReplayV1(
        simulation_vertex_count=vertex_count,
        read_sim_position_inv_mass_callback=lambda: buffer,
        advance_callback=lambda: None,
        synchronize_callback=lambda: None,
    )
    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    state = getattr(replay, "context", None)
    assert isinstance(state, _PhysXState)
    state.pending[:, 2] = 0.0005 * (transition_index + 1)


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    state = getattr(replay, "context", None)
    assert isinstance(state, _PhysXState)
    state.pending.fill(0.0)


def _producer_kwargs(tmp_path: Path, replay_factory: Any) -> dict[str, Any]:
    return {
        "output_dir": tmp_path / "physx-fem-v1",
        "backend_kind": "physx-fem-v1",
        "replay_factory": replay_factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_control,
        "frame_count": 4,
        "material_query_indices": np.array([0, 2, 3], dtype=np.int64),
        "action_support": np.array([0.0, 0.7, 1.0], dtype=np.float32),
        "engine_revision": "a" * 40,
        "engine_version": "native-contract-test-v1",
        "producer_repository": "example/physx-deformable-producer",
        "producer_revision": "b" * 40,
        "producer_version": "producer-v1",
        "producer_artifacts": {"scene.cpp": "c" * 64},
        "topology_sha256": "d" * 64,
        "device": "cuda",
        "device_name": "contract-test-device",
        "time_step_s": 0.01,
        "scene_id": "physx-deformable-contract-test",
        "model_kind": "deformable-volume",
        "constitutive_model": "test-model",
        "integrator": "native-integrator",
        "solver": "native-solver",
        "substeps": 1,
        "engine_parameters": {
            "state_surface": "getSimPositionInvMassBufferD",
            "state_layout": "simulation-mesh-pxvec4-position-invmass",
        },
    }


def test_physx_adapter_runs_through_portable_material_producer(
    tmp_path: Path,
) -> None:
    def replay_factory() -> PhysXDeformableVolumeReplayV1:
        state = _PhysXState()
        return _replay(state)

    artifact = produce_material_trajectory_backend(
        **_producer_kwargs(tmp_path, replay_factory)
    )
    assert artifact["backend_kind"] == "physx-fem-v1"
    physical = load_physical_rollout_archive(
        tmp_path / "physx-fem-v1" / "physical-prediction.npz"
    )
    assert np.max(physical["prediction_m"][-1, :, 2]) > 0.0
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(physical["frame_zero_points_m"][None], 4, axis=0),
    )
