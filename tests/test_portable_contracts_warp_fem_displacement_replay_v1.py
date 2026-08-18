from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    WarpFEMDisplacementReplayV1,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


class _WarpArray:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.ascontiguousarray(value).copy()

    def numpy(self) -> np.ndarray:
        return self.value


class _WarpField:
    def __init__(self, value: np.ndarray, *, degree: object = 1) -> None:
        self.degree = degree
        self.dof_values = _WarpArray(value)


class _WarpState:
    def __init__(self, reference: np.ndarray) -> None:
        self.reference = np.ascontiguousarray(reference).copy()
        self.field = _WarpField(np.zeros_like(self.reference))
        self.pending = np.zeros_like(self.reference)

    def step(self) -> None:
        self.field.dof_values.value += self.pending
        self.pending.fill(0.0)


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


def test_warp_fem_replay_adds_displacements_to_frozen_reference() -> None:
    state = _WarpState(_frame_zero())
    events: list[str] = []

    def step() -> None:
        events.append("step")
        state.step()

    replay = WarpFEMDisplacementReplayV1(
        displacement_field=state.field,
        reference_positions_m=state.reference,
        step_callback=step,
        synchronize_callback=lambda: events.append("sync"),
        context=state,
    )
    state.reference[:, 0] = 100.0
    replay.synchronize()
    positions = replay.get_material_positions_m()
    positions[0, 0] = 200.0
    state.pending[:, 2] = 0.001
    replay.step()

    assert replay.context is state
    assert events == ["sync", "step"]
    np.testing.assert_array_equal(
        replay.get_material_positions_m()[:, :2],
        _frame_zero()[:, :2],
    )
    np.testing.assert_allclose(
        replay.get_material_positions_m()[:, 2],
        np.full(4, 0.001),
    )


@pytest.mark.parametrize(
    ("field", "kwargs", "error", "message"),
    (
        (_WarpField(np.zeros((4, 3))), {"step_callback": None}, TypeError, "step_callback"),
        (
            _WarpField(np.zeros((4, 3))),
            {"synchronize_callback": None},
            TypeError,
            "synchronize_callback",
        ),
        (object(), {}, TypeError, "integer degree"),
        (_WarpField(np.zeros((4, 3)), degree=True), {}, TypeError, "integer degree"),
        (_WarpField(np.zeros((4, 3)), degree=2), {}, ValueError, "degree-1"),
    ),
)
def test_warp_fem_replay_rejects_invalid_surface_configuration(
    field: object,
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    parameters: dict[str, Any] = {
        "displacement_field": field,
        "reference_positions_m": _frame_zero(),
        "step_callback": lambda: None,
    }
    parameters.update(kwargs)
    with pytest.raises(error, match=message):
        WarpFEMDisplacementReplayV1(**parameters)


def test_warp_fem_replay_requires_native_array_surface() -> None:
    field = _WarpField(np.zeros((4, 3), dtype=np.float64))
    field.dof_values = np.zeros((4, 3), dtype=np.float64)
    with pytest.raises(TypeError, match=r"dof_values.*numpy"):
        WarpFEMDisplacementReplayV1(
            displacement_field=field,
            reference_positions_m=_frame_zero(),
            step_callback=lambda: None,
        )


@pytest.mark.parametrize(
    ("reference", "message"),
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
def test_warp_fem_replay_rejects_invalid_reference_positions(
    reference: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        WarpFEMDisplacementReplayV1(
            displacement_field=_WarpField(np.zeros((4, 3), dtype=np.float64)),
            reference_positions_m=reference,
            step_callback=lambda: None,
        )


@pytest.mark.parametrize(
    ("displacement", "message"),
    (
        (np.zeros((4, 2), dtype=np.float64), r"shape \(N,3\)"),
        (np.zeros((4, 3), dtype=np.int64), "floating point"),
        (
            np.array(
                [[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]],
                dtype=np.float64,
            ),
            "non-finite",
        ),
        (np.zeros((3, 3), dtype=np.float64), "rosters must match"),
    ),
)
def test_warp_fem_replay_rejects_invalid_displacement_state(
    displacement: np.ndarray,
    message: str,
) -> None:
    replay = WarpFEMDisplacementReplayV1(
        displacement_field=_WarpField(displacement),
        reference_positions_m=_frame_zero(),
        step_callback=lambda: None,
    )
    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    state = getattr(replay, "context", None)
    assert isinstance(state, _WarpState)
    state.pending[:, 2] = 0.0005 * (transition_index + 1)


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    state = getattr(replay, "context", None)
    assert isinstance(state, _WarpState)
    state.pending.fill(0.0)


def _producer_kwargs(tmp_path: Path, replay_factory: Any) -> dict[str, Any]:
    return {
        "output_dir": tmp_path / "warp-fem-v1",
        "backend_kind": "warp-fem-v1",
        "replay_factory": replay_factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_control,
        "frame_count": 4,
        "material_query_indices": np.array([0, 2, 3], dtype=np.int64),
        "action_support": np.array([0.0, 0.7, 1.0], dtype=np.float64),
        "engine_revision": "a" * 40,
        "engine_version": "native-test-v1",
        "producer_repository": "example/warp-fem-producer",
        "producer_revision": "b" * 40,
        "producer_version": "producer-v1",
        "producer_artifacts": {"scene.py": "c" * 64},
        "topology_sha256": "d" * 64,
        "device": "cuda",
        "device_name": "contract-test-device",
        "time_step_s": 0.01,
        "scene_id": "warp-fem-contract-test",
        "model_kind": "deformable-solid",
        "constitutive_model": "test-model",
        "integrator": "native-integrator",
        "solver": "native-solver",
        "substeps": 1,
    }


def test_warp_fem_adapter_runs_through_portable_material_producer(
    tmp_path: Path,
) -> None:
    def replay_factory() -> WarpFEMDisplacementReplayV1:
        state = _WarpState(_frame_zero())
        return WarpFEMDisplacementReplayV1(
            displacement_field=state.field,
            reference_positions_m=state.reference,
            step_callback=state.step,
            context=state,
        )

    artifact = produce_material_trajectory_backend(
        **_producer_kwargs(tmp_path, replay_factory)
    )
    assert artifact["backend_kind"] == "warp-fem-v1"
    physical = load_physical_rollout_archive(
        tmp_path / "warp-fem-v1" / "physical-prediction.npz"
    )
    assert np.max(physical["prediction_m"][-1, :, 2]) > 0.0
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(physical["frame_zero_points_m"][None], 4, axis=0),
    )
