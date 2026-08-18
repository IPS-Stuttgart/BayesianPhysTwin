from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    MuJoCoFlexReplayV1,
    SofaMechanicalObjectReplayV1,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


class _PositionData:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class _SofaMechanicalObject:
    def __init__(self, value: np.ndarray) -> None:
        self.position = _PositionData(np.ascontiguousarray(value).copy())
        self.pending = np.zeros_like(self.position.value)


class _MuJoCoModel:
    def __init__(
        self,
        addresses: np.ndarray | list[int],
        counts: np.ndarray | list[int],
    ) -> None:
        self.flex_vertadr = np.asarray(addresses)
        self.flex_vertnum = np.asarray(counts)


class _MuJoCoData:
    def __init__(self, value: np.ndarray) -> None:
        self.flexvert_xpos = np.ascontiguousarray(value).copy()
        self.pending = np.zeros_like(self.flexvert_xpos)


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


def test_sofa_replay_reads_vec3_state_and_advances_registered_step() -> None:
    mechanical = _SofaMechanicalObject(_frame_zero())
    events: list[object] = []
    root = object()

    def animate(observed_root: object, dt: float) -> None:
        assert observed_root is root
        events.append(("step", dt))
        mechanical.position.value += mechanical.pending
        mechanical.pending.fill(0.0)

    replay = SofaMechanicalObjectReplayV1(
        mechanical_object=mechanical,
        root_node=root,
        animate_callback=animate,
        time_step_s=0.01,
        synchronize_callback=lambda: events.append("sync"),
        context=mechanical,
    )
    replay.synchronize()
    positions = replay.get_material_positions_m()
    positions[0, 0] = 10.0
    assert mechanical.position.value[0, 0] == 0.0
    mechanical.pending[:, 2] = 0.001
    replay.step()

    assert replay.context is mechanical
    assert events == ["sync", ("step", 0.01)]
    np.testing.assert_allclose(
        replay.get_material_positions_m()[:, 2],
        np.full(4, 0.001),
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"animate_callback": None}, TypeError, "animate_callback"),
        ({"synchronize_callback": None}, TypeError, "synchronize_callback"),
        ({"mechanical_object": object()}, TypeError, "position.value"),
        ({"time_step_s": True}, ValueError, "finite positive"),
        ({"time_step_s": -0.1}, ValueError, "finite positive"),
    ),
)
def test_sofa_replay_rejects_invalid_surface_configuration(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    parameters: dict[str, Any] = {
        "mechanical_object": _SofaMechanicalObject(_frame_zero()),
        "root_node": object(),
        "animate_callback": lambda root, dt: None,
        "time_step_s": 0.01,
    }
    parameters.update(kwargs)
    with pytest.raises(error, match=message):
        SofaMechanicalObjectReplayV1(**parameters)


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
def test_sofa_replay_rejects_invalid_mechanical_positions(
    value: np.ndarray,
    message: str,
) -> None:
    replay = SofaMechanicalObjectReplayV1(
        mechanical_object=_SofaMechanicalObject(value),
        root_node=object(),
        animate_callback=lambda root, dt: None,
        time_step_s=0.01,
    )
    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()


def test_mujoco_replay_selects_exact_flex_slice_and_steps() -> None:
    model = _MuJoCoModel([0, 2], [2, 2])
    data = _MuJoCoData(_frame_zero())
    events: list[str] = []

    def step(observed_model: object, observed_data: object) -> None:
        assert observed_model is model
        assert observed_data is data
        events.append("step")
        data.flexvert_xpos += data.pending
        data.pending.fill(0.0)

    replay = MuJoCoFlexReplayV1(
        model=model,
        data=data,
        flex_id=np.int64(1),
        step_callback=step,
        synchronize_callback=lambda: events.append("sync"),
        context=data,
    )
    replay.synchronize()
    positions = replay.get_material_positions_m()
    np.testing.assert_array_equal(positions, _frame_zero()[2:4])
    positions[0, 0] = 100.0
    assert data.flexvert_xpos[2, 0] == 0.02
    data.pending[:, 1] = 0.002
    replay.step()

    assert replay.flex_id == 1
    assert replay.context is data
    assert events == ["sync", "step"]
    np.testing.assert_allclose(
        replay.get_material_positions_m()[:, 1],
        np.full(2, 0.002),
    )


@pytest.mark.parametrize(
    ("model", "flex_id", "kwargs", "error", "message"),
    (
        (
            _MuJoCoModel([0], [1]),
            0,
            {"step_callback": None},
            TypeError,
            "step_callback",
        ),
        (
            _MuJoCoModel([0], [1]),
            0,
            {"synchronize_callback": None},
            TypeError,
            "synchronize_callback",
        ),
        (_MuJoCoModel([0], [1]), -1, {}, ValueError, "nonnegative integer"),
        (object(), 0, {}, TypeError, "flex_vertadr"),
        (_MuJoCoModel([[0]], [1]), 0, {}, ValueError, "one-dimensional"),
        (_MuJoCoModel([0.0], [1]), 0, {}, ValueError, "integer indices"),
        (_MuJoCoModel([0, 2], [1]), 0, {}, ValueError, "same shape"),
        (_MuJoCoModel([0], [1]), 1, {}, ValueError, "flex roster"),
        (_MuJoCoModel([-1], [1]), 0, {}, ValueError, "address must be nonnegative"),
        (_MuJoCoModel([0], [0]), 0, {}, ValueError, "at least one vertex"),
    ),
)
def test_mujoco_replay_rejects_invalid_model_configuration(
    model: object,
    flex_id: int,
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    parameters: dict[str, Any] = {
        "model": model,
        "data": _MuJoCoData(_frame_zero()),
        "flex_id": flex_id,
        "step_callback": lambda model, data: None,
    }
    parameters.update(kwargs)
    with pytest.raises(error, match=message):
        MuJoCoFlexReplayV1(**parameters)


def test_mujoco_replay_rejects_missing_or_out_of_bounds_data() -> None:
    model = _MuJoCoModel([2], [3])
    missing = MuJoCoFlexReplayV1(
        model=model,
        data=object(),
        flex_id=0,
        step_callback=lambda model, data: None,
    )
    with pytest.raises(TypeError, match="flexvert_xpos"):
        missing.get_material_positions_m()

    too_short = MuJoCoFlexReplayV1(
        model=model,
        data=_MuJoCoData(_frame_zero()),
        flex_id=0,
        step_callback=lambda model, data: None,
    )
    with pytest.raises(ValueError, match="slice exceeds"):
        too_short.get_material_positions_m()


def _producer_kwargs(
    tmp_path: Path,
    *,
    backend_kind: str,
    replay_factory: Any,
) -> dict[str, Any]:
    return {
        "output_dir": tmp_path / backend_kind,
        "backend_kind": backend_kind,
        "replay_factory": replay_factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_control,
        "frame_count": 4,
        "material_query_indices": np.array([0, 2, 3], dtype=np.int64),
        "action_support": np.array([0.0, 0.7, 1.0], dtype=np.float64),
        "engine_revision": "a" * 40,
        "engine_version": "native-test-v1",
        "producer_repository": "example/native-replay-producer",
        "producer_revision": "b" * 40,
        "producer_version": "producer-v1",
        "producer_artifacts": {"scene.py": "c" * 64},
        "topology_sha256": "d" * 64,
        "device": "cpu",
        "device_name": "contract-test-cpu",
        "time_step_s": 0.01,
        "scene_id": "native-replay-contract-test",
        "model_kind": "deformable-solid",
        "constitutive_model": "test-model",
        "integrator": "native-integrator",
        "solver": "native-solver",
        "substeps": 1,
    }


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    amplitude = 0.0005 * (transition_index + 1)
    context = getattr(replay, "context", None)
    if isinstance(context, _SofaMechanicalObject):
        context.pending[:, 2] = amplitude
    elif isinstance(context, _MuJoCoData):
        context.pending[:, 2] = amplitude
    else:
        raise AssertionError("unexpected replay context")


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    context = getattr(replay, "context", None)
    if isinstance(context, _SofaMechanicalObject):
        context.pending.fill(0.0)
    elif isinstance(context, _MuJoCoData):
        context.pending.fill(0.0)
    else:
        raise AssertionError("unexpected replay context")


def test_sofa_adapter_runs_through_portable_material_producer(tmp_path: Path) -> None:
    def replay_factory() -> SofaMechanicalObjectReplayV1:
        mechanical = _SofaMechanicalObject(_frame_zero())
        root = object()

        def animate(observed_root: object, dt: float) -> None:
            assert observed_root is root
            assert dt == 0.01
            mechanical.position.value += mechanical.pending
            mechanical.pending.fill(0.0)

        return SofaMechanicalObjectReplayV1(
            mechanical_object=mechanical,
            root_node=root,
            animate_callback=animate,
            time_step_s=0.01,
            context=mechanical,
        )

    artifact = produce_material_trajectory_backend(
        **_producer_kwargs(
            tmp_path,
            backend_kind="sofa-fem-v1",
            replay_factory=replay_factory,
        )
    )
    assert artifact["backend_kind"] == "sofa-fem-v1"
    physical = load_physical_rollout_archive(
        tmp_path / "sofa-fem-v1" / "physical-prediction.npz"
    )
    assert np.max(physical["prediction_m"][-1, :, 2]) > 0.0
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(physical["frame_zero_points_m"][None], 4, axis=0),
    )


def test_mujoco_adapter_runs_through_portable_material_producer(tmp_path: Path) -> None:
    def replay_factory() -> MuJoCoFlexReplayV1:
        model = _MuJoCoModel([0], [4])
        data = _MuJoCoData(_frame_zero())

        def step(observed_model: object, observed_data: object) -> None:
            assert observed_model is model
            assert observed_data is data
            data.flexvert_xpos += data.pending
            data.pending.fill(0.0)

        return MuJoCoFlexReplayV1(
            model=model,
            data=data,
            flex_id=0,
            step_callback=step,
            context=data,
        )

    artifact = produce_material_trajectory_backend(
        **_producer_kwargs(
            tmp_path,
            backend_kind="mujoco-flex-v1",
            replay_factory=replay_factory,
        )
    )
    assert artifact["backend_kind"] == "mujoco-flex-v1"
    physical = load_physical_rollout_archive(
        tmp_path / "mujoco-flex-v1" / "physical-prediction.npz"
    )
    assert np.max(physical["prediction_m"][-1, :, 2]) > 0.0


def test_position_validation_is_shared_by_mujoco_adapter() -> None:
    model = _MuJoCoModel([0], [2])
    data = _MuJoCoData(np.zeros((2, 3), dtype=np.int64))
    replay = MuJoCoFlexReplayV1(
        model=model,
        data=data,
        flex_id=0,
        step_callback=lambda model, data: None,
    )
    with pytest.raises(ValueError, match="floating point"):
        replay.get_material_positions_m()

    data.flexvert_xpos = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"shape \(N,3\)"):
        replay.get_material_positions_m()

    data.flexvert_xpos = np.zeros((2, 3), dtype=np.float64)
    data.flexvert_xpos[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        replay.get_material_positions_m()
