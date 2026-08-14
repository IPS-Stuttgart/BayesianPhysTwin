from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.genesis_mpm_producer_v1 as producer
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.external_physics_backend_v1 import (
    file_sha256,
    load_external_entity_rollout,
)


class _TensorView:
    def __init__(self, value: npt.NDArray[Any], events: list[object]) -> None:
        self._value = value
        self._events = events

    def detach(self) -> _TensorView:
        self._events.append("detach")
        return self

    def cpu(self) -> _TensorView:
        self._events.append("cpu")
        return self

    def numpy(self) -> npt.NDArray[Any]:
        self._events.append("numpy")
        return self._value


class _Entity:
    def __init__(self, scene: _Scene, *, tensor_output: bool) -> None:
        self._scene = scene
        self._tensor_output = tensor_output

    def get_particles_pos(self, envs_idx: object | None = None) -> object:
        positions = self._scene.positions
        if envs_idx is not None:
            indices = np.asarray(envs_idx, dtype=np.int64)
            positions = positions[indices]
        copied = np.ascontiguousarray(positions).copy()
        if self._tensor_output:
            return _TensorView(copied, self._scene.events)
        return copied


class _Scene:
    def __init__(
        self,
        positions: npt.NDArray[Any],
        *,
        tensor_output: bool,
        mutation: str | None = None,
    ) -> None:
        self.positions = np.ascontiguousarray(positions).copy()
        self.pending_z = 0.0
        self.events: list[object] = []
        self.mutation = mutation
        self.entity = _Entity(self, tensor_output=tensor_output)

    def step(self) -> None:
        self.events.append("step")
        self.positions[..., 2] += self.pending_z
        self.pending_z = 0.0
        mutation = self.mutation
        self.mutation = None
        if mutation == "drop-particle":
            if self.positions.ndim == 2:
                self.positions = self.positions[:-1]
            else:
                self.positions = self.positions[:, :-1]
        elif mutation == "float64":
            self.positions = self.positions.astype(np.float64)
        elif mutation == "non-finite":
            self.positions[..., 0, 0] = np.nan


def _base_positions(dtype: npt.DTypeLike = np.float32) -> npt.NDArray[Any]:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=dtype,
    )


def _factory(
    initial_states: Sequence[npt.NDArray[Any]] | None = None,
    *,
    tensor_output: bool = False,
    mutations: Sequence[str | None] = (None, None),
) -> tuple[producer.ReplayFactory, list[_Scene]]:
    states = tuple(initial_states or (_base_positions(), _base_positions()))
    scenes: list[_Scene] = []

    def build() -> tuple[_Scene, _Entity]:
        index = len(scenes)
        scene = _Scene(
            states[index],
            tensor_output=tensor_output,
            mutation=mutations[index],
        )
        scenes.append(scene)
        return scene, scene.entity

    return build, scenes


def _driven_control(
    transition_index: int,
    scene: producer.GenesisSceneV1,
    entity: producer.GenesisMPMEntityV1,
) -> None:
    del entity
    concrete = cast(_Scene, scene)
    concrete.events.append(("driven", transition_index))
    concrete.pending_z = 0.01 * (transition_index + 1)


def _zero_action_control(
    transition_index: int,
    scene: producer.GenesisSceneV1,
    entity: producer.GenesisMPMEntityV1,
) -> None:
    del entity
    concrete = cast(_Scene, scene)
    concrete.events.append(("zero", transition_index))
    concrete.pending_z = 0.0


def _producer_kwargs(path: Path, factory: producer.ReplayFactory) -> dict[str, Any]:
    return {
        "output_path": path,
        "replay_factory": factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_action_control,
        "frame_count": 4,
        "query_entity_indices": np.array([0, 2], dtype=np.int32),
        "action_support": [0, 1],
    }


def test_producer_runs_fresh_replays_and_preserves_particle_identity(
    tmp_path: Path,
) -> None:
    factory, scenes = _factory(tensor_output=True)
    output = tmp_path / "genesis-rollout.npz"

    result = producer.produce_genesis_mpm_entity_rollout(
        **_producer_kwargs(output, factory)
    )

    _, arrays = load_external_entity_rollout(output)
    np.testing.assert_allclose(
        arrays["driven_entity_positions_m"][:, 0, 2],
        np.array([0.0, 0.01, 0.03, 0.06], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        arrays["zero_action_entity_positions_m"],
        np.repeat(_base_positions()[None], 4, axis=0),
    )
    np.testing.assert_array_equal(
        arrays["query_entity_indices"],
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["action_support"],
        np.array([0.0, 1.0], dtype=np.float32),
    )

    assert len(scenes) == 2
    assert [
        event
        for event in scenes[0].events
        if event == "step" or isinstance(event, tuple)
    ] == [
        ("driven", 0),
        "step",
        ("driven", 1),
        "step",
        ("driven", 2),
        "step",
    ]
    assert {"detach", "cpu", "numpy"} <= set(scenes[0].events)
    assert result["profile_id"] == "genesis-mpm-v1"
    assert result["frame_count"] == 4
    assert result["entity_count"] == 3
    assert result["query_count"] == 2
    assert result["position_dtype"] == np.dtype(np.float32).str
    assert result["env_index"] is None
    assert result["independent_replay_count"] == 2
    assert result["raw_rollout_sha256"] == file_sha256(output)
    identity = {
        key: value for key, value in result.items() if key != "producer_result_id"
    }
    assert result["producer_result_id"] == content_id(identity)


def test_producer_rejects_cached_replay_objects(tmp_path: Path) -> None:
    scene = _Scene(_base_positions(), tensor_output=False)

    def cached_factory() -> tuple[_Scene, _Entity]:
        scene.positions = _base_positions()
        scene.pending_z = 0.0
        return scene, scene.entity

    with pytest.raises(ValueError, match="fresh Genesis scene and entity"):
        producer.produce_genesis_mpm_entity_rollout(
            output_path=tmp_path / "raw.npz",
            replay_factory=cached_factory,
            driven_control=_driven_control,
            zero_action_control=_zero_action_control,
            frame_count=2,
            query_entity_indices=[0],
            action_support=[1.0],
        )


def test_producer_selects_one_batched_environment(tmp_path: Path) -> None:
    first = np.stack([_base_positions(), _base_positions() + 1.0], axis=0)
    factory, _ = _factory((first, first.copy()))
    output = tmp_path / "selected.npz"

    result = producer.produce_genesis_mpm_entity_rollout(
        output_path=output,
        replay_factory=factory,
        driven_control=_driven_control,
        zero_action_control=_zero_action_control,
        frame_count=2,
        query_entity_indices=[1],
        action_support=[0.5],
        env_index=1,
    )

    _, arrays = load_external_entity_rollout(output)
    np.testing.assert_array_equal(
        arrays["driven_entity_positions_m"][0],
        _base_positions() + 1.0,
    )
    assert result["env_index"] == 1


def test_singleton_batch_is_accepted_without_environment_selection(
    tmp_path: Path,
) -> None:
    initial = _base_positions()[None]
    factory, _ = _factory((initial, initial.copy()))
    output = tmp_path / "singleton.npz"
    producer.produce_genesis_mpm_entity_rollout(
        output_path=output,
        replay_factory=factory,
        driven_control=_driven_control,
        zero_action_control=_zero_action_control,
        frame_count=2,
        query_entity_indices=[0],
        action_support=[1.0],
    )
    _, arrays = load_external_entity_rollout(output)
    assert arrays["driven_entity_positions_m"].shape == (2, 3, 3)


def test_producer_is_deterministic_and_refuses_clobber(tmp_path: Path) -> None:
    first_factory, _ = _factory()
    second_factory, _ = _factory()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_result = producer.produce_genesis_mpm_entity_rollout(
        **_producer_kwargs(first, first_factory)
    )
    second_result = producer.produce_genesis_mpm_entity_rollout(
        **_producer_kwargs(second, second_factory)
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_result["producer_result_id"] == second_result["producer_result_id"]

    called = False

    def forbidden_factory() -> tuple[_Scene, _Entity]:
        nonlocal called
        called = True
        raise AssertionError("the replay must not start for an existing output")

    with pytest.raises(FileExistsError):
        producer.produce_genesis_mpm_entity_rollout(
            **_producer_kwargs(first, forbidden_factory)
        )
    assert called is False


def test_producer_rejects_symlink_output_paths_before_replay(tmp_path: Path) -> None:
    broken_link = tmp_path / "broken.npz"
    broken_link.symlink_to(tmp_path / "missing.npz")
    factory, scenes = _factory()
    with pytest.raises(FileExistsError):
        producer.produce_genesis_mpm_entity_rollout(
            **_producer_kwargs(broken_link, factory)
        )
    assert scenes == []

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    factory, scenes = _factory()
    with pytest.raises(ValueError, match="traverse a symlink"):
        producer.produce_genesis_mpm_entity_rollout(
            **_producer_kwargs(linked_directory / "raw.npz", factory)
        )
    assert scenes == []


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (1, "at least two"),
        (True, "at least two"),
        (2.0, "at least two"),
    ],
)
def test_producer_rejects_invalid_frame_counts(
    tmp_path: Path,
    value: object,
    message: str,
) -> None:
    factory, _ = _factory()
    kwargs = _producer_kwargs(tmp_path / "raw.npz", factory)
    kwargs["frame_count"] = value
    with pytest.raises(ValueError, match=message):
        producer.produce_genesis_mpm_entity_rollout(**kwargs)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_producer_rejects_invalid_environment_indices(
    tmp_path: Path,
    value: object,
) -> None:
    factory, _ = _factory()
    kwargs = _producer_kwargs(tmp_path / "raw.npz", factory)
    kwargs["env_index"] = value
    with pytest.raises(ValueError, match="env_index"):
        producer.produce_genesis_mpm_entity_rollout(**kwargs)


def test_producer_rejects_noncallable_entrypoints(tmp_path: Path) -> None:
    factory, _ = _factory()
    kwargs = _producer_kwargs(tmp_path / "raw.npz", factory)
    kwargs["replay_factory"] = None
    with pytest.raises(ValueError, match="replay_factory"):
        producer.produce_genesis_mpm_entity_rollout(**kwargs)

    kwargs = _producer_kwargs(tmp_path / "raw.npz", factory)
    kwargs["driven_control"] = None
    with pytest.raises(ValueError, match="controls"):
        producer.produce_genesis_mpm_entity_rollout(**kwargs)

    kwargs = _producer_kwargs(tmp_path / "raw.npz", factory)
    kwargs["zero_action_control"] = None
    with pytest.raises(ValueError, match="controls"):
        producer.produce_genesis_mpm_entity_rollout(**kwargs)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: [], "must return"),
        (lambda: (object(),), "must return"),
        (lambda: (object(), _Entity), "scene must expose"),
        (
            lambda: (
                _Scene(_base_positions(), tensor_output=False),
                object(),
            ),
            "entity must expose",
        ),
    ],
)
def test_producer_rejects_invalid_replay_surfaces(
    tmp_path: Path,
    factory: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        producer.produce_genesis_mpm_entity_rollout(
            output_path=tmp_path / "raw.npz",
            replay_factory=factory,
            driven_control=_driven_control,
            zero_action_control=_zero_action_control,
            frame_count=2,
            query_entity_indices=[0],
            action_support=[1.0],
        )


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.zeros((2, 3, 3), dtype=np.float32), "exact env_index"),
        (np.zeros((3, 2), dtype=np.float32), "shape"),
        (np.zeros((3, 3), dtype=np.int64), "floating point"),
        (np.full((3, 3), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_producer_rejects_invalid_particle_position_outputs(
    tmp_path: Path,
    positions: npt.NDArray[Any],
    message: str,
) -> None:
    factory, _ = _factory((positions, positions.copy()))
    with pytest.raises(ValueError, match=message):
        producer.produce_genesis_mpm_entity_rollout(
            output_path=tmp_path / "raw.npz",
            replay_factory=factory,
            driven_control=_driven_control,
            zero_action_control=_zero_action_control,
            frame_count=2,
            query_entity_indices=[0],
            action_support=[1.0],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("drop-particle", "changed particle count"),
        ("float64", "changed particle dtype"),
        ("non-finite", "non-finite"),
    ],
)
def test_producer_rejects_in_replay_identity_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    factory, _ = _factory(mutations=(mutation, None))
    with pytest.raises(ValueError, match=message):
        producer.produce_genesis_mpm_entity_rollout(
            **_producer_kwargs(tmp_path / "raw.npz", factory)
        )


def test_producer_rejects_cross_replay_shape_dtype_and_frame_zero_drift(
    tmp_path: Path,
) -> None:
    cases = [
        (
            (_base_positions(), _base_positions()[:-1]),
            "shapes or dtypes differ",
        ),
        (
            (_base_positions(), _base_positions(np.float64)),
            "shapes or dtypes differ",
        ),
        (
            (_base_positions(), _base_positions() + 0.001),
            "differ at frame zero",
        ),
    ]
    for index, (states, message) in enumerate(cases):
        factory, _ = _factory(states)
        with pytest.raises(ValueError, match=message):
            producer.produce_genesis_mpm_entity_rollout(
                **_producer_kwargs(tmp_path / f"raw-{index}.npz", factory)
            )


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([], "must not be empty"),
        ([0, 0], "must be unique"),
        ([3], "exceeds"),
        ([True], "integer vector"),
        ([0.0], "integer vector"),
        ([[0]], "integer vector"),
    ],
)
def test_producer_rejects_invalid_query_indices(
    tmp_path: Path,
    indices: object,
    message: str,
) -> None:
    factory, _ = _factory()
    kwargs = _producer_kwargs(tmp_path / "raw.npz", factory)
    kwargs["query_entity_indices"] = indices
    with pytest.raises(ValueError, match=message):
        producer.produce_genesis_mpm_entity_rollout(**kwargs)


@pytest.mark.parametrize(
    ("support", "message"),
    [
        ([], "matching the query count"),
        ([True, False], "numeric vector"),
        (["0", "1"], "numeric vector"),
        ([[0.0, 1.0]], "matching the query count"),
        ([0.0, np.nan], "finite vector"),
        ([0.0, 1.1], "finite vector"),
    ],
)
def test_producer_rejects_invalid_action_support(
    tmp_path: Path,
    support: object,
    message: str,
) -> None:
    factory, _ = _factory()
    kwargs = _producer_kwargs(tmp_path / "raw.npz", factory)
    kwargs["action_support"] = support
    with pytest.raises(ValueError, match=message):
        producer.produce_genesis_mpm_entity_rollout(**kwargs)


def test_publish_cleanup_survives_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, _ = _factory()

    def racing_link(source: object, destination: object) -> None:
        del source, destination
        raise FileExistsError("racing writer")

    monkeypatch.setattr(producer.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="racing writer"):
        producer.produce_genesis_mpm_entity_rollout(
            **_producer_kwargs(tmp_path / "raw.npz", factory)
        )
    assert not tuple(tmp_path.glob(".raw.npz.*.tmp"))


def test_directory_fsync_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(path: object, flags: object) -> int:
        del path, flags
        raise OSError("unsupported")

    monkeypatch.setattr(producer.os, "open", fail_open)
    producer._fsync_directory(tmp_path)

    monkeypatch.undo()

    def fail_fsync(descriptor: int) -> None:
        del descriptor
        raise OSError("unsupported")

    monkeypatch.setattr(producer.os, "fsync", fail_fsync)
    producer._fsync_directory(tmp_path)


def test_backend_producer_writes_bound_genesis_runtime(tmp_path: Path) -> None:
    factory, _ = _factory()
    raw = tmp_path / "raw.npz"
    runtime = tmp_path / "runtime.json"

    result = producer.produce_genesis_mpm_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=runtime,
        replay_factory=factory,
        driven_control=_driven_control,
        zero_action_control=_zero_action_control,
        frame_count=3,
        query_entity_indices=[0, 2],
        action_support=[0.0, 1.0],
        engine_revision="a" * 40,
        engine_version="test-genesis",
        producer_repository="IPS-Stuttgart/BayesianPhysTwin",
        producer_revision="b" * 40,
        coordinate_frame="right-handed-z-up-world-v1",
        time_step_s=0.01,
        topology_sha256="c" * 64,
        material_model="neo-hookean",
        observation_end_frame_exclusive=1,
        parameterization={"young_modulus_pa": 50000.0, "poisson_ratio": 0.3},
        producer_artifacts={"configs/genesis-scene.json": "d" * 64},
    )

    assert raw.is_file()
    assert runtime.is_file()
    assert result["rollout"]["profile_id"] == "genesis-mpm-v1"
    assert result["runtime"]["backend_profile"]["profile_id"] == ("genesis-mpm-v1")
    assert result["runtime"]["frame_count"] == 3
    assert result["runtime"]["query_count"] == 2
    assert result["runtime"]["raw_rollout_sha256"] == file_sha256(raw)
    assert result["runtime"]["information_boundary"] == {
        "observation_end_frame_exclusive": 1,
        "future_observations_used": False,
        "outcomes_used_for_selection": False,
        "target_outcomes_used": False,
        "known_action_used": True,
    }


def test_backend_preflights_output_paths_before_replay(tmp_path: Path) -> None:
    called = False

    def forbidden_factory() -> tuple[_Scene, _Entity]:
        nonlocal called
        called = True
        scene = _Scene(_base_positions(), tensor_output=False)
        return scene, scene.entity

    same = tmp_path / "same"
    with pytest.raises(ValueError, match="paths must differ"):
        producer.produce_genesis_mpm_backend(
            raw_rollout_path=same,
            runtime_manifest_path=same,
            replay_factory=forbidden_factory,
            driven_control=_driven_control,
            zero_action_control=_zero_action_control,
            frame_count=2,
            query_entity_indices=[0],
            action_support=[1.0],
            engine_revision="a" * 40,
            engine_version="test",
            producer_repository="owner/producer",
            producer_revision="b" * 40,
            coordinate_frame="world",
            time_step_s=0.01,
            topology_sha256="c" * 64,
            material_model="elastic",
            observation_end_frame_exclusive=1,
        )
    assert called is False

    runtime = tmp_path / "runtime.json"
    runtime.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        producer.produce_genesis_mpm_backend(
            raw_rollout_path=tmp_path / "raw.npz",
            runtime_manifest_path=runtime,
            replay_factory=forbidden_factory,
            driven_control=_driven_control,
            zero_action_control=_zero_action_control,
            frame_count=2,
            query_entity_indices=[0],
            action_support=[1.0],
            engine_revision="a" * 40,
            engine_version="test",
            producer_repository="owner/producer",
            producer_revision="b" * 40,
            coordinate_frame="world",
            time_step_s=0.01,
            topology_sha256="c" * 64,
            material_model="elastic",
            observation_end_frame_exclusive=1,
        )
    assert called is False
    assert not (tmp_path / "raw.npz").exists()

    raw = tmp_path / "raw-existing.npz"
    raw.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        producer.produce_genesis_mpm_backend(
            raw_rollout_path=raw,
            runtime_manifest_path=tmp_path / "other-runtime.json",
            replay_factory=forbidden_factory,
            driven_control=_driven_control,
            zero_action_control=_zero_action_control,
            frame_count=2,
            query_entity_indices=[0],
            action_support=[1.0],
            engine_revision="a" * 40,
            engine_version="test",
            producer_repository="owner/producer",
            producer_revision="b" * 40,
            coordinate_frame="world",
            time_step_s=0.01,
            topology_sha256="c" * 64,
            material_model="elastic",
            observation_end_frame_exclusive=1,
        )
    assert called is False
