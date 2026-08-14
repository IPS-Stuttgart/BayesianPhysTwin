from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.jax_fem_producer_v1 as producer
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.external_physics_backend_v1 import (
    file_sha256,
    load_external_entity_rollout,
)


class _JaxArray:
    def __init__(
        self,
        value: npt.NDArray[Any],
        events: list[object],
        *,
        return_none: bool = False,
    ) -> None:
        self._value = value
        self._events = events
        self._return_none = return_none

    def block_until_ready(self) -> _JaxArray | None:
        self._events.append("ready")
        return None if self._return_none else self

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[Any]:
        value = np.asarray(self._value, dtype=dtype)
        return value.copy() if copy else value


class _Replay:
    def __init__(
        self,
        reference: npt.NDArray[Any],
        *,
        solution_mode: str = "list",
        displacement_field_index: int = 0,
        mutation: str | None = None,
        jax_like: bool = True,
        return_none_when_ready: bool = False,
    ) -> None:
        self.reference = np.ascontiguousarray(reference).copy()
        self.solution_mode = solution_mode
        self.displacement_field_index = displacement_field_index
        self.mutation = mutation
        self.jax_like = jax_like
        self.return_none_when_ready = return_none_when_ready
        self.pending_z_m = 0.0
        self.current_z_m = 0.0
        self.events: list[object] = []

    def _value(self, value: npt.NDArray[Any]) -> object:
        copied = np.ascontiguousarray(value).copy()
        if not self.jax_like:
            return copied
        return _JaxArray(
            copied,
            self.events,
            return_none=self.return_none_when_ready,
        )

    def get_reference_points_m(self) -> object:
        return self._value(self.reference)

    def solve(self) -> object:
        self.events.append("solve")
        self.current_z_m += self.pending_z_m
        self.pending_z_m = 0.0
        displacement = np.zeros_like(self.reference)
        displacement[:, 2] = self.current_z_m

        mutation = self.mutation
        self.mutation = None
        if mutation == "drop-node":
            displacement = displacement[:-1]
        elif mutation == "float64":
            displacement = displacement.astype(np.float64)
        elif mutation == "non-finite":
            displacement[0, 0] = np.nan
        elif mutation == "reference-drift":
            self.reference = self.reference + np.asarray(
                0.001,
                dtype=self.reference.dtype,
            )
        elif mutation == "overflow":
            displacement.fill(np.finfo(displacement.dtype).max)

        selected = self._value(displacement)
        if self.solution_mode == "direct":
            return selected
        if self.solution_mode == "empty":
            return []
        auxiliary = self._value(
            np.zeros((len(self.reference), 1), dtype=self.reference.dtype)
        )
        fields = [selected, auxiliary]
        if self.displacement_field_index == 1:
            fields.reverse()
        if self.solution_mode == "tuple":
            return tuple(fields)
        return fields


def _base_points(dtype: npt.DTypeLike = np.float32) -> npt.NDArray[Any]:
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
    **replay_options: Any,
) -> tuple[producer.ReplayFactory, list[_Replay]]:
    states = tuple(initial_states or (_base_points(), _base_points()))
    replays: list[_Replay] = []

    def build() -> _Replay:
        replay = _Replay(states[len(replays)], **replay_options)
        replays.append(replay)
        return replay

    return build, replays


def _driven_control(
    transition_index: int,
    replay: producer.JaxFemReplayV1,
) -> None:
    concrete = cast(_Replay, replay)
    concrete.events.append(("driven", transition_index))
    concrete.pending_z_m = 0.01 * (transition_index + 1)


def _zero_action_control(
    transition_index: int,
    replay: producer.JaxFemReplayV1,
) -> None:
    concrete = cast(_Replay, replay)
    concrete.events.append(("zero", transition_index))
    concrete.pending_z_m = 0.0


def _producer_kwargs(
    path: Path,
    factory: producer.ReplayFactory,
) -> dict[str, Any]:
    return {
        "output_path": path,
        "replay_factory": factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_action_control,
        "frame_count": 4,
        "query_entity_indices": np.array([0, 2], dtype=np.int32),
        "action_support": [0, 1],
    }


def test_producer_runs_fresh_solve_sequences_with_jax_synchronization(
    tmp_path: Path,
) -> None:
    factory, replays = _factory()
    output = tmp_path / "jax-fem-rollout.npz"

    result = producer.produce_jax_fem_entity_rollout(
        **_producer_kwargs(output, factory)
    )

    _, arrays = load_external_entity_rollout(output)
    np.testing.assert_allclose(
        arrays["driven_entity_positions_m"][:, 0, 2],
        np.array([0.0, 0.01, 0.03, 0.06], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        arrays["zero_action_entity_positions_m"],
        np.repeat(_base_points()[None], 4, axis=0),
    )
    np.testing.assert_array_equal(
        arrays["query_entity_indices"],
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["action_support"],
        np.array([0.0, 1.0], dtype=np.float32),
    )

    assert len(replays) == 2
    action_events = [
        event
        for event in replays[0].events
        if event == "solve" or isinstance(event, tuple)
    ]
    assert action_events == [
        ("driven", 0),
        "solve",
        ("driven", 1),
        "solve",
        ("driven", 2),
        "solve",
    ]
    assert "ready" in replays[0].events
    assert result["profile_id"] == "jax-fem-v1"
    assert result["frame_count"] == 4
    assert result["entity_count"] == 3
    assert result["query_count"] == 2
    assert result["position_dtype"] == np.dtype(np.float32).str
    assert result["solution_index"] == 0
    assert result["solution_semantics"] == (
        "nodal-displacement-from-fixed-reference"
    )
    assert result["independent_replay_count"] == 2
    assert result["action_timing"] == "control-before-solve"
    assert result["raw_rollout_sha256"] == file_sha256(output)
    identity = {
        key: value for key, value in result.items() if key != "producer_result_id"
    }
    assert result["producer_result_id"] == content_id(identity)


def test_direct_numpy_solution_uses_default_field_index(tmp_path: Path) -> None:
    factory, replays = _factory(solution_mode="direct", jax_like=False)
    output = tmp_path / "direct.npz"
    producer.produce_jax_fem_entity_rollout(
        **_producer_kwargs(output, factory)
    )
    assert "ready" not in replays[0].events


def test_tuple_solution_selects_declared_displacement_field(tmp_path: Path) -> None:
    factory, _ = _factory(
        solution_mode="tuple",
        displacement_field_index=1,
    )
    output = tmp_path / "tuple.npz"
    arguments = _producer_kwargs(output, factory)
    arguments["solution_index"] = 1
    producer.produce_jax_fem_entity_rollout(**arguments)


def test_jax_synchronization_may_return_none(tmp_path: Path) -> None:
    factory, _ = _factory(return_none_when_ready=True)
    producer.produce_jax_fem_entity_rollout(
        **_producer_kwargs(tmp_path / "none.npz", factory)
    )


def test_producer_rejects_cached_replay_object(tmp_path: Path) -> None:
    replay = _Replay(_base_points())

    with pytest.raises(ValueError, match="fresh JAX-FEM replay objects"):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(tmp_path / "raw.npz", lambda: replay)
        )


@pytest.mark.parametrize(
    ("states", "message"),
    [
        ((_base_points(), _base_points()[:-1]), "shapes or dtypes differ"),
        (
            (_base_points(), _base_points(np.float64)),
            "shapes or dtypes differ",
        ),
        (
            (_base_points(), _base_points() + np.float32(0.001)),
            "differ in reference mesh points",
        ),
    ],
)
def test_producer_rejects_cross_replay_reference_mismatch(
    tmp_path: Path,
    states: tuple[npt.NDArray[Any], npt.NDArray[Any]],
    message: str,
) -> None:
    factory, _ = _factory(states)
    with pytest.raises(ValueError, match=message):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(tmp_path / "raw.npz", factory)
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("drop-node", "displacement shape"),
        ("float64", "displacement dtype"),
        ("non-finite", "displacement contains non-finite"),
        ("reference-drift", "changed reference mesh points"),
    ],
)
def test_producer_rejects_invalid_or_drifting_solve_outputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    factory, _ = _factory(mutation=mutation)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match=message):
            producer.produce_jax_fem_entity_rollout(
                **_producer_kwargs(tmp_path / "raw.npz", factory)
            )


def test_producer_rejects_nonfinite_absolute_positions(tmp_path: Path) -> None:
    limit = np.finfo(np.float32).max
    reference = np.full((3, 3), limit * np.float32(0.75), dtype=np.float32)
    factory, _ = _factory((reference, reference.copy()), mutation="overflow")
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="absolute positions contain non-finite"):
            producer.produce_jax_fem_entity_rollout(
                **_producer_kwargs(tmp_path / "raw.npz", factory)
            )


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        (np.zeros((0, 3), dtype=np.float32), "shape"),
        (np.zeros((3, 2), dtype=np.float32), "shape"),
        (np.zeros((3, 3), dtype=np.int64), "floating point"),
        (np.full((3, 3), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_producer_rejects_invalid_reference_points(
    tmp_path: Path,
    reference: npt.NDArray[Any],
    message: str,
) -> None:
    factory, _ = _factory((reference, reference.copy()))
    with pytest.raises(ValueError, match=message):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(tmp_path / "raw.npz", factory)
        )


def test_producer_rejects_invalid_replay_surfaces(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="get_reference_points_m"):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(tmp_path / "missing-reference.npz", lambda: object())
        )

    class _ReferenceOnly:
        def get_reference_points_m(self) -> npt.NDArray[Any]:
            return _base_points()

    with pytest.raises(ValueError, match="solve"):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(
                tmp_path / "missing-solve.npz",
                lambda: _ReferenceOnly(),
            )
        )


@pytest.mark.parametrize(
    ("mode", "solution_index", "message"),
    [
        ("empty", 0, "returned no solution fields"),
        ("list", 2, "exceeds JAX-FEM solution field count"),
        ("direct", 1, "must be zero for a direct"),
    ],
)
def test_producer_rejects_invalid_solution_selection(
    tmp_path: Path,
    mode: str,
    solution_index: int,
    message: str,
) -> None:
    factory, _ = _factory(solution_mode=mode)
    arguments = _producer_kwargs(tmp_path / "raw.npz", factory)
    arguments["solution_index"] = solution_index
    with pytest.raises(ValueError, match=message):
        producer.produce_jax_fem_entity_rollout(**arguments)


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_producer_rejects_invalid_solution_index(
    tmp_path: Path,
    value: object,
) -> None:
    factory, _ = _factory()
    arguments = _producer_kwargs(tmp_path / "raw.npz", factory)
    arguments["solution_index"] = value
    with pytest.raises(ValueError, match="solution_index"):
        producer.produce_jax_fem_entity_rollout(**arguments)


@pytest.mark.parametrize("value", [1, True, 2.0])
def test_producer_rejects_invalid_frame_count(
    tmp_path: Path,
    value: object,
) -> None:
    factory, _ = _factory()
    arguments = _producer_kwargs(tmp_path / "raw.npz", factory)
    arguments["frame_count"] = value
    with pytest.raises(ValueError, match="at least two"):
        producer.produce_jax_fem_entity_rollout(**arguments)


def test_producer_rejects_noncallable_entrypoints(tmp_path: Path) -> None:
    factory, _ = _factory()
    arguments = _producer_kwargs(tmp_path / "factory.npz", factory)
    arguments["replay_factory"] = None
    with pytest.raises(ValueError, match="replay_factory"):
        producer.produce_jax_fem_entity_rollout(**arguments)

    arguments = _producer_kwargs(tmp_path / "driven.npz", factory)
    arguments["driven_control"] = None
    with pytest.raises(ValueError, match="controls"):
        producer.produce_jax_fem_entity_rollout(**arguments)

    arguments = _producer_kwargs(tmp_path / "zero.npz", factory)
    arguments["zero_action_control"] = None
    with pytest.raises(ValueError, match="controls"):
        producer.produce_jax_fem_entity_rollout(**arguments)


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([], "must not be empty"),
        ([0, 0], "must be unique"),
        ([3], "exceeds JAX-FEM node count"),
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
    arguments = _producer_kwargs(tmp_path / "raw.npz", factory)
    arguments["query_entity_indices"] = indices
    with pytest.raises(ValueError, match=message):
        producer.produce_jax_fem_entity_rollout(**arguments)


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
    arguments = _producer_kwargs(tmp_path / "raw.npz", factory)
    arguments["action_support"] = support
    with pytest.raises(ValueError, match=message):
        producer.produce_jax_fem_entity_rollout(**arguments)


def test_producer_is_deterministic_and_refuses_clobber(tmp_path: Path) -> None:
    first_factory, _ = _factory()
    second_factory, _ = _factory()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_result = producer.produce_jax_fem_entity_rollout(
        **_producer_kwargs(first, first_factory)
    )
    second_result = producer.produce_jax_fem_entity_rollout(
        **_producer_kwargs(second, second_factory)
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_result["producer_result_id"] == second_result["producer_result_id"]

    called = False

    def forbidden_factory() -> _Replay:
        nonlocal called
        called = True
        raise AssertionError("the replay must not start for an existing output")

    with pytest.raises(FileExistsError):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(first, forbidden_factory)
        )
    assert called is False


def test_producer_rejects_symlink_output_paths_before_replay(tmp_path: Path) -> None:
    broken_link = tmp_path / "broken.npz"
    broken_link.symlink_to(tmp_path / "missing.npz")
    factory, replays = _factory()
    with pytest.raises(FileExistsError):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(broken_link, factory)
        )
    assert replays == []

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    factory, replays = _factory()
    with pytest.raises(ValueError, match="traverse a symlink"):
        producer.produce_jax_fem_entity_rollout(
            **_producer_kwargs(linked_directory / "raw.npz", factory)
        )
    assert replays == []


def test_publish_cleanup_survives_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, _ = _factory()

    def racing_link(source: object, destination: object) -> None:
        del source, destination
        raise FileExistsError("racing writer")

    monkeypatch.setattr(producer.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="racing writer"):
        producer.produce_jax_fem_entity_rollout(
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


def test_backend_producer_writes_bound_jax_fem_runtime(tmp_path: Path) -> None:
    factory, _ = _factory()
    raw = tmp_path / "raw.npz"
    runtime = tmp_path / "runtime.json"

    result = producer.produce_jax_fem_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=runtime,
        replay_factory=factory,
        driven_control=_driven_control,
        zero_action_control=_zero_action_control,
        frame_count=3,
        query_entity_indices=[0, 2],
        action_support=[0.0, 1.0],
        engine_revision="a" * 40,
        engine_version="test-jax-fem",
        producer_repository="IPS-Stuttgart/BayesianPhysTwin",
        producer_revision="b" * 40,
        coordinate_frame="right-handed-z-up-world-v1",
        time_step_s=0.01,
        topology_sha256="c" * 64,
        material_model="neo-hookean",
        observation_end_frame_exclusive=1,
        parameterization={"young_modulus_pa": 50000.0, "poisson_ratio": 0.3},
        producer_artifacts={"configs/jax-fem-scene.json": "d" * 64},
    )

    assert raw.is_file()
    assert runtime.is_file()
    assert result["rollout"]["profile_id"] == "jax-fem-v1"
    assert result["runtime"]["backend_profile"]["profile_id"] == "jax-fem-v1"
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

    def forbidden_factory() -> _Replay:
        nonlocal called
        called = True
        raise AssertionError("backend preflight must run before the solver")

    common: dict[str, Any] = {
        "replay_factory": forbidden_factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_action_control,
        "frame_count": 2,
        "query_entity_indices": [0],
        "action_support": [1.0],
        "engine_revision": "a" * 40,
        "engine_version": "test",
        "producer_repository": "owner/producer",
        "producer_revision": "b" * 40,
        "coordinate_frame": "world",
        "time_step_s": 0.01,
        "topology_sha256": "c" * 64,
        "material_model": "elastic",
        "observation_end_frame_exclusive": 1,
    }

    same = tmp_path / "same"
    with pytest.raises(ValueError, match="paths must differ"):
        producer.produce_jax_fem_backend(
            raw_rollout_path=same,
            runtime_manifest_path=same,
            **common,
        )
    assert called is False

    runtime = tmp_path / "runtime.json"
    runtime.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        producer.produce_jax_fem_backend(
            raw_rollout_path=tmp_path / "raw.npz",
            runtime_manifest_path=runtime,
            **common,
        )
    assert called is False
    assert not (tmp_path / "raw.npz").exists()

    raw = tmp_path / "raw-existing.npz"
    raw.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        producer.produce_jax_fem_backend(
            raw_rollout_path=raw,
            runtime_manifest_path=tmp_path / "other-runtime.json",
            **common,
        )
    assert called is False
