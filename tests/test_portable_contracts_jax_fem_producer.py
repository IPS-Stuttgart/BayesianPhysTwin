from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.jax_fem_producer_v1 as producer
from bayesian_phystwin.lagrangian_backend_v1 import (
    PHYSICAL_ARCHIVE_FILENAME,
    RAW_ARCHIVE_FILENAME,
    RUNTIME_FILENAME,
    file_sha256,
    validate_lagrangian_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


class _ArrayFacade:
    def __init__(self, value: npt.NDArray[Any], events: list[object]) -> None:
        self.value = value
        self.events = events

    def block_until_ready(self) -> _ArrayFacade:
        self.events.append("block")
        return self

    def detach(self) -> _ArrayFacade:
        self.events.append("detach")
        return self

    def cpu(self) -> _ArrayFacade:
        self.events.append("cpu")
        return self

    def numpy(self) -> npt.NDArray[Any]:
        self.events.append("numpy")
        return self.value


class _Replay:
    def __init__(
        self,
        reference: npt.NDArray[Any],
        *,
        solution_mode: str = "direct",
        displacement_field_index: int = 0,
        mutation: str | None = None,
    ) -> None:
        self.reference = np.ascontiguousarray(reference).copy()
        self.solution_mode = solution_mode
        self.displacement_field_index = displacement_field_index
        self.mutation = mutation
        self.pending_z_m = 0.0
        self.current_z_m = 0.0
        self.solve_count = 0
        self.events: list[object] = []

    def _facade(self, value: npt.NDArray[Any]) -> _ArrayFacade:
        return _ArrayFacade(np.ascontiguousarray(value).copy(), self.events)

    def get_reference_points_m(self) -> object:
        return self._facade(self.reference)

    def solve(self) -> object:
        self.events.append("solve")
        self.solve_count += 1
        self.current_z_m += self.pending_z_m
        self.pending_z_m = 0.0
        displacement = np.zeros_like(self.reference)
        displacement[:, 2] = self.current_z_m

        mutation = self.mutation
        self.mutation = None
        if mutation == "shape":
            displacement = displacement[:-1]
        elif mutation == "dtype":
            displacement = displacement.astype(np.float64)
        elif mutation == "nonfinite":
            displacement[0, 0] = np.nan
        elif mutation == "reference-drift":
            self.reference = self.reference + np.asarray(
                0.001,
                dtype=self.reference.dtype,
            )

        selected = self._facade(displacement)
        if self.solution_mode == "direct":
            return selected
        if self.solution_mode == "empty":
            return []
        auxiliary = self._facade(
            np.zeros((len(self.reference), 1), dtype=self.reference.dtype)
        )
        fields = [selected, auxiliary]
        if self.displacement_field_index == 1:
            fields.reverse()
        return tuple(fields) if self.solution_mode == "tuple" else fields


def _reference(*, shift_m: float = 0.0) -> npt.NDArray[np.float32]:
    return np.array(
        [
            [0.00 + shift_m, 0.0, 0.0],
            [0.05 + shift_m, 0.0, 0.0],
            [0.10 + shift_m, 0.0, 0.0],
            [0.15 + shift_m, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def _driven_control(
    transition_index: int,
    replay: producer.JaxFemReplayV1,
) -> None:
    concrete = cast(_Replay, replay)
    concrete.events.append(("driven", transition_index))
    concrete.pending_z_m = 0.001 * (transition_index + 1)


def _zero_action_control(
    transition_index: int,
    replay: producer.JaxFemReplayV1,
) -> None:
    concrete = cast(_Replay, replay)
    concrete.events.append(("zero", transition_index))
    concrete.pending_z_m = 0.0


def _kwargs(
    tmp_path: Path,
    *,
    output_name: str = "output",
    replay_factory: Any | None = None,
    source_kind: str = "synthetic",
) -> dict[str, Any]:
    instances: list[_Replay] = []

    def default_factory() -> _Replay:
        replay = _Replay(_reference())
        instances.append(replay)
        return replay

    return {
        "output_dir": tmp_path / output_name,
        "replay_factory": default_factory if replay_factory is None else replay_factory,
        "driven_control": _driven_control,
        "zero_action_control": _zero_action_control,
        "frame_count": 5,
        "material_query_indices": np.array([3, 0, 2], dtype=np.int32),
        "action_support": [1.0, 0.0, 0.5],
        "engine_revision": "a" * 40,
        "engine_version": "jax-fem-test-runtime",
        "source_artifacts": {
            "producer/scene.py": "b" * 64,
            "producer/mesh.vtk": "c" * 64,
        },
        "device": "cpu",
        "load_step_size": 0.25,
        "element_type": "HEX8",
        "constitutive_model": "neo-hookean",
        "nonlinear_solver": "newton",
        "source_kind": source_kind,
        "_instances": instances,
    }


def _produce(kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[_Replay]]:
    instances = kwargs.pop("_instances")
    artifact = producer.produce_jax_fem_backend(**kwargs)
    return artifact, instances


def test_callback_replay_delegates_and_rejects_noncallables() -> None:
    calls: list[str] = []
    points = _reference()
    replay = producer.CallbackJaxFemReplayV1(
        reference_points_callback=lambda: points,
        solve_callback=lambda: calls.append("solve"),
        context="problem",
    )
    assert replay.context == "problem"
    assert replay.get_reference_points_m() is points
    replay.solve()
    assert calls == ["solve"]

    with pytest.raises(TypeError, match="reference_points_callback must be callable"):
        producer.CallbackJaxFemReplayV1(
            reference_points_callback=cast(Any, None),
            solve_callback=lambda: None,
        )
    with pytest.raises(TypeError, match="solve_callback must be callable"):
        producer.CallbackJaxFemReplayV1(
            reference_points_callback=lambda: points,
            solve_callback=cast(Any, None),
        )


def test_host_array_capture_and_none_returning_synchronization() -> None:
    raw = _reference()
    captured = producer._to_numpy(raw)
    np.testing.assert_array_equal(captured, raw)
    assert captured is not raw

    class _ReadyNone:
        def block_until_ready(self) -> None:
            return None

        def __array__(
            self,
            dtype: np.dtype[Any] | None = None,
            copy: bool | None = None,
        ) -> npt.NDArray[Any]:
            value = np.asarray(raw, dtype=dtype)
            return value.copy() if copy else value

    np.testing.assert_array_equal(producer._to_numpy(_ReadyNone()), raw)


def test_fresh_jax_fem_solves_publish_existing_lagrangian_contract(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    output = Path(kwargs["output_dir"])
    artifact, instances = _produce(kwargs)

    assert artifact == validate_lagrangian_backend(output)
    assert artifact["backend_profile"] == "jax-fem-quasistatic-v1"
    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert [item.solve_count for item in instances] == [4, 4]
    ordered = [
        event
        for event in instances[0].events
        if event == "solve" or isinstance(event, tuple)
    ]
    assert ordered == [
        ("driven", 0),
        "solve",
        ("driven", 1),
        "solve",
        ("driven", 2),
        "solve",
        ("driven", 3),
        "solve",
    ]
    assert {"block", "detach", "cpu", "numpy"} <= set(instances[0].events)

    physical = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    expected_frame_zero = _reference()[[3, 0, 2]]
    np.testing.assert_array_equal(physical["frame_zero_points_m"], expected_frame_zero)
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(expected_frame_zero[None], 5, axis=0),
    )
    np.testing.assert_allclose(
        physical["prediction_m"][:, 0, 2],
        np.array([0.0, 0.001, 0.003, 0.006, 0.010], dtype=np.float32),
    )

    runtime_path = output / "provenance" / RUNTIME_FILENAME
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["engine_repository"] == "deepmodeling/jax-fem"
    assert runtime["step_axis"] == "load-step"
    assert runtime["step_units"] == "1"
    assert runtime["step_size"] == 0.25
    assert runtime["backend_metadata"] == {
        "constitutive_model": "neo-hookean",
        "differentiation_mode": "jax-autodiff",
        "element_type": "HEX8",
        "nonlinear_solver": "newton",
        "precision": "float32",
    }
    assert runtime["information_boundary"] == {
        "dataset_payload_read": False,
        "future_observations_read": False,
        "known_action_used": True,
        "outcomes_read": False,
        "source_kind": "synthetic",
    }
    assert runtime["source_artifacts"]["producer/scene.py"] == "b" * 64
    producer_source = "bayesian_phystwin/jax_fem_producer_v1.py"
    assert runtime["source_artifacts"][producer_source] == file_sha256(
        Path(producer.__file__)
    )


def test_source_only_boundary_is_bound(tmp_path: Path) -> None:
    kwargs = _kwargs(tmp_path, source_kind="source-only")
    output = Path(kwargs["output_dir"])
    _, instances = _produce(kwargs)

    runtime = json.loads(
        (output / "provenance" / RUNTIME_FILENAME).read_text(encoding="utf-8")
    )
    assert runtime["information_boundary"]["source_kind"] == "source-only"
    assert runtime["information_boundary"]["dataset_payload_read"] is True
    assert len(instances) == 2


def test_producer_is_byte_deterministic(tmp_path: Path) -> None:
    first, _ = _produce(_kwargs(tmp_path, output_name="first"))
    second, _ = _produce(_kwargs(tmp_path, output_name="second"))
    assert first["artifact_id"] == second["artifact_id"]

    for relative in (
        "lagrangian-backend.json",
        PHYSICAL_ARCHIVE_FILENAME,
        "SHA256SUMS",
        f"provenance/{RAW_ARCHIVE_FILENAME}",
        f"provenance/{RUNTIME_FILENAME}",
    ):
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()


def test_rejects_reused_or_different_fresh_replays(tmp_path: Path) -> None:
    shared = _Replay(_reference())
    reused = _kwargs(
        tmp_path,
        output_name="reused",
        replay_factory=lambda: shared,
    )
    reused.pop("_instances")
    with pytest.raises(ValueError, match="fresh JAX-FEM replay objects"):
        producer.produce_jax_fem_backend(**reused)

    calls = 0

    def shifted_factory() -> _Replay:
        nonlocal calls
        replay = _Replay(_reference(shift_m=0.001 if calls else 0.0))
        calls += 1
        return replay

    shifted = _kwargs(
        tmp_path,
        output_name="shifted",
        replay_factory=shifted_factory,
    )
    shifted.pop("_instances")
    with pytest.raises(ValueError, match="differ in reference mesh points"):
        producer.produce_jax_fem_backend(**shifted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("shape", "displacement shape"),
        ("dtype", "displacement dtype"),
        ("nonfinite", "displacement contains non-finite"),
        ("reference-drift", "changed reference mesh points"),
    ),
)
def test_rejects_invalid_solve_outputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    def factory() -> _Replay:
        return _Replay(_reference(), mutation=mutation)

    kwargs = _kwargs(
        tmp_path,
        output_name=mutation,
        replay_factory=factory,
    )
    kwargs.pop("_instances")
    with pytest.raises(ValueError, match=message):
        producer.produce_jax_fem_backend(**kwargs)


def test_rejects_unselected_multifield_solution(tmp_path: Path) -> None:
    def factory() -> _Replay:
        return _Replay(_reference(), solution_mode="list")

    kwargs = _kwargs(
        tmp_path,
        output_name="multifield",
        replay_factory=factory,
    )
    kwargs.pop("_instances")
    with pytest.raises(ValueError, match="select one displacement field"):
        producer.produce_jax_fem_backend(**kwargs)


def test_static_contract_failures_precede_engine_execution(tmp_path: Path) -> None:
    calls = 0

    def factory() -> _Replay:
        nonlocal calls
        calls += 1
        return _Replay(_reference())

    invalid_source = _kwargs(tmp_path, replay_factory=factory)
    invalid_source.pop("_instances")
    invalid_source["source_kind"] = "target"
    with pytest.raises(ValueError, match="synthetic or source-only"):
        producer.produce_jax_fem_backend(**invalid_source)
    assert calls == 0

    reserved = _kwargs(tmp_path, output_name="reserved", replay_factory=factory)
    reserved.pop("_instances")
    reserved["source_artifacts"] = {
        "bayesian_phystwin/jax_fem_producer_v1.py": "a" * 64
    }
    with pytest.raises(ValueError, match="reserved producer source path"):
        producer.produce_jax_fem_backend(**reserved)
    assert calls == 0

    invalid_queries = _kwargs(tmp_path, output_name="queries", replay_factory=factory)
    invalid_queries.pop("_instances")
    invalid_queries["material_query_indices"] = [1, 1]
    invalid_queries["action_support"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="must be unique"):
        producer.produce_jax_fem_backend(**invalid_queries)
    assert calls == 0

    invalid_support = _kwargs(tmp_path, output_name="support", replay_factory=factory)
    invalid_support.pop("_instances")
    invalid_support["action_support"] = [1.1, 0.0, 0.5]
    with pytest.raises(ValueError, match=r"finite vector in \[0,1\]"):
        producer.produce_jax_fem_backend(**invalid_support)
    assert calls == 0

    invalid_frames = _kwargs(
        tmp_path,
        output_name="frames",
        replay_factory=factory,
    )
    invalid_frames.pop("_instances")
    invalid_frames["frame_count"] = 1
    with pytest.raises(ValueError, match="frame_count must be an integer >= 2"):
        producer.produce_jax_fem_backend(**invalid_frames)
    assert calls == 0

    invalid_step = _kwargs(
        tmp_path,
        output_name="step",
        replay_factory=factory,
    )
    invalid_step.pop("_instances")
    invalid_step["load_step_size"] = 0.0
    with pytest.raises(ValueError, match="load_step_size"):
        producer.produce_jax_fem_backend(**invalid_step)
    assert calls == 0

    invalid_factory = _kwargs(tmp_path, output_name="factory")
    invalid_factory.pop("_instances")
    invalid_factory["replay_factory"] = None
    with pytest.raises(ValueError, match="replay_factory must be callable"):
        producer.produce_jax_fem_backend(**invalid_factory)

    invalid_control = _kwargs(
        tmp_path,
        output_name="control",
        replay_factory=factory,
    )
    invalid_control.pop("_instances")
    invalid_control["driven_control"] = None
    with pytest.raises(ValueError, match="controls must be callable"):
        producer.produce_jax_fem_backend(**invalid_control)
    assert calls == 0

    existing = tmp_path / "existing"
    existing.mkdir()
    exists = _kwargs(tmp_path, output_name="existing", replay_factory=factory)
    exists.pop("_instances")
    with pytest.raises(FileExistsError):
        producer.produce_jax_fem_backend(**exists)
    assert calls == 0
