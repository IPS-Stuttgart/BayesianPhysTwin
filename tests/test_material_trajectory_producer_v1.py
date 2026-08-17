from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.material_trajectory_backend_v1 import (
    PHYSICAL_ARCHIVE_FILENAME,
    RUNTIME_FILENAME,
    validate_material_trajectory_backend,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    CallbackMaterialTrajectoryReplayV1,
    MATERIAL_TRAJECTORY_PRODUCER_PROTOCOL,
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive

PROMISING_BACKENDS = (
    "warp-fem-v1",
    "sofa-fem-v1",
    "position-based-dynamics-v1",
)


class _ArrayFacade:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.calls: list[str] = []

    def block_until_ready(self) -> _ArrayFacade:
        self.calls.append("block")
        return self

    def detach(self) -> _ArrayFacade:
        self.calls.append("detach")
        return self

    def cpu(self) -> _ArrayFacade:
        self.calls.append("cpu")
        return self

    def numpy(self) -> np.ndarray:
        self.calls.append("numpy")
        return self.value


class _Replay:
    def __init__(
        self,
        frame_zero: np.ndarray,
        *,
        mode: str = "normal",
    ) -> None:
        self.positions = np.ascontiguousarray(frame_zero).copy()
        self.pending = np.zeros_like(self.positions)
        self.mode = mode
        self.step_count = 0
        self.synchronize_count = 0
        self.last_facade: _ArrayFacade | None = None

    def synchronize(self) -> None:
        self.synchronize_count += 1

    def get_material_positions_m(self) -> object:
        value = self.positions
        if self.mode == "shape" and self.step_count:
            value = value[:-1]
        elif self.mode == "dtype" and self.step_count:
            value = value.astype(np.float64)
        elif self.mode == "nonfinite":
            value = value.copy()
            value[0, 0] = np.nan
        if self.mode == "facade":
            self.last_facade = _ArrayFacade(value)
            return self.last_facade
        return value

    def step(self) -> None:
        self.positions += self.pending
        self.pending.fill(0.0)
        self.step_count += 1


def _frame_zero(*, shift_m: float = 0.0) -> np.ndarray:
    return np.stack(
        (
            np.linspace(0.0, 0.12, 5, dtype=np.float32) + shift_m,
            np.zeros(5, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
        ),
        axis=1,
    )


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    concrete = cast(_Replay, replay)
    spatial = np.linspace(0.0, 1.0, len(concrete.positions), dtype=np.float32)
    concrete.pending[:, 2] = 0.001 * (transition_index + 1) * spatial


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    cast(_Replay, replay).pending.fill(0.0)


def _kwargs(
    tmp_path: Path,
    *,
    backend_kind: str = "warp-fem-v1",
    replay_factory: Any | None = None,
    output_name: str = "output",
    engine_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instances: list[_Replay] = []

    def default_factory() -> _Replay:
        replay = _Replay(_frame_zero())
        instances.append(replay)
        return replay

    return {
        "output_dir": tmp_path / output_name,
        "backend_kind": backend_kind,
        "replay_factory": (
            default_factory if replay_factory is None else replay_factory
        ),
        "driven_control": _driven_control,
        "zero_action_control": _zero_control,
        "frame_count": 6,
        "material_query_indices": np.array([4, 0, 2], dtype=np.int64),
        "action_support": np.array([1.0, 0.0, 0.5], dtype=np.float32),
        "engine_revision": "a" * 40,
        "engine_version": "test-engine-v1",
        "producer_repository": "example/backend-producer",
        "producer_revision": "b" * 40,
        "producer_version": "producer-v1",
        "producer_artifacts": {"producer.py": "c" * 64},
        "topology_sha256": "d" * 64,
        "device": "cpu",
        "device_name": "contract-test-device",
        "time_step_s": 1.0 / 120.0,
        "scene_id": "bending-beam-v1",
        "model_kind": "deformable-solid",
        "constitutive_model": "neo-hookean",
        "integrator": "implicit-euler",
        "solver": "backend-native-solver",
        "substeps": 2,
        "engine_parameters": {
            "density_kg_m3": 1000.0,
            "young_modulus_pa": 500000.0,
            **(engine_parameters or {}),
        },
        "_instances": instances,
    }


def _produce(kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[_Replay]]:
    instances = kwargs.pop("_instances")
    artifact = produce_material_trajectory_backend(**kwargs)
    return artifact, instances


def test_callback_replay_delegates_and_rejects_noncallables() -> None:
    calls: list[str] = []
    positions = _frame_zero()
    replay = CallbackMaterialTrajectoryReplayV1(
        synchronize_callback=lambda: calls.append("synchronize"),
        positions_callback=lambda: positions,
        step_callback=lambda: calls.append("step"),
        context="engine-context",
    )
    replay.synchronize()
    assert replay.context == "engine-context"
    assert replay.get_material_positions_m() is positions
    replay.step()
    assert calls == ["synchronize", "step"]

    with pytest.raises(TypeError, match="positions_callback must be callable"):
        CallbackMaterialTrajectoryReplayV1(
            synchronize_callback=lambda: None,
            positions_callback=cast(Any, None),
            step_callback=lambda: None,
        )


@pytest.mark.parametrize("backend_kind", PROMISING_BACKENDS)
def test_promising_backends_use_one_fresh_replay_producer(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    kwargs = _kwargs(tmp_path, backend_kind=backend_kind)
    output = Path(kwargs["output_dir"])
    artifact, instances = _produce(kwargs)

    assert artifact == validate_material_trajectory_backend(output)
    assert artifact["backend_kind"] == backend_kind
    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert [item.step_count for item in instances] == [5, 5]
    assert [item.synchronize_count for item in instances] == [6, 6]

    runtime = json.loads(
        (output / "provenance" / RUNTIME_FILENAME).read_text(encoding="utf-8")
    )
    parameters = runtime["simulation"]["engine_parameters"]
    assert parameters["producer_protocol"] == MATERIAL_TRAJECTORY_PRODUCER_PROTOCOL
    assert parameters["synchronization"] == "before-every-state-capture"
    assert parameters["independent_replay_count"] == 2
    assert parameters["action_timing"] == "control-before-step"
    assert parameters["producer_repository"] == "example/backend-producer"
    assert parameters["producer_artifacts"] == {"producer.py": "c" * 64}

    physical = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    expected_frame_zero = _frame_zero()[[4, 0, 2]]
    np.testing.assert_array_equal(physical["frame_zero_points_m"], expected_frame_zero)
    np.testing.assert_array_equal(
        physical["zero_action_readout_m"],
        np.repeat(expected_frame_zero[None], 6, axis=0),
    )
    assert np.max(physical["prediction_m"][-1, :, 2]) > 0.0


def test_producer_is_byte_deterministic_and_copies_array_facades(
    tmp_path: Path,
) -> None:
    all_instances: list[_Replay] = []

    def factory() -> _Replay:
        replay = _Replay(_frame_zero(), mode="facade")
        all_instances.append(replay)
        return replay

    first_kwargs = _kwargs(
        tmp_path,
        replay_factory=factory,
        output_name="first",
    )
    first_kwargs.pop("_instances")
    first = produce_material_trajectory_backend(**first_kwargs)
    second_kwargs = _kwargs(
        tmp_path,
        replay_factory=factory,
        output_name="second",
    )
    second_kwargs.pop("_instances")
    second = produce_material_trajectory_backend(**second_kwargs)

    assert first["artifact_id"] == second["artifact_id"]
    assert len(all_instances) == 4
    for replay in all_instances:
        assert replay.last_facade is not None
        assert replay.last_facade.calls == ["block", "detach", "cpu", "numpy"]
    for relative in (
        "material-trajectory-backend.json",
        PHYSICAL_ARCHIVE_FILENAME,
        "SHA256SUMS",
        "provenance/material-trajectory-rollout.npz",
        f"provenance/{RUNTIME_FILENAME}",
    ):
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()


def test_producer_rejects_reused_or_different_frame_zero_replays(
    tmp_path: Path,
) -> None:
    shared = _Replay(_frame_zero())
    reused = _kwargs(tmp_path, replay_factory=lambda: shared, output_name="reused")
    reused.pop("_instances")
    with pytest.raises(ValueError, match="fresh material replay objects"):
        produce_material_trajectory_backend(**reused)

    calls = 0

    def shifted_factory() -> _Replay:
        nonlocal calls
        replay = _Replay(_frame_zero(shift_m=0.001 if calls else 0.0))
        calls += 1
        return replay

    shifted = _kwargs(
        tmp_path,
        replay_factory=shifted_factory,
        output_name="shifted",
    )
    shifted.pop("_instances")
    with pytest.raises(ValueError, match="differ at frame zero"):
        produce_material_trajectory_backend(**shifted)


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("shape", "changed state shape"),
        ("dtype", "changed position dtype"),
        ("nonfinite", "contain non-finite"),
    ),
)
def test_producer_rejects_unstable_material_state(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    def factory() -> _Replay:
        return _Replay(_frame_zero(), mode=mode)

    kwargs = _kwargs(
        tmp_path,
        replay_factory=factory,
        output_name=mode,
    )
    kwargs.pop("_instances")
    with pytest.raises(ValueError, match=message):
        produce_material_trajectory_backend(**kwargs)


def test_producer_fails_before_execution_on_invalid_provenance(
    tmp_path: Path,
) -> None:
    calls = 0

    def factory() -> _Replay:
        nonlocal calls
        calls += 1
        return _Replay(_frame_zero())

    reserved = _kwargs(
        tmp_path,
        replay_factory=factory,
        engine_parameters={"topology_sha256": "caller-value"},
    )
    reserved.pop("_instances")
    with pytest.raises(ValueError, match="reserved producer-attestation key"):
        produce_material_trajectory_backend(**reserved)
    assert calls == 0

    invalid = _kwargs(
        tmp_path,
        replay_factory=factory,
        output_name="invalid",
    )
    invalid.pop("_instances")
    invalid["topology_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        produce_material_trajectory_backend(**invalid)
    assert calls == 0


def test_producer_rejects_invalid_surface_queries_and_existing_output(
    tmp_path: Path,
) -> None:
    missing = _kwargs(
        tmp_path,
        replay_factory=lambda: object(),
        output_name="missing",
    )
    missing.pop("_instances")
    with pytest.raises(ValueError, match=r"must expose synchronize\(\)"):
        produce_material_trajectory_backend(**missing)

    duplicate = _kwargs(tmp_path, output_name="duplicate")
    duplicate.pop("_instances")
    duplicate["material_query_indices"] = [1, 1]
    duplicate["action_support"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="must be unique"):
        produce_material_trajectory_backend(**duplicate)

    invalid_support = _kwargs(tmp_path, output_name="support")
    invalid_support.pop("_instances")
    invalid_support["action_support"] = [1.1, 0.0, 0.5]
    with pytest.raises(ValueError, match=r"finite vector in \[0,1\]"):
        produce_material_trajectory_backend(**invalid_support)

    existing = tmp_path / "existing"
    existing.mkdir()
    exists = _kwargs(tmp_path, output_name="existing")
    exists.pop("_instances")
    with pytest.raises(FileExistsError):
        produce_material_trajectory_backend(**exists)
