"""Produce strict external-physics rollouts from Genesis MPM replays.

This module deliberately has no Genesis import.  A caller supplies a factory for
fresh, already-built Genesis scenes and the corresponding MPM entity.  The
adapter relies only on the public ``MPMEntity.get_particles_pos`` and
``Scene.step`` methods, so importing Bayesian-PhysTwin never imports the heavy
engine runtime.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._portable_contracts import content_id
from .external_physics_backend_v1 import (
    array_sha256,
    file_sha256,
    load_external_entity_rollout,
    write_external_physics_runtime_manifest,
)
from .physical_rollout_v1 import write_deterministic_npz
from .physics_backend_registry_v1 import get_backend_profile

GENESIS_MPM_PROFILE_ID: Final = "genesis-mpm-v1"
GENESIS_MPM_PRODUCER_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.genesis-mpm-producer"
)
GENESIS_MPM_PRODUCER_RESULT_VERSION: Final = 1

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]
ReplayFactory: TypeAlias = Callable[[], tuple["GenesisSceneV1", "GenesisMPMEntityV1"]]
ReplayControl: TypeAlias = Callable[
    [int, "GenesisSceneV1", "GenesisMPMEntityV1"],
    None,
]


class GenesisSceneV1(Protocol):
    """Minimum built-scene surface used by the producer."""

    def step(self) -> object:
        """Advance one Genesis scene step."""

        ...


class GenesisMPMEntityV1(Protocol):
    """Minimum MPM entity surface used by the producer."""

    def get_particles_pos(self, envs_idx: object | None = None) -> object:
        """Return current particle positions from Genesis."""

        ...


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _positive_frame_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("frame_count must be an integer at least two")
    return value


def _environment_index(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("env_index must be a nonnegative integer or None")
    return value


def _to_numpy(value: object) -> npt.NDArray[Any]:
    """Detach a Genesis/Torch-like tensor and copy it into host NumPy memory."""

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    to_numpy = getattr(current, "numpy", None)
    if callable(to_numpy):
        current = to_numpy()
    return np.ascontiguousarray(np.asarray(current)).copy()


def _capture_positions(
    entity: GenesisMPMEntityV1,
    *,
    env_index: int | None,
) -> FloatArray:
    if env_index is None:
        raw = entity.get_particles_pos()
    else:
        raw = entity.get_particles_pos(envs_idx=[env_index])
    positions = _to_numpy(raw)
    if positions.ndim == 3:
        _require(
            positions.shape[0] == 1,
            "Genesis batched particle positions require an exact env_index",
        )
        positions = np.ascontiguousarray(positions[0]).copy()
    _require(
        positions.ndim == 2
        and positions.shape[0] >= 1
        and positions.shape[1] == 3,
        "Genesis particle positions must have shape (P,3)",
    )
    _require(
        np.issubdtype(positions.dtype, np.floating),
        "Genesis particle positions must be floating point",
    )
    _require(
        np.all(np.isfinite(positions)),
        "Genesis particle positions contain non-finite values",
    )
    return cast(FloatArray, positions)


def _build_replay(
    replay_factory: ReplayFactory,
) -> tuple[GenesisSceneV1, GenesisMPMEntityV1]:
    replay = replay_factory()
    if not isinstance(replay, tuple) or len(replay) != 2:
        raise ValueError("replay_factory must return (scene, mpm_entity)")
    scene, entity = replay
    if not callable(getattr(scene, "step", None)):
        raise ValueError("Genesis replay scene must expose step()")
    if not callable(getattr(entity, "get_particles_pos", None)):
        raise ValueError(
            "Genesis replay entity must expose get_particles_pos()"
        )
    return scene, entity


def _record_replay(
    scene: GenesisSceneV1,
    entity: GenesisMPMEntityV1,
    control: ReplayControl,
    *,
    frame_count: int,
    env_index: int | None,
    label: str,
) -> FloatArray:
    frames = [_capture_positions(entity, env_index=env_index)]
    expected_shape = frames[0].shape
    expected_dtype = frames[0].dtype
    for transition_index in range(frame_count - 1):
        control(transition_index, scene, entity)
        scene.step()
        positions = _capture_positions(entity, env_index=env_index)
        _require(
            positions.shape == expected_shape,
            f"{label} Genesis replay changed particle count or shape",
        )
        _require(
            positions.dtype == expected_dtype,
            f"{label} Genesis replay changed particle dtype",
        )
        frames.append(positions)
    return cast(FloatArray, np.ascontiguousarray(np.stack(frames, axis=0)))


def _query_indices(
    values: Sequence[int] | IntegerArray,
    *,
    entity_count: int,
) -> npt.NDArray[np.int64]:
    indices = np.ascontiguousarray(np.asarray(values))
    _require(indices.ndim == 1, "query_entity_indices must be an integer vector")
    _require(len(indices) >= 1, "query_entity_indices must not be empty")
    _require(
        np.issubdtype(indices.dtype, np.integer)
        and not np.issubdtype(indices.dtype, np.bool_),
        "query_entity_indices must be an integer vector",
    )
    result = np.ascontiguousarray(indices, dtype=np.int64)
    _require(
        len(np.unique(result)) == len(result),
        "query_entity_indices must be unique",
    )
    _require(
        np.all((result >= 0) & (result < entity_count)),
        "query entity index exceeds Genesis particle count",
    )
    return result


def _action_support(
    values: Sequence[float] | FloatArray,
    *,
    query_count: int,
    dtype: np.dtype[Any],
) -> FloatArray:
    raw = np.ascontiguousarray(np.asarray(values))
    _require(
        raw.shape == (query_count,)
        and not np.issubdtype(raw.dtype, np.bool_)
        and (
            np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.floating)
        ),
        "action_support must be a numeric vector matching the query count",
    )
    support = np.ascontiguousarray(raw, dtype=dtype)
    _require(
        np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action_support must be a finite vector in [0,1]",
    )
    return cast(FloatArray, support)


def _fsync_directory(directory: Path) -> None:
    flags = (
        os.O_RDONLY
        | int(getattr(os, "O_DIRECTORY", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
    )
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _new_output_path(output_path: str | Path, *, name: str) -> Path:
    destination = Path(output_path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not any(parent.is_symlink() for parent in destination.parents),
        f"{name} path must not traverse a symlink",
    )
    return destination


def _publish_rollout(
    output_path: str | Path,
    arrays: Mapping[str, npt.NDArray[Any]],
) -> tuple[Path, dict[str, npt.NDArray[Any]]]:
    destination = _new_output_path(output_path, name="raw rollout")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        write_deterministic_npz(temporary, arrays)
        _, validated = load_external_entity_rollout(temporary)
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    published, validated = load_external_entity_rollout(destination)
    return published, validated


def produce_genesis_mpm_entity_rollout(
    *,
    output_path: str | Path,
    replay_factory: ReplayFactory,
    driven_control: ReplayControl,
    zero_action_control: ReplayControl,
    frame_count: int,
    query_entity_indices: Sequence[int] | IntegerArray,
    action_support: Sequence[float] | FloatArray,
    env_index: int | None = None,
) -> dict[str, Any]:
    """Run fresh driven and zero-action Genesis scenes and publish one archive.

    Frame zero is captured before either control callback is invoked.  For each
    subsequent frame the corresponding callback runs immediately before
    ``scene.step()``.  ``replay_factory`` is called once per arm and must create a
    fresh, already-built scene with identical initial particle sampling and order.
    """

    frames = _positive_frame_count(frame_count)
    environment = _environment_index(env_index)
    if not callable(replay_factory):
        raise ValueError("replay_factory must be callable")
    if not callable(driven_control) or not callable(zero_action_control):
        raise ValueError("driven and zero-action controls must be callable")
    destination = _new_output_path(output_path, name="raw rollout")

    driven_scene, driven_entity = _build_replay(replay_factory)
    driven = _record_replay(
        driven_scene,
        driven_entity,
        driven_control,
        frame_count=frames,
        env_index=environment,
        label="driven",
    )
    zero_scene, zero_entity = _build_replay(replay_factory)
    _require(
        zero_scene is not driven_scene and zero_entity is not driven_entity,
        "replay_factory must return fresh Genesis scene and entity objects",
    )
    zero = _record_replay(
        zero_scene,
        zero_entity,
        zero_action_control,
        frame_count=frames,
        env_index=environment,
        label="zero-action",
    )
    _require(
        zero.shape == driven.shape and zero.dtype == driven.dtype,
        "driven and zero-action Genesis replay shapes or dtypes differ",
    )
    _require(
        np.array_equal(driven[0], zero[0]),
        "fresh Genesis replays differ at frame zero; seed and sampling must match",
    )

    indices = _query_indices(
        query_entity_indices,
        entity_count=int(driven.shape[1]),
    )
    support = _action_support(
        action_support,
        query_count=len(indices),
        dtype=driven.dtype,
    )
    arrays: dict[str, npt.NDArray[Any]] = {
        "driven_entity_positions_m": driven,
        "zero_action_entity_positions_m": zero,
        "query_entity_indices": indices,
        "action_support": support,
    }
    published, validated = _publish_rollout(destination, arrays)
    validated_driven = validated["driven_entity_positions_m"]
    validated_indices = np.asarray(
        validated["query_entity_indices"],
        dtype=np.int64,
    )
    identity: dict[str, Any] = {
        "schema": GENESIS_MPM_PRODUCER_RESULT_SCHEMA,
        "schema_version": GENESIS_MPM_PRODUCER_RESULT_VERSION,
        "profile_id": GENESIS_MPM_PROFILE_ID,
        "frame_count": int(validated_driven.shape[0]),
        "entity_count": int(validated_driven.shape[1]),
        "query_count": int(len(validated_indices)),
        "position_dtype": validated_driven.dtype.str,
        "env_index": environment,
        "entity_identity_sha256": array_sha256(validated_driven[0]),
        "query_indices_sha256": array_sha256(validated_indices),
        "raw_rollout_sha256": file_sha256(published),
        "independent_replay_count": 2,
        "action_timing": "control-before-scene-step",
    }
    return cast(
        dict[str, Any],
        plain_json({**identity, "producer_result_id": content_id(identity)}),
    )


def produce_genesis_mpm_backend(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    replay_factory: ReplayFactory,
    driven_control: ReplayControl,
    zero_action_control: ReplayControl,
    frame_count: int,
    query_entity_indices: Sequence[int] | IntegerArray,
    action_support: Sequence[float] | FloatArray,
    engine_revision: str,
    engine_version: str,
    producer_repository: str,
    producer_revision: str,
    coordinate_frame: str,
    time_step_s: float,
    topology_sha256: str,
    material_model: str,
    observation_end_frame_exclusive: int,
    parameterization: Mapping[str, Any] | None = None,
    producer_artifacts: Mapping[str, str] | None = None,
    env_index: int | None = None,
) -> dict[str, Any]:
    """Produce a Genesis raw archive and its strict runtime manifest."""

    raw_requested = Path(raw_rollout_path)
    runtime_requested = Path(runtime_manifest_path)
    if raw_requested.absolute() == runtime_requested.absolute():
        raise ValueError("raw rollout and runtime manifest paths must differ")
    raw_destination = _new_output_path(raw_requested, name="raw rollout")
    runtime_destination = _new_output_path(
        runtime_requested,
        name="runtime manifest",
    )

    rollout = produce_genesis_mpm_entity_rollout(
        output_path=raw_destination,
        replay_factory=replay_factory,
        driven_control=driven_control,
        zero_action_control=zero_action_control,
        frame_count=frame_count,
        query_entity_indices=query_entity_indices,
        action_support=action_support,
        env_index=env_index,
    )
    runtime = write_external_physics_runtime_manifest(
        output_path=runtime_destination,
        raw_rollout_path=raw_destination,
        profile=get_backend_profile(GENESIS_MPM_PROFILE_ID),
        engine_revision=engine_revision,
        engine_version=engine_version,
        producer_repository=producer_repository,
        producer_revision=producer_revision,
        coordinate_frame=coordinate_frame,
        time_step_s=time_step_s,
        topology_sha256=topology_sha256,
        material_model=material_model,
        observation_end_frame_exclusive=observation_end_frame_exclusive,
        parameterization=parameterization,
        producer_artifacts=producer_artifacts,
    )
    return {"rollout": rollout, "runtime": runtime}


__all__ = [
    "GENESIS_MPM_PRODUCER_RESULT_SCHEMA",
    "GENESIS_MPM_PRODUCER_RESULT_VERSION",
    "GENESIS_MPM_PROFILE_ID",
    "GenesisMPMEntityV1",
    "GenesisSceneV1",
    "ReplayControl",
    "ReplayFactory",
    "produce_genesis_mpm_backend",
    "produce_genesis_mpm_entity_rollout",
]
