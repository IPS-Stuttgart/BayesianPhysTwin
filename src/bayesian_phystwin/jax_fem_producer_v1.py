"""Produce strict external-physics rollouts from JAX-FEM solve sequences.

This module deliberately has no JAX or JAX-FEM import. A caller supplies a
factory for fresh replay wrappers. Each wrapper exposes fixed reference mesh
points and one solve step, allowing JAX-FEM to remain an optional producer-side
dependency rather than a Bayesian-PhysTwin runtime dependency.
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

JAX_FEM_PROFILE_ID: Final = "jax-fem-v1"
JAX_FEM_PRODUCER_RESULT_SCHEMA: Final = "bayesian-phystwin.jax-fem-producer"
JAX_FEM_PRODUCER_RESULT_VERSION: Final = 1
JAX_FEM_SOLUTION_SEMANTICS: Final = "nodal-displacement-from-fixed-reference"

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]
ReplayFactory: TypeAlias = Callable[[], "JaxFemReplayV1"]
ReplayControl: TypeAlias = Callable[[int, "JaxFemReplayV1"], None]


class JaxFemReplayV1(Protocol):
    """Minimum producer-side wrapper surface used by the adapter."""

    def get_reference_points_m(self) -> object:
        """Return fixed FEM mesh-node coordinates with shape ``(N,3)``."""

        ...

    def solve(self) -> object:
        """Solve one load or time step and return displacement field(s)."""

        ...


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _positive_frame_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("frame_count must be an integer at least two")
    return value


def _solution_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("solution_index must be a nonnegative integer")
    return value


def _to_numpy(value: object) -> npt.NDArray[Any]:
    """Synchronize a JAX-like value and copy it into host NumPy memory."""

    current = value
    block_until_ready = getattr(current, "block_until_ready", None)
    if callable(block_until_ready):
        synchronized = block_until_ready()
        if synchronized is not None:
            current = synchronized
    return np.ascontiguousarray(np.asarray(current)).copy()


def _capture_reference(replay: JaxFemReplayV1) -> FloatArray:
    points = _to_numpy(replay.get_reference_points_m())
    _require(
        points.ndim == 2 and points.shape[0] >= 1 and points.shape[1] == 3,
        "JAX-FEM reference points must have shape (N,3)",
    )
    _require(
        np.issubdtype(points.dtype, np.floating),
        "JAX-FEM reference points must be floating point",
    )
    _require(
        np.all(np.isfinite(points)),
        "JAX-FEM reference points contain non-finite values",
    )
    return cast(FloatArray, points)


def _build_replay(replay_factory: ReplayFactory) -> JaxFemReplayV1:
    replay = replay_factory()
    if not callable(getattr(replay, "get_reference_points_m", None)):
        raise ValueError("JAX-FEM replay must expose get_reference_points_m()")
    if not callable(getattr(replay, "solve", None)):
        raise ValueError("JAX-FEM replay must expose solve()")
    return replay


def _select_solution(value: object, *, solution_index: int) -> object:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("JAX-FEM solve returned no solution fields")
        if solution_index >= len(value):
            raise ValueError("solution_index exceeds JAX-FEM solution field count")
        return value[solution_index]
    if solution_index != 0:
        raise ValueError("solution_index must be zero for a direct JAX-FEM solution")
    return value


def _capture_displacement(
    replay: JaxFemReplayV1,
    *,
    solution_index: int,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype[Any],
) -> FloatArray:
    selected = _select_solution(replay.solve(), solution_index=solution_index)
    displacement = _to_numpy(selected)
    _require(
        displacement.shape == expected_shape,
        "JAX-FEM displacement shape differs from reference mesh nodes",
    )
    _require(
        np.issubdtype(displacement.dtype, np.floating),
        "JAX-FEM displacement must be floating point",
    )
    _require(
        displacement.dtype == expected_dtype,
        "JAX-FEM displacement dtype differs from reference points",
    )
    _require(
        np.all(np.isfinite(displacement)),
        "JAX-FEM displacement contains non-finite values",
    )
    return cast(FloatArray, displacement)


def _record_replay(
    replay: JaxFemReplayV1,
    control: ReplayControl,
    *,
    frame_count: int,
    solution_index: int,
    label: str,
) -> tuple[FloatArray, FloatArray]:
    reference = _capture_reference(replay)
    frames = [reference.copy()]
    expected_shape = cast(tuple[int, int], reference.shape)
    expected_dtype = reference.dtype
    for transition_index in range(frame_count - 1):
        control(transition_index, replay)
        displacement = _capture_displacement(
            replay,
            solution_index=solution_index,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
        )
        current_reference = _capture_reference(replay)
        _require(
            current_reference.dtype == expected_dtype
            and np.array_equal(current_reference, reference),
            f"{label} JAX-FEM replay changed reference mesh points",
        )
        positions = np.ascontiguousarray(reference + displacement)
        _require(
            np.all(np.isfinite(positions)),
            f"{label} JAX-FEM absolute positions contain non-finite values",
        )
        frames.append(cast(FloatArray, positions))
    trajectory = np.ascontiguousarray(np.stack(frames, axis=0))
    return reference, cast(FloatArray, trajectory)


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
        "query entity index exceeds JAX-FEM node count",
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
        np.all(np.isfinite(support)) and np.all((support >= 0.0) & (support <= 1.0)),
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
        load_external_entity_rollout(temporary)
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    published, validated = load_external_entity_rollout(destination)
    return published, validated


def produce_jax_fem_entity_rollout(
    *,
    output_path: str | Path,
    replay_factory: ReplayFactory,
    driven_control: ReplayControl,
    zero_action_control: ReplayControl,
    frame_count: int,
    query_entity_indices: Sequence[int] | IntegerArray,
    action_support: Sequence[float] | FloatArray,
    solution_index: int = 0,
) -> dict[str, Any]:
    """Run fresh driven and zero-action JAX-FEM solve sequences.

    Frame zero is the fixed reference mesh. For each subsequent frame, the
    corresponding control callback runs immediately before ``solve()``. The
    selected solution field must be a 3-D nodal displacement array relative to
    the fixed reference points.
    """

    frames = _positive_frame_count(frame_count)
    selected_index = _solution_index(solution_index)
    if not callable(replay_factory):
        raise ValueError("replay_factory must be callable")
    if not callable(driven_control) or not callable(zero_action_control):
        raise ValueError("driven and zero-action controls must be callable")
    destination = _new_output_path(output_path, name="raw rollout")

    driven_replay = _build_replay(replay_factory)
    driven_reference, driven = _record_replay(
        driven_replay,
        driven_control,
        frame_count=frames,
        solution_index=selected_index,
        label="driven",
    )
    zero_replay = _build_replay(replay_factory)
    _require(
        zero_replay is not driven_replay,
        "replay_factory must return fresh JAX-FEM replay objects",
    )
    zero_reference, zero = _record_replay(
        zero_replay,
        zero_action_control,
        frame_count=frames,
        solution_index=selected_index,
        label="zero-action",
    )
    _require(
        zero.shape == driven.shape and zero.dtype == driven.dtype,
        "driven and zero-action JAX-FEM replay shapes or dtypes differ",
    )
    _require(
        np.array_equal(driven_reference, zero_reference),
        "fresh JAX-FEM replays differ in reference mesh points",
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
        "schema": JAX_FEM_PRODUCER_RESULT_SCHEMA,
        "schema_version": JAX_FEM_PRODUCER_RESULT_VERSION,
        "profile_id": JAX_FEM_PROFILE_ID,
        "frame_count": int(validated_driven.shape[0]),
        "entity_count": int(validated_driven.shape[1]),
        "query_count": int(len(validated_indices)),
        "position_dtype": validated_driven.dtype.str,
        "solution_index": selected_index,
        "solution_semantics": JAX_FEM_SOLUTION_SEMANTICS,
        "entity_identity_sha256": array_sha256(validated_driven[0]),
        "query_indices_sha256": array_sha256(validated_indices),
        "raw_rollout_sha256": file_sha256(published),
        "independent_replay_count": 2,
        "action_timing": "control-before-solve",
    }
    return cast(
        dict[str, Any],
        plain_json({**identity, "producer_result_id": content_id(identity)}),
    )


def produce_jax_fem_backend(
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
    solution_index: int = 0,
) -> dict[str, Any]:
    """Produce a JAX-FEM raw archive and its strict runtime manifest."""

    raw_requested = Path(raw_rollout_path)
    runtime_requested = Path(runtime_manifest_path)
    if raw_requested.absolute() == runtime_requested.absolute():
        raise ValueError("raw rollout and runtime manifest paths must differ")
    raw_destination = _new_output_path(raw_requested, name="raw rollout")
    runtime_destination = _new_output_path(
        runtime_requested,
        name="runtime manifest",
    )

    rollout = produce_jax_fem_entity_rollout(
        output_path=raw_destination,
        replay_factory=replay_factory,
        driven_control=driven_control,
        zero_action_control=zero_action_control,
        frame_count=frame_count,
        query_entity_indices=query_entity_indices,
        action_support=action_support,
        solution_index=solution_index,
    )
    runtime = write_external_physics_runtime_manifest(
        output_path=runtime_destination,
        raw_rollout_path=raw_destination,
        profile=get_backend_profile(JAX_FEM_PROFILE_ID),
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
    "JAX_FEM_PRODUCER_RESULT_SCHEMA",
    "JAX_FEM_PRODUCER_RESULT_VERSION",
    "JAX_FEM_PROFILE_ID",
    "JAX_FEM_SOLUTION_SEMANTICS",
    "JaxFemReplayV1",
    "ReplayControl",
    "ReplayFactory",
    "produce_jax_fem_backend",
    "produce_jax_fem_entity_rollout",
]
