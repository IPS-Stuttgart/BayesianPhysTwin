"""Dependency-free producer for fixed-identity material trajectory backends.

External engines remain optional. Callers provide a small replay wrapper around a
fresh, already-configured simulator scene; this module records matched driven and
zero-action trajectories and publishes them through the existing strict
``material-trajectory-v1`` transport.
"""

from __future__ import annotations

import platform
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    integer_array,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    repository_name,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .material_trajectory_backend_v1 import (
    CANONICAL_COORDINATE_FRAME,
    MATERIAL_TRAJECTORY_RUNTIME_SCHEMA,
    MATERIAL_TRAJECTORY_SCHEMA_VERSION,
    array_sha256,
    file_sha256,
    get_material_backend_profile,
    load_material_trajectory_rollout,
    materialize_material_trajectory_backend,
    validate_material_runtime_manifest,
)
from .physical_rollout_v1 import write_deterministic_npz

MATERIAL_TRAJECTORY_PRODUCER_PROTOCOL: Final = (
    "fresh-replay-control-before-step-v1"
)

_RESERVED_ENGINE_PARAMETER_KEYS: Final = frozenset(
    {
        "producer_protocol",
        "producer_repository",
        "producer_revision",
        "producer_artifacts",
        "topology_sha256",
        "frame_zero_state_sha256",
        "query_indices_sha256",
        "synchronization",
        "independent_replay_count",
        "action_timing",
    }
)
_INFORMATION_BOUNDARY: Final = {
    "future_observations_read": False,
    "future_outcomes_read": False,
    "known_action_used": True,
    "action_support_uses_observation_residuals": False,
    "material_query_indices_fixed_at_frame_zero": True,
}

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]
ReplayFactory: TypeAlias = Callable[[], "MaterialTrajectoryReplayV1"]
ReplayControl: TypeAlias = Callable[[int, "MaterialTrajectoryReplayV1"], None]


class MaterialTrajectoryReplayV1(Protocol):
    """Minimum engine-wrapper surface consumed by the portable producer.

    ``synchronize`` must make all pending engine work visible to the host before
    ``get_material_positions_m`` returns. CPU engines may implement it as a no-op.
    Positions must use one persistent material order and already be expressed in
    metres in ``right-handed-z-up-world-v1``.
    """

    def synchronize(self) -> object:
        """Synchronize pending engine work before a host-side state capture."""

        ...

    def get_material_positions_m(self) -> object:
        """Return current fixed-identity material positions with shape ``(S, 3)``."""

        ...

    def step(self) -> object:
        """Advance the simulator by one registered output step."""

        ...


def _no_op() -> None:
    return None


@dataclass(slots=True)
class CallbackMaterialTrajectoryReplayV1:
    """Build a replay wrapper from three engine-specific zero-argument callbacks."""

    synchronize_callback: Callable[[], object]
    positions_callback: Callable[[], object]
    step_callback: Callable[[], object]
    context: object | None = None

    def __post_init__(self) -> None:
        for name in (
            "synchronize_callback",
            "positions_callback",
            "step_callback",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> object:
        return self.positions_callback()

    def step(self) -> object:
        return self.step_callback()


@dataclass(slots=True)
class DrakeDeformableBodyReplayV1:
    """Adapt one Drake ``DeformableBody`` to the fixed-material replay protocol.

    Drake returns body vertex positions as a ``(3, N)`` matrix. The portable
    producer uses ``(N, 3)`` rows, so this adapter performs and validates the
    conversion without importing ``pydrake`` into BayesianPhysTwin.
    """

    deformable_body: object
    plant_context_callback: Callable[[], object]
    advance_callback: Callable[[], object]
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.deformable_body, "GetPositions", None)):
            raise TypeError("deformable_body must expose GetPositions(context)")
        for name in (
            "plant_context_callback",
            "advance_callback",
            "synchronize_callback",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        get_positions = getattr(self.deformable_body, "GetPositions")
        matrix = _to_numpy(get_positions(self.plant_context_callback()))
        _require(
            matrix.ndim == 2 and matrix.shape[0] == 3 and matrix.shape[1] >= 1,
            "Drake GetPositions must return a matrix with shape (3,N)",
        )
        _require(
            np.issubdtype(matrix.dtype, np.floating),
            "Drake GetPositions must return floating positions",
        )
        _require(
            np.all(np.isfinite(matrix)),
            "Drake GetPositions returned non-finite positions",
        )
        return cast(FloatArray, np.ascontiguousarray(matrix.T))

    def step(self) -> object:
        return self.advance_callback()


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _new_output_directory(path: str | Path) -> Path:
    output = Path(path).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not any(parent.is_symlink() for parent in output.parents),
        "output path must not traverse a symlink",
    )
    return output


def _to_numpy(value: object) -> npt.NDArray[Any]:
    """Synchronize common array facades and copy them into host NumPy memory."""

    current = value
    block_until_ready = getattr(current, "block_until_ready", None)
    if callable(block_until_ready):
        synchronized = block_until_ready()
        if synchronized is not None:
            current = synchronized
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


def _build_replay(replay_factory: ReplayFactory) -> MaterialTrajectoryReplayV1:
    replay = replay_factory()
    for method in ("synchronize", "get_material_positions_m", "step"):
        if not callable(getattr(replay, method, None)):
            raise ValueError(f"material replay must expose {method}()")
    return replay


def _capture_positions(
    replay: MaterialTrajectoryReplayV1,
    *,
    label: str,
) -> FloatArray:
    replay.synchronize()
    positions = _to_numpy(replay.get_material_positions_m())
    _require(
        positions.ndim == 2
        and positions.shape[0] >= 1
        and positions.shape[1] == 3,
        f"{label} material positions must have shape (S,3)",
    )
    _require(
        np.issubdtype(positions.dtype, np.floating),
        f"{label} material positions must be floating point",
    )
    _require(
        np.all(np.isfinite(positions)),
        f"{label} material positions contain non-finite values",
    )
    return cast(FloatArray, positions)


def _record_replay(
    replay: MaterialTrajectoryReplayV1,
    control: ReplayControl,
    *,
    frame_count: int,
    label: str,
) -> FloatArray:
    frames = [_capture_positions(replay, label=label)]
    expected_shape = frames[0].shape
    expected_dtype = frames[0].dtype
    for transition_index in range(frame_count - 1):
        control(transition_index, replay)
        replay.step()
        current = _capture_positions(replay, label=label)
        _require(
            current.shape == expected_shape,
            f"{label} material replay changed state shape",
        )
        _require(
            current.dtype == expected_dtype,
            f"{label} material replay changed position dtype",
        )
        frames.append(current)
    return cast(FloatArray, np.ascontiguousarray(np.stack(frames, axis=0)))


def _query_indices(
    values: Sequence[int] | IntegerArray,
    *,
    state_count: int,
) -> npt.NDArray[np.int64]:
    indices = np.ascontiguousarray(
        integer_array(values, name="material_query_indices"),
        dtype=np.int64,
    )
    _require(indices.ndim == 1, "material_query_indices must be an integer vector")
    _require(len(indices) >= 1, "material_query_indices must not be empty")
    _require(
        len(np.unique(indices)) == len(indices),
        "material_query_indices must be unique",
    )
    _require(
        np.all((indices >= 0) & (indices < state_count)),
        "material query index exceeds state count",
    )
    return indices


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


def _validated_engine_parameter_base(
    values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parameters = plain_json(
        frozen_finite_json_mapping(values, name="engine_parameters")
    )
    _require(
        not (_RESERVED_ENGINE_PARAMETER_KEYS & set(parameters)),
        "engine_parameters contains a reserved producer-attestation key",
    )
    return parameters


def _engine_parameters(
    base_parameters: Mapping[str, Any],
    *,
    producer_repository: str,
    producer_revision: str,
    producer_artifacts: Mapping[str, Any],
    topology_sha256: str,
    frame_zero_state_sha256: str,
    query_indices_sha256: str,
) -> dict[str, Any]:
    parameters = dict(base_parameters)
    parameters.update(
        {
            "producer_protocol": MATERIAL_TRAJECTORY_PRODUCER_PROTOCOL,
            "producer_repository": producer_repository,
            "producer_revision": producer_revision,
            "producer_artifacts": plain_json(producer_artifacts),
            "topology_sha256": topology_sha256,
            "frame_zero_state_sha256": frame_zero_state_sha256,
            "query_indices_sha256": query_indices_sha256,
            "synchronization": "before-every-state-capture",
            "independent_replay_count": 2,
            "action_timing": "control-before-step",
        }
    )
    return parameters


def produce_material_trajectory_backend(
    *,
    output_dir: str | Path,
    backend_kind: str,
    replay_factory: ReplayFactory,
    driven_control: ReplayControl,
    zero_action_control: ReplayControl,
    frame_count: int,
    material_query_indices: Sequence[int] | IntegerArray,
    action_support: Sequence[float] | FloatArray,
    engine_revision: str,
    engine_version: str,
    producer_repository: str,
    producer_revision: str,
    producer_version: str,
    producer_artifacts: Mapping[str, str],
    topology_sha256: str,
    device: str,
    device_name: str,
    time_step_s: float,
    scene_id: str,
    model_kind: str,
    constitutive_model: str,
    integrator: str,
    solver: str,
    substeps: int,
    engine_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run matched fresh replays and publish one strict backend bundle.

    Frame zero is captured before either control callback is invoked. For every
    later frame, the corresponding callback runs immediately before ``step``.
    ``replay_factory`` is called exactly twice and must return fresh wrappers with
    identical frame-zero material identities and ordering.
    """

    output = _new_output_directory(output_dir)
    profile = get_material_backend_profile(backend_kind)
    frames = _positive_integer(frame_count, name="frame_count", minimum=2)
    step_s = _positive_float(time_step_s, name="time_step_s")
    registered_substeps = _positive_integer(substeps, name="substeps")
    if not callable(replay_factory):
        raise ValueError("replay_factory must be callable")
    if not callable(driven_control) or not callable(zero_action_control):
        raise ValueError("driven and zero-action controls must be callable")

    revision = exact_revision(engine_revision, name="engine_revision")
    version = nonempty_string(engine_version, name="engine_version")
    producer_repo = repository_name(
        producer_repository, name="producer_repository"
    )
    producer_rev = exact_revision(producer_revision, name="producer_revision")
    producer_ver = nonempty_string(producer_version, name="producer_version")
    artifacts = source_artifact_mapping(
        producer_artifacts,
        name="producer_artifacts",
    )
    topology = sha256_digest(topology_sha256, name="topology_sha256")
    registered_device = nonempty_string(device, name="device")
    registered_device_name = nonempty_string(device_name, name="device_name")
    registered_scene = nonempty_string(scene_id, name="scene_id")
    registered_model = nonempty_string(model_kind, name="model_kind")
    registered_constitutive = nonempty_string(
        constitutive_model, name="constitutive_model"
    )
    registered_integrator = nonempty_string(integrator, name="integrator")
    registered_solver = nonempty_string(solver, name="solver")
    parameter_base = _validated_engine_parameter_base(engine_parameters)

    driven_replay = _build_replay(replay_factory)
    driven = _record_replay(
        driven_replay,
        driven_control,
        frame_count=frames,
        label="driven",
    )
    zero_replay = _build_replay(replay_factory)
    _require(
        zero_replay is not driven_replay,
        "replay_factory must return fresh material replay objects",
    )
    zero = _record_replay(
        zero_replay,
        zero_action_control,
        frame_count=frames,
        label="zero-action",
    )
    _require(
        zero.shape == driven.shape and zero.dtype == driven.dtype,
        "driven and zero-action material replay shapes or dtypes differ",
    )
    _require(
        np.array_equal(driven[0], zero[0]),
        "fresh material replays differ at frame zero",
    )

    indices = _query_indices(
        material_query_indices,
        state_count=int(driven.shape[1]),
    )
    support = _action_support(
        action_support,
        query_count=len(indices),
        dtype=driven.dtype,
    )
    attestations = _engine_parameters(
        parameter_base,
        producer_repository=producer_repo,
        producer_revision=producer_rev,
        producer_artifacts=artifacts,
        topology_sha256=topology,
        frame_zero_state_sha256=array_sha256(driven[0]),
        query_indices_sha256=array_sha256(indices),
    )

    with tempfile.TemporaryDirectory(
        prefix=".material-trajectory-producer-",
        dir=output.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        raw_path = temporary / "material-trajectory-rollout.npz"
        runtime_path = temporary / "material-runtime.json"
        write_deterministic_npz(
            raw_path,
            {
                "driven_material_positions_m": driven,
                "zero_action_material_positions_m": zero,
                "material_query_indices": indices,
                "action_support": support,
            },
        )
        _, raw = load_material_trajectory_rollout(raw_path)
        validated_driven = raw["driven_material_positions_m"]
        validated_indices = np.asarray(raw["material_query_indices"], dtype=np.int64)
        identity: dict[str, Any] = {
            "schema": MATERIAL_TRAJECTORY_RUNTIME_SCHEMA,
            "schema_version": MATERIAL_TRAJECTORY_SCHEMA_VERSION,
            "backend_kind": profile.backend_kind,
            "engine_repository": profile.engine_repository,
            "engine_revision": revision,
            "engine_version": version,
            "producer_version": producer_ver,
            "python_version": platform.python_version(),
            "device": registered_device,
            "device_name": registered_device_name,
            "coordinate_frame": CANONICAL_COORDINATE_FRAME,
            "position_units": "m",
            "time_units": "s",
            "frame_count": int(validated_driven.shape[0]),
            "state_count": int(validated_driven.shape[1]),
            "query_count": int(len(validated_indices)),
            "time_step_s": step_s,
            "solver_family": profile.solver_family,
            "identity_kind": profile.identity_kind,
            "simulation": {
                "scene_id": registered_scene,
                "model_kind": registered_model,
                "constitutive_model": registered_constitutive,
                "integrator": registered_integrator,
                "solver": registered_solver,
                "substeps": registered_substeps,
                "engine_parameters": attestations,
            },
            "information_boundary": _INFORMATION_BOUNDARY,
            "raw_rollout_sha256": file_sha256(raw_path),
        }
        runtime = {**identity, "runtime_id": content_id(identity)}
        write_atomic_json(runtime, runtime_path, overwrite=False)
        validate_material_runtime_manifest(runtime, raw_rollout_path=raw_path)
        return materialize_material_trajectory_backend(
            raw_rollout_path=raw_path,
            runtime_manifest_path=runtime_path,
            output_dir=output,
        )


__all__ = [
    "CallbackMaterialTrajectoryReplayV1",
    "DrakeDeformableBodyReplayV1",
    "MATERIAL_TRAJECTORY_PRODUCER_PROTOCOL",
    "MaterialTrajectoryReplayV1",
    "ReplayControl",
    "ReplayFactory",
    "produce_material_trajectory_backend",
]
