"""Fresh-solve producer for the registered JAX-FEM backend profile.

JAX and JAX-FEM remain optional producer-side dependencies. Callers provide a
small wrapper around a fresh, already-configured JAX-FEM problem; this module
records matched driven and zero-action solve sequences and publishes them only
through the existing strict ``lagrangian-export-v1`` transport.
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

from ._canonical_contracts import integer_array, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    source_artifact_mapping,
    write_atomic_json,
)
from .lagrangian_backend_v1 import (
    JAX_FEM_PROFILE,
    JAX_FEM_REPOSITORY,
    LAGRANGIAN_BACKEND_PROFILES,
    LAGRANGIAN_RUNTIME_SCHEMA,
    LAGRANGIAN_SCHEMA_VERSION,
    file_sha256,
    load_lagrangian_rollout,
    materialize_lagrangian_backend,
    validate_lagrangian_runtime_manifest,
)
from .physical_rollout_v1 import write_deterministic_npz

JAX_FEM_PRODUCER_PROTOCOL: Final = "fresh-quasistatic-control-before-solve-v1"
JAX_FEM_SOLUTION_SEMANTICS: Final = "total-nodal-displacement-from-fixed-reference-v1"
_PRODUCER_SOURCE_ARTIFACT: Final = "bayesian_phystwin/jax_fem_producer_v1.py"
_CANONICAL_COORDINATE_FRAME: Final = "right-handed-z-up-world-v1"
_SUPPORTED_FLOAT_DTYPES: Final[frozenset[np.dtype[Any]]] = frozenset(
    {np.dtype("float32"), np.dtype("float64")}
)

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]
ReplayFactory: TypeAlias = Callable[[], "JaxFemReplayV1"]
ReplayControl: TypeAlias = Callable[[int, "JaxFemReplayV1"], None]


class JaxFemReplayV1(Protocol):
    """Minimum producer-side wrapper consumed by the JAX-FEM adapter.

    Reference points must retain one fixed mesh-node order for the complete
    replay. ``solve`` returns one selected displacement field interpreted as
    total nodal displacement from the fixed reference mesh after the current
    load step. A multi-field JAX-FEM wrapper must select its displacement field
    before returning so that field selection is bound by the wrapper source.
    """

    def get_reference_points_m(self) -> object:
        """Return fixed reference mesh-node positions with shape ``(P, 3)``."""

        ...

    def solve(self) -> object:
        """Solve the current load step and return one selected displacement."""

        ...


@dataclass(slots=True)
class CallbackJaxFemReplayV1:
    """Build a replay wrapper from reference and solve callbacks."""

    reference_points_callback: Callable[[], object]
    solve_callback: Callable[[], object]
    context: object | None = None

    def __post_init__(self) -> None:
        if not callable(self.reference_points_callback):
            raise TypeError("reference_points_callback must be callable")
        if not callable(self.solve_callback):
            raise TypeError("solve_callback must be callable")

    def get_reference_points_m(self) -> object:
        return self.reference_points_callback()

    def solve(self) -> object:
        return self.solve_callback()


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
    """Synchronize common array facades and copy into contiguous host memory."""

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


def _build_replay(replay_factory: ReplayFactory) -> JaxFemReplayV1:
    replay = replay_factory()
    if not callable(getattr(replay, "get_reference_points_m", None)):
        raise ValueError("JAX-FEM replay must expose get_reference_points_m()")
    if not callable(getattr(replay, "solve", None)):
        raise ValueError("JAX-FEM replay must expose solve()")
    return replay


def _capture_reference(
    replay: JaxFemReplayV1,
    *,
    label: str,
) -> FloatArray:
    points = _to_numpy(replay.get_reference_points_m())
    _require(
        points.ndim == 2 and points.shape[0] >= 1 and points.shape[1] == 3,
        f"{label} JAX-FEM reference points must have shape (P,3)",
    )
    _require(
        points.dtype in _SUPPORTED_FLOAT_DTYPES,
        f"{label} JAX-FEM reference points must use float32 or float64",
    )
    _require(
        np.all(np.isfinite(points)),
        f"{label} JAX-FEM reference points contain non-finite values",
    )
    return cast(FloatArray, points)


def _capture_displacement(
    replay: JaxFemReplayV1,
    *,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype[Any],
    label: str,
) -> FloatArray:
    solved = replay.solve()
    _require(
        not isinstance(solved, (list, tuple)),
        "JAX-FEM replay must select one displacement field before publication",
    )
    displacement = _to_numpy(solved)
    _require(
        displacement.shape == expected_shape,
        f"{label} JAX-FEM displacement shape differs from reference mesh nodes",
    )
    _require(
        displacement.dtype == expected_dtype,
        f"{label} JAX-FEM displacement dtype differs from reference points",
    )
    _require(
        np.all(np.isfinite(displacement)),
        f"{label} JAX-FEM displacement contains non-finite values",
    )
    return cast(FloatArray, displacement)


def _record_replay(
    replay: JaxFemReplayV1,
    control: ReplayControl,
    *,
    reference: FloatArray,
    frame_count: int,
    label: str,
) -> FloatArray:
    frames = [reference.copy()]
    expected_shape = cast(tuple[int, int], reference.shape)
    expected_dtype = reference.dtype
    for transition_index in range(frame_count - 1):
        control(transition_index, replay)
        displacement = _capture_displacement(
            replay,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            label=label,
        )
        current_reference = _capture_reference(replay, label=label)
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
    return cast(FloatArray, trajectory)


def _query_indices(values: Sequence[int] | IntegerArray) -> npt.NDArray[np.int64]:
    indices = np.ascontiguousarray(
        integer_array(values, name="material_query_indices"),
        dtype=np.dtype("<i8"),
    )
    _require(indices.ndim == 1, "material_query_indices must be an integer vector")
    _require(len(indices) >= 1, "material_query_indices must not be empty")
    _require(
        len(np.unique(indices)) == len(indices),
        "material_query_indices must be unique",
    )
    return indices


def _raw_action_support(
    values: Sequence[float] | FloatArray,
    *,
    query_count: int,
) -> npt.NDArray[Any]:
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
    _require(
        np.all(np.isfinite(raw)) and np.all((raw >= 0.0) & (raw <= 1.0)),
        "action_support must be a finite vector in [0,1]",
    )
    return raw


def _source_artifacts(values: Mapping[str, str]) -> dict[str, Any]:
    artifacts = cast(
        dict[str, Any],
        plain_json(source_artifact_mapping(values, name="source_artifacts")),
    )
    _require(
        _PRODUCER_SOURCE_ARTIFACT not in artifacts,
        "source_artifacts contains the reserved producer source path",
    )
    artifacts[_PRODUCER_SOURCE_ARTIFACT] = file_sha256(Path(__file__))
    return cast(
        dict[str, Any],
        plain_json(source_artifact_mapping(artifacts, name="source_artifacts")),
    )


def produce_jax_fem_backend(
    *,
    output_dir: str | Path,
    replay_factory: ReplayFactory,
    driven_control: ReplayControl,
    zero_action_control: ReplayControl,
    frame_count: int,
    material_query_indices: Sequence[int] | IntegerArray,
    action_support: Sequence[float] | FloatArray,
    engine_revision: str,
    engine_version: str,
    source_artifacts: Mapping[str, str],
    device: str,
    load_step_size: float,
    element_type: str,
    constitutive_model: str,
    nonlinear_solver: str,
    source_kind: str = "synthetic",
) -> dict[str, Any]:
    """Run matched fresh JAX-FEM solves and publish one portable backend bundle.

    Frame zero is the fixed reference mesh. For each later frame, the arm-specific
    control callback runs immediately before ``solve``. The selected solution
    field must be total nodal displacement from that fixed reference. A successful
    publication calls the factory exactly twice and requires distinct wrappers;
    the wrapper implementation remains responsible for fresh underlying solver,
    warm-start, and boundary-condition state.
    """

    output = _new_output_directory(output_dir)
    frames = _positive_integer(frame_count, name="frame_count", minimum=2)
    step_size = _positive_float(load_step_size, name="load_step_size")
    if not callable(replay_factory):
        raise ValueError("replay_factory must be callable")
    if not callable(driven_control) or not callable(zero_action_control):
        raise ValueError("driven and zero-action controls must be callable")

    revision = exact_revision(engine_revision, name="engine_revision")
    version = nonempty_string(engine_version, name="engine_version")
    registered_device = nonempty_string(device, name="device")
    registered_element = nonempty_string(element_type, name="element_type")
    registered_constitutive = nonempty_string(
        constitutive_model, name="constitutive_model"
    )
    registered_solver = nonempty_string(nonlinear_solver, name="nonlinear_solver")
    registered_source_kind = nonempty_string(source_kind, name="source_kind")
    _require(
        registered_source_kind in {"synthetic", "source-only"},
        "source_kind must be synthetic or source-only",
    )
    artifacts = _source_artifacts(source_artifacts)
    indices = _query_indices(material_query_indices)
    raw_support = _raw_action_support(action_support, query_count=len(indices))

    driven_replay = _build_replay(replay_factory)
    driven_reference = _capture_reference(driven_replay, label="driven")
    _require(
        np.all((indices >= 0) & (indices < driven_reference.shape[0])),
        "material query index exceeds JAX-FEM node count",
    )
    driven = _record_replay(
        driven_replay,
        driven_control,
        reference=driven_reference,
        frame_count=frames,
        label="driven",
    )
    zero_replay = _build_replay(replay_factory)
    _require(
        zero_replay is not driven_replay,
        "replay_factory must return fresh JAX-FEM replay objects",
    )
    zero_reference = _capture_reference(zero_replay, label="zero-action")
    _require(
        zero_reference.shape == driven_reference.shape
        and zero_reference.dtype == driven_reference.dtype,
        "fresh JAX-FEM replay reference shapes or dtypes differ",
    )
    _require(
        np.array_equal(driven_reference, zero_reference),
        "fresh JAX-FEM replays differ in reference mesh points",
    )
    zero = _record_replay(
        zero_replay,
        zero_action_control,
        reference=zero_reference,
        frame_count=frames,
        label="zero-action",
    )
    _require(
        zero.shape == driven.shape and zero.dtype == driven.dtype,
        "driven and zero-action JAX-FEM replay shapes or dtypes differ",
    )
    support = cast(
        FloatArray,
        np.ascontiguousarray(raw_support, dtype=driven.dtype),
    )

    definition = LAGRANGIAN_BACKEND_PROFILES[JAX_FEM_PROFILE]
    precision = "float32" if driven.dtype == np.dtype("float32") else "float64"
    with tempfile.TemporaryDirectory(
        prefix=".jax-fem-producer-",
        dir=output.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        raw_path = temporary / "lagrangian-rollout.npz"
        runtime_path = temporary / "lagrangian-runtime.json"
        write_deterministic_npz(
            raw_path,
            {
                "driven_point_positions_m": driven,
                "zero_action_point_positions_m": zero,
                "material_query_indices": indices,
                "action_support": support,
            },
        )
        _, raw = load_lagrangian_rollout(raw_path)
        validated_driven = raw["driven_point_positions_m"]
        validated_indices = raw["material_query_indices"]
        identity: dict[str, Any] = {
            "schema": LAGRANGIAN_RUNTIME_SCHEMA,
            "schema_version": LAGRANGIAN_SCHEMA_VERSION,
            "backend_profile": JAX_FEM_PROFILE,
            "engine_repository": JAX_FEM_REPOSITORY,
            "engine_revision": revision,
            "engine_version": version,
            "python_version": platform.python_version(),
            "device": registered_device,
            "coordinate_frame": _CANONICAL_COORDINATE_FRAME,
            "position_units": "m",
            "step_axis": definition["step_axis"],
            "step_units": definition["step_units"],
            "step_size": step_size,
            "frame_count": int(validated_driven.shape[0]),
            "point_count": int(validated_driven.shape[1]),
            "query_count": int(len(validated_indices)),
            "identity_kind": definition["identity_kind"],
            "solver_family": definition["solver_family"],
            "backend_metadata": {
                "element_type": registered_element,
                "constitutive_model": registered_constitutive,
                "nonlinear_solver": registered_solver,
                "differentiation_mode": "jax-autodiff",
                "precision": precision,
            },
            "source_artifacts": artifacts,
            "information_boundary": {
                "source_kind": registered_source_kind,
                "dataset_payload_read": registered_source_kind == "source-only",
                "future_observations_read": False,
                "outcomes_read": False,
                "known_action_used": True,
            },
            "raw_rollout_sha256": file_sha256(raw_path),
        }
        runtime = {**identity, "runtime_id": content_id(identity)}
        write_atomic_json(runtime, runtime_path, overwrite=False)
        validate_lagrangian_runtime_manifest(
            runtime,
            raw_rollout_path=raw_path,
        )
        return materialize_lagrangian_backend(
            raw_rollout_path=raw_path,
            runtime_manifest_path=runtime_path,
            output_dir=output,
        )


__all__ = [
    "CallbackJaxFemReplayV1",
    "JAX_FEM_PRODUCER_PROTOCOL",
    "JAX_FEM_SOLUTION_SEMANTICS",
    "JaxFemReplayV1",
    "ReplayControl",
    "ReplayFactory",
    "produce_jax_fem_backend",
]
