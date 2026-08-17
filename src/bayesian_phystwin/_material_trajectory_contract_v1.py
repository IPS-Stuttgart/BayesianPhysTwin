"""Strict fixed-material trajectory contract for external simulators."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
)
from .physical_rollout_v1 import validate_physical_rollout_arrays

MATERIAL_TRAJECTORY_RUNTIME_SCHEMA: Final = (
    "bayesian-phystwin.material-trajectory-runtime"
)
MATERIAL_TRAJECTORY_SCHEMA_VERSION: Final = 1
MATERIAL_TRAJECTORY_RAW_ARRAY_NAMES: Final = frozenset(
    {
        "driven_material_positions_m",
        "zero_action_material_positions_m",
        "material_query_indices",
        "action_support",
    }
)
CANONICAL_COORDINATE_FRAME: Final = "right-handed-z-up-world-v1"


@dataclass(frozen=True, slots=True)
class MaterialBackendProfile:
    """Immutable engine facts required by the portable bridge."""

    backend_kind: str
    engine_repository: str
    solver_family: str
    identity_kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "backend_kind": self.backend_kind,
            "engine_repository": self.engine_repository,
            "solver_family": self.solver_family,
            "identity_kind": self.identity_kind,
        }


MATERIAL_BACKEND_PROFILES: Final[Mapping[str, MaterialBackendProfile]] = (
    MappingProxyType(
        {
            "warp-fem-v1": MaterialBackendProfile(
                backend_kind="warp-fem-v1",
                engine_repository="NVIDIA/warp",
                solver_family="gpu-differentiable-fem",
                identity_kind="mesh-node-index",
            ),
            "sofa-fem-v1": MaterialBackendProfile(
                backend_kind="sofa-fem-v1",
                engine_repository="sofa-framework/sofa",
                solver_family="finite-element-method",
                identity_kind="mechanical-node-index",
            ),
            "genesis-mpm-v1": MaterialBackendProfile(
                backend_kind="genesis-mpm-v1",
                engine_repository="Genesis-Embodied-AI/genesis-world",
                solver_family="material-point-method",
                identity_kind="material-particle-index",
            ),
            "position-based-dynamics-v1": MaterialBackendProfile(
                backend_kind="position-based-dynamics-v1",
                engine_repository=("InteractiveComputerGraphics/PositionBasedDynamics"),
                solver_family="position-based-dynamics-xpbd",
                identity_kind="simulation-particle-index",
            ),
            "physx-fem-v1": MaterialBackendProfile(
                backend_kind="physx-fem-v1",
                engine_repository="NVIDIA-Omniverse/PhysX",
                solver_family="gpu-fem-deformables",
                identity_kind="deformable-simulation-vertex-index",
            ),
            "mujoco-flex-v1": MaterialBackendProfile(
                backend_kind="mujoco-flex-v1",
                engine_repository="google-deepmind/mujoco",
                solver_family="mujoco-flex",
                identity_kind="flex-vertex-index",
            ),
            "drake-fem-v1": MaterialBackendProfile(
                backend_kind="drake-fem-v1",
                engine_repository="RobotLocomotion/drake",
                solver_family="finite-element-method",
                identity_kind="deformable-body-vertex-index",
            ),
        }
    )
)

_RUNTIME_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_kind",
        "engine_repository",
        "engine_revision",
        "engine_version",
        "producer_version",
        "python_version",
        "device",
        "device_name",
        "coordinate_frame",
        "position_units",
        "time_units",
        "frame_count",
        "state_count",
        "query_count",
        "time_step_s",
        "solver_family",
        "identity_kind",
        "simulation",
        "information_boundary",
        "raw_rollout_sha256",
        "runtime_id",
    }
)
_SIMULATION_FIELDS: Final = frozenset(
    {
        "scene_id",
        "model_kind",
        "constitutive_model",
        "integrator",
        "solver",
        "substeps",
        "engine_parameters",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "future_observations_read",
        "future_outcomes_read",
        "known_action_used",
        "action_support_uses_observation_residuals",
        "material_query_indices_fixed_at_frame_zero",
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


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: npt.NDArray[Any]) -> str:
    """Hash an array while binding its dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def get_material_backend_profile(backend_kind: object) -> MaterialBackendProfile:
    """Resolve one supported engine profile or fail closed."""

    name = nonempty_string(backend_kind, name="backend_kind")
    try:
        return MATERIAL_BACKEND_PROFILES[name]
    except KeyError as error:
        supported = ", ".join(sorted(MATERIAL_BACKEND_PROFILES))
        raise ValueError(
            f"unsupported material backend kind {name!r}; supported: {supported}"
        ) from error


def material_backend_profile_records() -> tuple[dict[str, str], ...]:
    """Return sorted JSON-ready profile records for discovery tooling."""

    return tuple(
        MATERIAL_BACKEND_PROFILES[name].to_dict()
        for name in sorted(MATERIAL_BACKEND_PROFILES)
    )


def validate_material_runtime_manifest(
    value: Mapping[str, Any],
    *,
    raw_rollout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one content-addressed external simulator runtime record."""

    require_exact_fields(value, expected=_RUNTIME_FIELDS, name="runtime manifest")
    _require(
        value.get("schema") == MATERIAL_TRAJECTORY_RUNTIME_SCHEMA,
        "runtime schema changed",
    )
    _require(
        value.get("schema_version") == MATERIAL_TRAJECTORY_SCHEMA_VERSION,
        "runtime schema version changed",
    )
    profile = get_material_backend_profile(value.get("backend_kind"))
    repository = repository_name(
        value.get("engine_repository"), name="engine_repository"
    )
    _require(
        repository == profile.engine_repository,
        "runtime engine repository does not match backend profile",
    )
    exact_revision(value.get("engine_revision"), name="engine_revision")
    for name in (
        "engine_version",
        "producer_version",
        "python_version",
        "device",
        "device_name",
    ):
        nonempty_string(value.get(name), name=name)
    _require(
        value.get("coordinate_frame") == CANONICAL_COORDINATE_FRAME,
        "coordinate frame must be the canonical right-handed z-up world frame",
    )
    _require(value.get("position_units") == "m", "position units must be metres")
    _require(value.get("time_units") == "s", "time units must be seconds")
    frame_count = _positive_integer(value.get("frame_count"), name="frame_count")
    _require(frame_count >= 2, "frame_count must be at least two")
    _positive_integer(value.get("state_count"), name="state_count")
    _positive_integer(value.get("query_count"), name="query_count")
    _finite_positive(value.get("time_step_s"), name="time_step_s")
    _require(
        value.get("solver_family") == profile.solver_family,
        "runtime solver family does not match backend profile",
    )
    _require(
        value.get("identity_kind") == profile.identity_kind,
        "runtime identity kind does not match backend profile",
    )

    simulation = _mapping(value.get("simulation"), name="simulation")
    require_exact_fields(simulation, expected=_SIMULATION_FIELDS, name="simulation")
    for name in (
        "scene_id",
        "model_kind",
        "constitutive_model",
        "integrator",
        "solver",
    ):
        nonempty_string(simulation.get(name), name=f"simulation.{name}")
    _positive_integer(simulation.get("substeps"), name="simulation.substeps")
    parameters = frozen_finite_json_mapping(
        _mapping(
            simulation.get("engine_parameters"),
            name="simulation.engine_parameters",
        ),
        name="simulation.engine_parameters",
    )
    _require(bool(parameters), "simulation.engine_parameters must not be empty")

    boundary = _mapping(value.get("information_boundary"), name="information_boundary")
    require_exact_fields(
        boundary, expected=_BOUNDARY_FIELDS, name="information_boundary"
    )
    _require(
        dict(boundary) == _INFORMATION_BOUNDARY,
        "runtime information boundary changed",
    )
    raw_digest = sha256_digest(
        value.get("raw_rollout_sha256"), name="raw_rollout_sha256"
    )
    identity = {key: item for key, item in value.items() if key != "runtime_id"}
    _require(
        value.get("runtime_id") == content_id(identity), "runtime identity changed"
    )
    if raw_rollout_path is not None:
        raw_path = _ordinary_file(raw_rollout_path, name="raw rollout")
        _require(file_sha256(raw_path) == raw_digest, "raw rollout SHA-256 changed")
    return cast(dict[str, Any], plain_json(value))


def load_material_trajectory_rollout(
    path: str | Path,
) -> tuple[Path, dict[str, npt.NDArray[Any]]]:
    """Load fixed-identity material trajectories without pickle."""

    source = _ordinary_file(path, name="raw rollout")
    try:
        with np.load(source, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(MATERIAL_TRAJECTORY_RAW_ARRAY_NAMES),
                "raw material trajectory array roster changed",
            )
            arrays = {
                name: np.ascontiguousarray(np.asarray(stored[name])).copy()
                for name in stored.files
            }
    except (OSError, ValueError) as error:
        raise ValueError("cannot load raw material trajectory rollout") from error

    driven = arrays["driven_material_positions_m"]
    zero = arrays["zero_action_material_positions_m"]
    indices = arrays["material_query_indices"]
    support = arrays["action_support"]
    _require(
        driven.ndim == 3
        and driven.shape[0] >= 2
        and driven.shape[1] >= 1
        and driven.shape[2] == 3,
        "material positions must have shape (T,S,3)",
    )
    _require(zero.shape == driven.shape, "driven and zero-action shapes differ")
    _require(
        np.issubdtype(driven.dtype, np.floating)
        and np.issubdtype(zero.dtype, np.floating)
        and driven.dtype == zero.dtype,
        "material positions must share a floating dtype",
    )
    _require(
        np.all(np.isfinite(driven)) and np.all(np.isfinite(zero)),
        "material positions are non-finite",
    )
    _require(
        np.array_equal(driven[0], zero[0]),
        "driven and zero-action rollouts differ at frame zero",
    )
    _require(
        indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer),
        "material_query_indices must be an integer vector",
    )
    _require(len(indices) >= 1, "material_query_indices must not be empty")
    _require(
        len(np.unique(indices)) == len(indices),
        "material_query_indices must be unique",
    )
    _require(
        np.all((indices >= 0) & (indices < driven.shape[1])),
        "material query index exceeds state count",
    )
    _require(
        support.shape == (len(indices),)
        and np.issubdtype(support.dtype, np.floating)
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action_support is invalid",
    )
    _require(support.dtype == driven.dtype, "action_support dtype differs")
    return source, arrays


def physical_rollout_from_material_trajectory(
    arrays: Mapping[str, npt.NDArray[Any]],
) -> dict[str, FloatArray]:
    """Project persistent material identities into the physical-rollout API."""

    driven = np.asarray(arrays["driven_material_positions_m"])
    zero = np.asarray(arrays["zero_action_material_positions_m"])
    indices = np.asarray(arrays["material_query_indices"], dtype=np.int64)
    support = np.asarray(arrays["action_support"])
    frame_zero = np.ascontiguousarray(driven[0, indices])
    prediction = np.ascontiguousarray(driven[:, indices])
    zero_query = np.ascontiguousarray(zero[:, indices])
    persistence = np.ascontiguousarray(
        np.repeat(frame_zero[None], prediction.shape[0], axis=0)
    )
    physical = {
        "prediction_m": prediction,
        "persistence_m": persistence,
        "driven_readout_m": prediction.copy(),
        "zero_action_readout_m": zero_query,
        "action_support": np.ascontiguousarray(support),
        "frame_zero_points_m": frame_zero,
    }
    return validate_physical_rollout_arrays(physical)


__all__ = [
    "CANONICAL_COORDINATE_FRAME",
    "MATERIAL_BACKEND_PROFILES",
    "MATERIAL_TRAJECTORY_RAW_ARRAY_NAMES",
    "MATERIAL_TRAJECTORY_RUNTIME_SCHEMA",
    "MATERIAL_TRAJECTORY_SCHEMA_VERSION",
    "MaterialBackendProfile",
    "array_sha256",
    "file_sha256",
    "get_material_backend_profile",
    "load_material_trajectory_rollout",
    "material_backend_profile_records",
    "physical_rollout_from_material_trajectory",
    "validate_material_runtime_manifest",
]
