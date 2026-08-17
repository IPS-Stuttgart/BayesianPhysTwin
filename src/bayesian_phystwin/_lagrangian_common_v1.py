"""Shared constants and generic validation helpers for Lagrangian backends."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeAlias, TypedDict, cast

import numpy as np
import numpy.typing as npt

LAGRANGIAN_RUNTIME_SCHEMA: Final = "bayesian-phystwin.lagrangian-backend-runtime"
LAGRANGIAN_ARTIFACT_SCHEMA: Final = "bayesian-phystwin.lagrangian-backend"
LAGRANGIAN_SCHEMA_VERSION: Final = 1
JAX_FEM_PROFILE: Final = "jax-fem-quasistatic-v1"
GENESIS_MPM_PROFILE: Final = "genesis-world-mpm-v1"
JAX_FEM_REPOSITORY: Final = "deepmodeling/jax-fem"
GENESIS_WORLD_REPOSITORY: Final = "Genesis-Embodied-AI/genesis-world"
RAW_ARCHIVE_FILENAME: Final = "lagrangian-rollout.npz"
RUNTIME_FILENAME: Final = "lagrangian-runtime.json"
PHYSICAL_ARCHIVE_FILENAME: Final = "physical-prediction.npz"
ARTIFACT_FILENAME: Final = "lagrangian-backend.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

LAGRANGIAN_RAW_ARRAY_NAMES: Final = frozenset(
    {
        "driven_point_positions_m",
        "zero_action_point_positions_m",
        "material_query_indices",
        "action_support",
    }
)

LAGRANGIAN_BACKEND_CLAIM_BOUNDARY: Final = (
    "A content-addressed compatibility bridge for an externally executed "
    "fixed-identity deformable solver. The bundle validates exported point "
    "identity, units, provenance, and the portable physical-rollout mapping. "
    "It does not by itself prove that the upstream solver executed correctly, "
    "that gradients are correct, that real-data transfer succeeds, or that "
    "predictions are calibrated or state of the art."
)


class _ProfileDefinition(TypedDict):
    engine_repository: str
    identity_kind: str
    solver_family: str
    step_axis: str
    step_units: str
    metadata_fields: frozenset[str]


_PROFILE_DEFINITIONS: Final[dict[str, _ProfileDefinition]] = {
    JAX_FEM_PROFILE: {
        "engine_repository": JAX_FEM_REPOSITORY,
        "identity_kind": "mesh-node",
        "solver_family": "differentiable-fem",
        "step_axis": "load-step",
        "step_units": "1",
        "metadata_fields": frozenset(
            {
                "element_type",
                "constitutive_model",
                "nonlinear_solver",
                "differentiation_mode",
                "precision",
            }
        ),
    },
    GENESIS_MPM_PROFILE: {
        "engine_repository": GENESIS_WORLD_REPOSITORY,
        "identity_kind": "material-particle",
        "solver_family": "explicit-mpm",
        "step_axis": "time",
        "step_units": "s",
        "metadata_fields": frozenset(
            {
                "solver",
                "material_model",
                "compute_backend",
                "particle_size_m",
                "substeps",
                "gravity_m_s2",
                "differentiable",
                "precision",
            }
        ),
    },
}
LAGRANGIAN_BACKEND_PROFILES: Final[Mapping[str, _ProfileDefinition]] = MappingProxyType(
    _PROFILE_DEFINITIONS
)

_RUNTIME_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_profile",
        "engine_repository",
        "engine_revision",
        "engine_version",
        "python_version",
        "device",
        "coordinate_frame",
        "position_units",
        "step_axis",
        "step_units",
        "step_size",
        "frame_count",
        "point_count",
        "query_count",
        "identity_kind",
        "solver_family",
        "backend_metadata",
        "source_artifacts",
        "information_boundary",
        "raw_rollout_sha256",
        "runtime_id",
    }
)
_INFORMATION_BOUNDARY_FIELDS: Final = frozenset(
    {
        "source_kind",
        "dataset_payload_read",
        "future_observations_read",
        "outcomes_read",
        "known_action_used",
    }
)
_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_profile",
        "runtime_id",
        "inputs",
        "output",
        "mapping",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_INPUT_FIELDS: Final = frozenset({"raw_rollout", "runtime_manifest"})
_FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
_MAPPING_FIELDS: Final = frozenset(
    {
        "frame_count",
        "point_count",
        "query_count",
        "identity_kind",
        "material_identity_preserved",
        "query_indices_sha256",
        "coordinate_frame",
        "position_units",
        "step_axis",
        "step_units",
    }
)
_ROOT_ROSTER: Final = frozenset(
    {ARTIFACT_FILENAME, CHECKSUMS_FILENAME, PHYSICAL_ARCHIVE_FILENAME, "provenance"}
)
_PROVENANCE_ROSTER: Final = frozenset({RAW_ARCHIVE_FILENAME, RUNTIME_FILENAME})

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


def _finite_vector3(value: object, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{name} must contain three finite numbers")
    result = [float(item) for item in value]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite numbers")
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: npt.NDArray[Any]) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def describe_lagrangian_backend_profiles() -> list[dict[str, object]]:
    """Return a stable JSON-ready description of supported producer profiles."""

    return [
        {
            "backend_profile": profile,
            "engine_repository": definition["engine_repository"],
            "identity_kind": definition["identity_kind"],
            "solver_family": definition["solver_family"],
            "step_axis": definition["step_axis"],
            "step_units": definition["step_units"],
            "required_backend_metadata": sorted(definition["metadata_fields"]),
        }
        for profile, definition in sorted(LAGRANGIAN_BACKEND_PROFILES.items())
    ]
