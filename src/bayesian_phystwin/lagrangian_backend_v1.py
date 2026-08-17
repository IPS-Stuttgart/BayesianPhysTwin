"""Content-addressed bridges for external fixed-identity deformable solvers.

JAX-FEM and Genesis World remain optional external producers. This experimental
facade validates their fixed-identity exports and maps them into the portable
``physical_rollout_v1`` contract without importing either runtime.
"""

from ._lagrangian_artifact_v1 import (
    materialize_lagrangian_backend,
    validate_lagrangian_backend,
)
from ._lagrangian_common_v1 import (
    ARTIFACT_FILENAME,
    CHECKSUMS_FILENAME,
    GENESIS_MPM_PROFILE,
    GENESIS_WORLD_REPOSITORY,
    JAX_FEM_PROFILE,
    JAX_FEM_REPOSITORY,
    LAGRANGIAN_ARTIFACT_SCHEMA,
    LAGRANGIAN_BACKEND_CLAIM_BOUNDARY,
    LAGRANGIAN_BACKEND_PROFILES,
    LAGRANGIAN_RAW_ARRAY_NAMES,
    LAGRANGIAN_RUNTIME_SCHEMA,
    LAGRANGIAN_SCHEMA_VERSION,
    PHYSICAL_ARCHIVE_FILENAME,
    RAW_ARCHIVE_FILENAME,
    RUNTIME_FILENAME,
    array_sha256,
    describe_lagrangian_backend_profiles,
    file_sha256,
)
from ._lagrangian_runtime_v1 import (
    load_lagrangian_rollout,
    physical_rollout_from_lagrangian_points,
    validate_lagrangian_runtime_manifest,
)

__all__ = [
    "ARTIFACT_FILENAME",
    "CHECKSUMS_FILENAME",
    "GENESIS_MPM_PROFILE",
    "GENESIS_WORLD_REPOSITORY",
    "JAX_FEM_PROFILE",
    "JAX_FEM_REPOSITORY",
    "LAGRANGIAN_ARTIFACT_SCHEMA",
    "LAGRANGIAN_BACKEND_CLAIM_BOUNDARY",
    "LAGRANGIAN_BACKEND_PROFILES",
    "LAGRANGIAN_RAW_ARRAY_NAMES",
    "LAGRANGIAN_RUNTIME_SCHEMA",
    "LAGRANGIAN_SCHEMA_VERSION",
    "PHYSICAL_ARCHIVE_FILENAME",
    "RAW_ARCHIVE_FILENAME",
    "RUNTIME_FILENAME",
    "array_sha256",
    "describe_lagrangian_backend_profiles",
    "file_sha256",
    "load_lagrangian_rollout",
    "materialize_lagrangian_backend",
    "physical_rollout_from_lagrangian_points",
    "validate_lagrangian_backend",
    "validate_lagrangian_runtime_manifest",
]
