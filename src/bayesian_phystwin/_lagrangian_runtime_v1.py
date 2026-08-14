"""Strict runtime and trajectory contracts for external Lagrangian solvers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._lagrangian_common_v1 import (
    GENESIS_MPM_PROFILE,
    JAX_FEM_PROFILE,
    LAGRANGIAN_BACKEND_PROFILES,
    LAGRANGIAN_RAW_ARRAY_NAMES,
    LAGRANGIAN_RUNTIME_SCHEMA,
    LAGRANGIAN_SCHEMA_VERSION,
    FloatArray,
    _INFORMATION_BOUNDARY_FIELDS,
    _RUNTIME_FIELDS,
    _finite_positive,
    _finite_vector3,
    _mapping,
    _ordinary_file,
    _positive_integer,
    _require,
    file_sha256,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)
from .physical_rollout_v1 import validate_physical_rollout_arrays


def _validate_backend_metadata(
    profile: str,
    value: object,
) -> dict[str, Any]:
    metadata = _mapping(value, name="backend_metadata")
    definition = LAGRANGIAN_BACKEND_PROFILES[profile]
    expected_fields = definition["metadata_fields"]
    require_exact_fields(metadata, expected=expected_fields, name="backend_metadata")

    precision = nonempty_string(metadata.get("precision"), name="precision")
    _require(
        precision in {"float32", "float64"},
        "precision must be float32 or float64",
    )

    if profile == JAX_FEM_PROFILE:
        for name in (
            "element_type",
            "constitutive_model",
            "nonlinear_solver",
        ):
            nonempty_string(metadata.get(name), name=name)
        _require(
            metadata.get("differentiation_mode") == "jax-autodiff",
            "JAX-FEM differentiation_mode must be jax-autodiff",
        )
    elif profile == GENESIS_MPM_PROFILE:
        _require(metadata.get("solver") == "mpm", "Genesis solver must be mpm")
        for name in ("material_model", "compute_backend"):
            nonempty_string(metadata.get(name), name=name)
        _finite_positive(metadata.get("particle_size_m"), name="particle_size_m")
        _positive_integer(metadata.get("substeps"), name="substeps")
        _finite_vector3(metadata.get("gravity_m_s2"), name="gravity_m_s2")
        _require(
            type(metadata.get("differentiable")) is bool,
            "differentiable must be a boolean",
        )
    else:  # pragma: no cover - profile admission happens before dispatch
        raise AssertionError(profile)
    return cast(dict[str, Any], plain_json(metadata))


def _validate_information_boundary(value: object) -> dict[str, Any]:
    boundary = _mapping(value, name="information_boundary")
    require_exact_fields(
        boundary,
        expected=_INFORMATION_BOUNDARY_FIELDS,
        name="information_boundary",
    )
    source_kind = nonempty_string(boundary.get("source_kind"), name="source_kind")
    _require(
        source_kind in {"synthetic", "source-only"},
        "source_kind must be synthetic or source-only",
    )
    for name in (
        "dataset_payload_read",
        "future_observations_read",
        "outcomes_read",
        "known_action_used",
    ):
        _require(type(boundary.get(name)) is bool, f"{name} must be a boolean")
    _require(
        boundary.get("future_observations_read") is False,
        "future observations must remain closed",
    )
    _require(boundary.get("outcomes_read") is False, "outcomes must remain closed")
    _require(boundary.get("known_action_used") is True, "known action must be used")
    if source_kind == "synthetic":
        _require(
            boundary.get("dataset_payload_read") is False,
            "synthetic exports must not read dataset payloads",
        )
    else:
        _require(
            boundary.get("dataset_payload_read") is True,
            "source-only exports must bind dataset payload access",
        )
    return cast(dict[str, Any], plain_json(boundary))


def validate_lagrangian_runtime_manifest(
    value: Mapping[str, Any],
    *,
    raw_rollout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one exact, content-addressed external runtime manifest."""

    require_exact_fields(value, expected=_RUNTIME_FIELDS, name="runtime manifest")
    _require(value.get("schema") == LAGRANGIAN_RUNTIME_SCHEMA, "runtime schema changed")
    _require(
        value.get("schema_version") == LAGRANGIAN_SCHEMA_VERSION,
        "runtime schema version changed",
    )
    profile = nonempty_string(value.get("backend_profile"), name="backend_profile")
    _require(profile in LAGRANGIAN_BACKEND_PROFILES, "unknown backend_profile")
    definition = LAGRANGIAN_BACKEND_PROFILES[profile]

    engine_repository = repository_name(
        value.get("engine_repository"), name="engine_repository"
    )
    _require(
        engine_repository == definition["engine_repository"],
        "engine repository does not match backend profile",
    )
    exact_revision(value.get("engine_revision"), name="engine_revision")
    for name in ("engine_version", "python_version", "device"):
        nonempty_string(value.get(name), name=name)

    _require(
        value.get("coordinate_frame") == "right-handed-z-up-world-v1",
        "coordinate frame changed",
    )
    _require(value.get("position_units") == "m", "position units must be metres")
    _require(value.get("step_axis") == definition["step_axis"], "step axis changed")
    _require(value.get("step_units") == definition["step_units"], "step units changed")
    _finite_positive(value.get("step_size"), name="step_size")
    frame_count = _positive_integer(value.get("frame_count"), name="frame_count")
    _require(frame_count >= 2, "frame_count must be at least two")
    _positive_integer(value.get("point_count"), name="point_count")
    _positive_integer(value.get("query_count"), name="query_count")
    _require(
        value.get("identity_kind") == definition["identity_kind"],
        "identity kind changed",
    )
    _require(
        value.get("solver_family") == definition["solver_family"],
        "solver family changed",
    )
    _validate_backend_metadata(profile, value.get("backend_metadata"))
    artifacts = _mapping(value.get("source_artifacts"), name="source_artifacts")
    source_artifact_mapping(
        cast(Mapping[str, str], artifacts),
        name="source_artifacts",
    )
    _validate_information_boundary(value.get("information_boundary"))
    raw_digest = sha256_digest(
        value.get("raw_rollout_sha256"), name="raw_rollout_sha256"
    )

    identity = {key: item for key, item in value.items() if key != "runtime_id"}
    _require(
        value.get("runtime_id") == content_id(identity),
        "runtime identity changed",
    )
    if raw_rollout_path is not None:
        raw_path = _ordinary_file(raw_rollout_path, name="raw rollout")
        _require(file_sha256(raw_path) == raw_digest, "raw rollout SHA-256 changed")
    return cast(dict[str, Any], plain_json(value))


def load_lagrangian_rollout(
    path: str | Path,
) -> tuple[Path, dict[str, npt.NDArray[Any]]]:
    """Load one no-pickle fixed-identity point rollout."""

    source = _ordinary_file(path, name="raw rollout")
    try:
        with np.load(source, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(LAGRANGIAN_RAW_ARRAY_NAMES),
                "raw Lagrangian array roster changed",
            )
            arrays = {
                name: np.ascontiguousarray(np.asarray(stored[name])).copy()
                for name in stored.files
            }
    except (OSError, ValueError) as error:
        raise ValueError("cannot load raw Lagrangian rollout") from error

    driven = arrays["driven_point_positions_m"]
    zero = arrays["zero_action_point_positions_m"]
    indices = arrays["material_query_indices"]
    support = arrays["action_support"]
    _require(
        driven.ndim == 3
        and driven.shape[0] >= 2
        and driven.shape[1] >= 1
        and driven.shape[2] == 3,
        "Lagrangian point positions must have shape (T,P,3)",
    )
    _require(zero.shape == driven.shape, "driven and zero-action shapes differ")
    _require(
        np.issubdtype(driven.dtype, np.floating)
        and np.issubdtype(zero.dtype, np.floating)
        and driven.dtype == zero.dtype,
        "Lagrangian point positions must share a floating dtype",
    )
    _require(
        driven.dtype in {np.dtype("float32"), np.dtype("float64")},
        "Lagrangian point positions must use float32 or float64",
    )
    _require(
        np.all(np.isfinite(driven)) and np.all(np.isfinite(zero)),
        "Lagrangian point positions are non-finite",
    )
    _require(
        np.array_equal(driven[0], zero[0]),
        "Lagrangian rollouts differ at frame zero",
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
        "material query index exceeds point count",
    )
    _require(
        support.shape == (len(indices),)
        and np.issubdtype(support.dtype, np.floating)
        and support.dtype == driven.dtype
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action_support is invalid",
    )
    return source, arrays


def physical_rollout_from_lagrangian_points(
    arrays: Mapping[str, npt.NDArray[Any]],
) -> dict[str, FloatArray]:
    """Project persistent mesh nodes or material particles into the shared contract."""

    driven = np.asarray(arrays["driven_point_positions_m"])
    zero = np.asarray(arrays["zero_action_point_positions_m"])
    indices = np.asarray(arrays["material_query_indices"], dtype=np.int64)
    support = np.asarray(arrays["action_support"])
    prediction = np.ascontiguousarray(driven[:, indices])
    zero_query = np.ascontiguousarray(zero[:, indices])
    frame_zero = np.ascontiguousarray(prediction[0])
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

