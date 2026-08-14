"""Strict external-physics producers for the portable physical rollout contract."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .physical_rollout_v1 import (
    PHYSICAL_ROLLOUT_ARRAY_NAMES,
    load_physical_rollout_archive,
    validate_physical_rollout_arrays,
    write_deterministic_npz,
)
from .physics_backend_registry_v1 import (
    PhysicsBackendProfileV1,
    profile_from_mapping,
)

EXTERNAL_PHYSICS_RUNTIME_SCHEMA: Final = (
    "bayesian-phystwin.external-physics-runtime"
)
EXTERNAL_PHYSICS_ARTIFACT_SCHEMA: Final = (
    "bayesian-phystwin.external-physics-backend"
)
EXTERNAL_PHYSICS_SCHEMA_VERSION: Final = 1
EXTERNAL_PHYSICS_RAW_ARRAY_NAMES: Final = frozenset(
    {
        "driven_entity_positions_m",
        "zero_action_entity_positions_m",
        "query_entity_indices",
        "action_support",
    }
)
RAW_ARCHIVE_FILENAME: Final = "external-entity-rollout.npz"
RUNTIME_FILENAME: Final = "external-runtime.json"
PHYSICAL_ARCHIVE_FILENAME: Final = "physical-prediction.npz"
ARTIFACT_FILENAME: Final = "external-physics-backend.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

EXTERNAL_PHYSICS_CLAIM_BOUNDARY: Final = (
    "A simulator-neutral external physics producer with persistent query identity, "
    "exact source revisions, and a causal information boundary. The artifact is "
    "a candidate physical rollout only: claim-bearing use still requires a frozen "
    "source-only guard, uncertainty evaluation, and exact incumbent fallback."
)

_RUNTIME_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_profile",
        "engine_revision",
        "engine_version",
        "producer_repository",
        "producer_revision",
        "producer_artifacts",
        "coordinate_frame",
        "position_units",
        "time_units",
        "frame_count",
        "entity_count",
        "query_count",
        "time_step_s",
        "entity_identity_sha256",
        "topology_sha256",
        "material_model",
        "parameterization",
        "information_boundary",
        "raw_rollout_sha256",
        "runtime_id",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "observation_end_frame_exclusive",
        "future_observations_used",
        "outcomes_used_for_selection",
        "target_outcomes_used",
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
_FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
_INPUT_FIELDS: Final = frozenset({"raw_rollout", "runtime_manifest"})
_MAPPING_FIELDS: Final = frozenset(
    {
        "frame_count",
        "entity_count",
        "query_count",
        "persistent_entity_identity_preserved",
        "entity_identity_sha256",
        "query_indices_sha256",
        "coordinate_frame",
        "position_units",
    }
)

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Hash one ordinary file in bounded blocks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: npt.NDArray[Any]) -> str:
    """Hash dtype, shape, and C-order bytes for one array identity."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with literal string keys")
    return cast(Mapping[str, Any], value)


def load_external_entity_rollout(
    path: str | Path,
) -> tuple[Path, dict[str, npt.NDArray[Any]]]:
    """Load a persistent-entity driven/zero-action rollout without pickle."""

    source = _ordinary_file(path, name="raw rollout")
    try:
        with np.load(source, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(EXTERNAL_PHYSICS_RAW_ARRAY_NAMES),
                "external rollout array roster changed",
            )
            arrays = {
                name: np.ascontiguousarray(np.asarray(stored[name])).copy()
                for name in stored.files
            }
    except (OSError, ValueError) as error:
        raise ValueError("cannot load external entity rollout") from error

    driven = arrays["driven_entity_positions_m"]
    zero = arrays["zero_action_entity_positions_m"]
    indices = arrays["query_entity_indices"]
    support = arrays["action_support"]
    _require(
        driven.ndim == 3
        and driven.shape[0] >= 2
        and driven.shape[1] >= 1
        and driven.shape[2] == 3,
        "external entity positions must have shape (T,E,3)",
    )
    _require(zero.shape == driven.shape, "driven and zero-action shapes differ")
    _require(
        np.issubdtype(driven.dtype, np.floating)
        and np.issubdtype(zero.dtype, np.floating)
        and driven.dtype == zero.dtype,
        "external entity positions must share a floating dtype",
    )
    _require(
        np.all(np.isfinite(driven)) and np.all(np.isfinite(zero)),
        "external entity positions are non-finite",
    )
    _require(
        np.array_equal(driven[0], zero[0]),
        "external rollouts differ at frame zero",
    )
    _require(
        indices.ndim == 1
        and np.issubdtype(indices.dtype, np.integer)
        and not np.issubdtype(indices.dtype, np.bool_),
        "query_entity_indices must be an integer vector",
    )
    _require(len(indices) >= 1, "query_entity_indices must not be empty")
    _require(
        len(np.unique(indices)) == len(indices),
        "query_entity_indices must be unique",
    )
    _require(
        np.all((indices >= 0) & (indices < driven.shape[1])),
        "query entity index exceeds entity count",
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


def physical_rollout_from_external_entities(
    arrays: Mapping[str, npt.NDArray[Any]],
) -> dict[str, FloatArray]:
    """Project stable entity indices into the portable six-array contract."""

    driven = np.asarray(arrays["driven_entity_positions_m"])
    zero = np.asarray(arrays["zero_action_entity_positions_m"])
    indices = np.asarray(arrays["query_entity_indices"], dtype=np.int64)
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


def _validate_information_boundary(
    value: object,
    *,
    frame_count: int,
) -> dict[str, object]:
    boundary = _mapping(value, name="information_boundary")
    require_exact_fields(
        boundary,
        expected=_BOUNDARY_FIELDS,
        name="information_boundary",
    )
    observation_end = _nonnegative_integer(
        boundary.get("observation_end_frame_exclusive"),
        name="observation_end_frame_exclusive",
    )
    _require(
        observation_end <= frame_count,
        "observation_end_frame_exclusive exceeds frame_count",
    )
    future = genuine_boolean(
        boundary.get("future_observations_used"),
        name="future_observations_used",
    )
    outcomes = genuine_boolean(
        boundary.get("outcomes_used_for_selection"),
        name="outcomes_used_for_selection",
    )
    target = genuine_boolean(
        boundary.get("target_outcomes_used"),
        name="target_outcomes_used",
    )
    known_action = genuine_boolean(
        boundary.get("known_action_used"),
        name="known_action_used",
    )
    _require(not future, "future observations are forbidden")
    _require(not outcomes, "outcomes used for selection are forbidden")
    _require(not target, "target outcomes are forbidden")
    _require(known_action, "the driven rollout must bind the known action")
    return {
        "observation_end_frame_exclusive": observation_end,
        "future_observations_used": False,
        "outcomes_used_for_selection": False,
        "target_outcomes_used": False,
        "known_action_used": True,
    }


def validate_external_physics_runtime_manifest(
    value: Mapping[str, Any],
    *,
    raw_rollout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one self-contained external-producer runtime manifest."""

    require_exact_fields(value, expected=_RUNTIME_FIELDS, name="runtime manifest")
    _require(
        value.get("schema") == EXTERNAL_PHYSICS_RUNTIME_SCHEMA,
        "runtime schema changed",
    )
    _require(
        value.get("schema_version") == EXTERNAL_PHYSICS_SCHEMA_VERSION,
        "runtime schema version changed",
    )
    profile = profile_from_mapping(
        _mapping(value.get("backend_profile"), name="backend_profile")
    )
    engine_revision = exact_revision(
        value.get("engine_revision"), name="engine_revision"
    )
    engine_version = nonempty_string(
        value.get("engine_version"), name="engine_version"
    )
    producer_repository = repository_name(
        value.get("producer_repository"), name="producer_repository"
    )
    producer_revision = exact_revision(
        value.get("producer_revision"), name="producer_revision"
    )
    artifacts_raw = _mapping(
        value.get("producer_artifacts"), name="producer_artifacts"
    )
    producer_artifacts = source_artifact_mapping(
        cast(Mapping[str, str], artifacts_raw),
        name="producer_artifacts",
        allow_empty=True,
    )
    coordinate_frame = nonempty_string(
        value.get("coordinate_frame"), name="coordinate_frame"
    )
    _require(
        coordinate_frame.strip() == coordinate_frame,
        "coordinate_frame must not contain surrounding whitespace",
    )
    _require(value.get("position_units") == "m", "position units must be metres")
    _require(value.get("time_units") == "s", "time units must be seconds")
    frame_count = _positive_integer(value.get("frame_count"), name="frame_count")
    _require(frame_count >= 2, "frame_count must be at least two")
    entity_count = _positive_integer(value.get("entity_count"), name="entity_count")
    query_count = _positive_integer(value.get("query_count"), name="query_count")
    _require(query_count <= entity_count, "query_count exceeds entity_count")
    time_step_s = _finite_positive(value.get("time_step_s"), name="time_step_s")
    entity_identity = sha256_digest(
        value.get("entity_identity_sha256"), name="entity_identity_sha256"
    )
    topology = sha256_digest(
        value.get("topology_sha256"), name="topology_sha256"
    )
    material_model = nonempty_string(
        value.get("material_model"), name="material_model"
    )
    parameterization_raw = _mapping(
        value.get("parameterization"), name="parameterization"
    )
    parameterization = frozen_finite_json_mapping(
        parameterization_raw,
        name="parameterization",
    )
    information_boundary = _validate_information_boundary(
        value.get("information_boundary"),
        frame_count=frame_count,
    )
    raw_digest = sha256_digest(
        value.get("raw_rollout_sha256"), name="raw_rollout_sha256"
    )
    canonical_identity: dict[str, Any] = {
        "schema": EXTERNAL_PHYSICS_RUNTIME_SCHEMA,
        "schema_version": EXTERNAL_PHYSICS_SCHEMA_VERSION,
        "backend_profile": profile.to_dict(),
        "engine_revision": engine_revision,
        "engine_version": engine_version,
        "producer_repository": producer_repository,
        "producer_revision": producer_revision,
        "producer_artifacts": plain_json(producer_artifacts),
        "coordinate_frame": coordinate_frame,
        "position_units": "m",
        "time_units": "s",
        "frame_count": frame_count,
        "entity_count": entity_count,
        "query_count": query_count,
        "time_step_s": time_step_s,
        "entity_identity_sha256": entity_identity,
        "topology_sha256": topology,
        "material_model": material_model,
        "parameterization": plain_json(parameterization),
        "information_boundary": information_boundary,
        "raw_rollout_sha256": raw_digest,
    }
    supplied_identity = {
        key: item for key, item in value.items() if key != "runtime_id"
    }
    _require(
        plain_json(supplied_identity) == canonical_identity,
        "runtime manifest is not canonical",
    )
    runtime_id = sha256_digest(value.get("runtime_id"), name="runtime_id")
    _require(runtime_id == content_id(canonical_identity), "runtime identity changed")

    if raw_rollout_path is not None:
        raw_path, arrays = load_external_entity_rollout(raw_rollout_path)
        driven = arrays["driven_entity_positions_m"]
        indices = np.asarray(arrays["query_entity_indices"], dtype=np.int64)
        _require(file_sha256(raw_path) == raw_digest, "raw rollout SHA-256 changed")
        _require(driven.shape[0] == frame_count, "runtime frame count differs")
        _require(driven.shape[1] == entity_count, "runtime entity count differs")
        _require(len(indices) == query_count, "runtime query count differs")
        _require(
            array_sha256(driven[0]) == entity_identity,
            "frame-zero entity identity changed",
        )
    return {**canonical_identity, "runtime_id": runtime_id}


def build_external_physics_runtime_manifest(
    *,
    raw_rollout_path: str | Path,
    profile: PhysicsBackendProfileV1,
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
) -> dict[str, Any]:
    """Build a runtime record from a validated raw rollout and exact revisions."""

    raw_path, arrays = load_external_entity_rollout(raw_rollout_path)
    driven = arrays["driven_entity_positions_m"]
    indices = np.asarray(arrays["query_entity_indices"], dtype=np.int64)
    frame_count = int(driven.shape[0])
    boundary = {
        "observation_end_frame_exclusive": observation_end_frame_exclusive,
        "future_observations_used": False,
        "outcomes_used_for_selection": False,
        "target_outcomes_used": False,
        "known_action_used": True,
    }
    identity: dict[str, Any] = {
        "schema": EXTERNAL_PHYSICS_RUNTIME_SCHEMA,
        "schema_version": EXTERNAL_PHYSICS_SCHEMA_VERSION,
        "backend_profile": profile.to_dict(),
        "engine_revision": engine_revision,
        "engine_version": engine_version,
        "producer_repository": producer_repository,
        "producer_revision": producer_revision,
        "producer_artifacts": dict(producer_artifacts or {}),
        "coordinate_frame": coordinate_frame,
        "position_units": "m",
        "time_units": "s",
        "frame_count": frame_count,
        "entity_count": int(driven.shape[1]),
        "query_count": int(len(indices)),
        "time_step_s": time_step_s,
        "entity_identity_sha256": array_sha256(driven[0]),
        "topology_sha256": topology_sha256,
        "material_model": material_model,
        "parameterization": dict(parameterization or {}),
        "information_boundary": boundary,
        "raw_rollout_sha256": file_sha256(raw_path),
    }
    candidate = {**identity, "runtime_id": content_id(identity)}
    return validate_external_physics_runtime_manifest(
        candidate,
        raw_rollout_path=raw_path,
    )


def write_external_physics_runtime_manifest(
    *,
    output_path: str | Path,
    raw_rollout_path: str | Path,
    profile: PhysicsBackendProfileV1,
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
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build, atomically publish, reload, and validate a runtime manifest."""

    runtime = build_external_physics_runtime_manifest(
        raw_rollout_path=raw_rollout_path,
        profile=profile,
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
    destination = Path(output_path)
    write_atomic_json(runtime, destination, overwrite=overwrite)
    return validate_external_physics_runtime_manifest(
        load_strict_json_object(destination, label="external runtime manifest"),
        raw_rollout_path=raw_rollout_path,
    )


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _validate_file_record(
    value: object,
    *,
    root: Path,
    expected_path: str,
    name: str,
) -> Path:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    _require(record.get("path") == expected_path, f"{name} path changed")
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    byte_count = _positive_integer(
        record.get("byte_count"), name=f"{name}.byte_count"
    )
    path = _ordinary_file(root / expected_path, name=name)
    _require(path.stat().st_size == byte_count, f"{name} byte count changed")
    _require(file_sha256(path) == digest, f"{name} SHA-256 changed")
    return path


def materialize_external_physics_backend(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish one self-contained external physics backend bundle."""

    raw_source, raw = load_external_entity_rollout(raw_rollout_path)
    runtime_source = _ordinary_file(runtime_manifest_path, name="runtime manifest")
    runtime = validate_external_physics_runtime_manifest(
        load_strict_json_object(runtime_source, label="external runtime manifest"),
        raw_rollout_path=raw_source,
    )
    physical = physical_rollout_from_external_entities(raw)
    driven = raw["driven_entity_positions_m"]
    indices = np.asarray(raw["query_entity_indices"], dtype=np.int64)

    output = Path(output_dir).absolute()
    _require(not output.exists(), "output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        provenance = staging / "provenance"
        provenance.mkdir()
        raw_target = provenance / RAW_ARCHIVE_FILENAME
        runtime_target = provenance / RUNTIME_FILENAME
        shutil.copyfile(raw_source, raw_target)
        shutil.copyfile(runtime_source, runtime_target)
        physical_target = staging / PHYSICAL_ARCHIVE_FILENAME
        write_deterministic_npz(physical_target, physical)

        inputs = {
            "raw_rollout": _file_record(raw_target, relative_to=staging),
            "runtime_manifest": _file_record(runtime_target, relative_to=staging),
        }
        output_record = _file_record(physical_target, relative_to=staging)
        mapping = {
            "frame_count": int(driven.shape[0]),
            "entity_count": int(driven.shape[1]),
            "query_count": int(len(indices)),
            "persistent_entity_identity_preserved": True,
            "entity_identity_sha256": array_sha256(driven[0]),
            "query_indices_sha256": array_sha256(indices),
            "coordinate_frame": runtime["coordinate_frame"],
            "position_units": "m",
        }
        identity = {
            "schema": EXTERNAL_PHYSICS_ARTIFACT_SCHEMA,
            "schema_version": EXTERNAL_PHYSICS_SCHEMA_VERSION,
            "backend_profile": runtime["backend_profile"],
            "runtime_id": runtime["runtime_id"],
            "inputs": inputs,
            "output": output_record,
            "mapping": mapping,
            "information_boundary": runtime["information_boundary"],
            "claim_boundary": EXTERNAL_PHYSICS_CLAIM_BOUNDARY,
        }
        artifact = {**identity, "artifact_id": content_id(identity)}
        artifact_path = staging / ARTIFACT_FILENAME
        write_atomic_json(artifact, artifact_path, overwrite=False)
        checksum_paths = [artifact_path, physical_target, raw_target, runtime_target]
        checksum_text = "".join(
            f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}\n"
            for path in sorted(checksum_paths, key=lambda item: item.as_posix())
        )
        (staging / CHECKSUMS_FILENAME).write_text(checksum_text, encoding="ascii")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_external_physics_backend(output)


def validate_external_physics_backend(output_dir: str | Path) -> dict[str, Any]:
    """Validate bundle custody and rederive every physical rollout member."""

    requested_root = Path(output_dir).absolute()
    _require(
        requested_root.is_dir()
        and not requested_root.is_symlink()
        and not any(parent.is_symlink() for parent in requested_root.parents),
        "backend bundle is not an ordinary non-symlink directory",
    )
    root = requested_root.resolve(strict=True)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = {
        ARTIFACT_FILENAME,
        CHECKSUMS_FILENAME,
        PHYSICAL_ARCHIVE_FILENAME,
        f"provenance/{RAW_ARCHIVE_FILENAME}",
        f"provenance/{RUNTIME_FILENAME}",
    }
    _require(actual_files == expected_files, "backend bundle file roster changed")

    artifact = load_strict_json_object(
        root / ARTIFACT_FILENAME,
        label="external physics artifact",
    )
    require_exact_fields(
        artifact,
        expected=_ARTIFACT_FIELDS,
        name="external physics artifact",
    )
    _require(
        artifact.get("schema") == EXTERNAL_PHYSICS_ARTIFACT_SCHEMA,
        "artifact schema changed",
    )
    _require(
        artifact.get("schema_version") == EXTERNAL_PHYSICS_SCHEMA_VERSION,
        "artifact schema version changed",
    )
    profile = profile_from_mapping(
        _mapping(artifact.get("backend_profile"), name="backend_profile")
    )
    inputs = _mapping(artifact.get("inputs"), name="inputs")
    require_exact_fields(inputs, expected=_INPUT_FIELDS, name="inputs")
    raw_path = _validate_file_record(
        inputs.get("raw_rollout"),
        root=root,
        expected_path=f"provenance/{RAW_ARCHIVE_FILENAME}",
        name="raw_rollout",
    )
    runtime_path = _validate_file_record(
        inputs.get("runtime_manifest"),
        root=root,
        expected_path=f"provenance/{RUNTIME_FILENAME}",
        name="runtime_manifest",
    )
    physical_path = _validate_file_record(
        artifact.get("output"),
        root=root,
        expected_path=PHYSICAL_ARCHIVE_FILENAME,
        name="output",
    )
    runtime = validate_external_physics_runtime_manifest(
        load_strict_json_object(runtime_path, label="external runtime manifest"),
        raw_rollout_path=raw_path,
    )
    _require(
        artifact.get("runtime_id") == runtime["runtime_id"],
        "runtime binding changed",
    )
    _require(
        profile.to_dict() == runtime["backend_profile"],
        "backend profile binding changed",
    )

    _, raw = load_external_entity_rollout(raw_path)
    expected = physical_rollout_from_external_entities(raw)
    actual = load_physical_rollout_archive(physical_path)
    for name in sorted(PHYSICAL_ROLLOUT_ARRAY_NAMES):
        _require(
            actual[name].dtype == expected[name].dtype
            and actual[name].shape == expected[name].shape
            and actual[name].tobytes(order="C") == expected[name].tobytes(order="C"),
            f"physical rollout member {name} changed",
        )

    mapping = _mapping(artifact.get("mapping"), name="mapping")
    require_exact_fields(mapping, expected=_MAPPING_FIELDS, name="mapping")
    driven = raw["driven_entity_positions_m"]
    indices = np.asarray(raw["query_entity_indices"], dtype=np.int64)
    expected_mapping = {
        "frame_count": int(driven.shape[0]),
        "entity_count": int(driven.shape[1]),
        "query_count": int(len(indices)),
        "persistent_entity_identity_preserved": True,
        "entity_identity_sha256": array_sha256(driven[0]),
        "query_indices_sha256": array_sha256(indices),
        "coordinate_frame": runtime["coordinate_frame"],
        "position_units": "m",
    }
    _require(dict(mapping) == expected_mapping, "entity-query mapping changed")
    _require(
        artifact.get("information_boundary") == runtime["information_boundary"],
        "information boundary changed",
    )
    _require(
        artifact.get("claim_boundary") == EXTERNAL_PHYSICS_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    identity = {key: item for key, item in artifact.items() if key != "artifact_id"}
    _require(
        artifact.get("artifact_id") == content_id(identity),
        "artifact identity changed",
    )

    checksum_lines = (
        (root / CHECKSUMS_FILENAME).read_text(encoding="ascii").splitlines()
    )
    expected_lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(
            (root / ARTIFACT_FILENAME, physical_path, raw_path, runtime_path),
            key=lambda item: item.as_posix(),
        )
    ]
    _require(checksum_lines == expected_lines, "SHA256SUMS changed")
    return cast(dict[str, Any], plain_json(artifact))


__all__ = [
    "ARTIFACT_FILENAME",
    "CHECKSUMS_FILENAME",
    "EXTERNAL_PHYSICS_ARTIFACT_SCHEMA",
    "EXTERNAL_PHYSICS_CLAIM_BOUNDARY",
    "EXTERNAL_PHYSICS_RAW_ARRAY_NAMES",
    "EXTERNAL_PHYSICS_RUNTIME_SCHEMA",
    "EXTERNAL_PHYSICS_SCHEMA_VERSION",
    "PHYSICAL_ARCHIVE_FILENAME",
    "RAW_ARCHIVE_FILENAME",
    "RUNTIME_FILENAME",
    "array_sha256",
    "build_external_physics_runtime_manifest",
    "file_sha256",
    "load_external_entity_rollout",
    "materialize_external_physics_backend",
    "physical_rollout_from_external_entities",
    "validate_external_physics_backend",
    "validate_external_physics_runtime_manifest",
    "write_external_physics_runtime_manifest",
]
