"""Adapt fixed-identity Genesis MPM particles to physical rollout v1."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .physical_rollout_v1 import (
    PHYSICAL_ROLLOUT_ARRAY_NAMES,
    load_physical_rollout_archive,
    validate_physical_rollout_arrays,
    write_deterministic_npz,
)

GENESIS_MPM_RUNTIME_SCHEMA: Final = "bayesian-phystwin.genesis-mpm-runtime"
GENESIS_MPM_ARTIFACT_SCHEMA: Final = "bayesian-phystwin.genesis-mpm-backend"
GENESIS_MPM_SCHEMA_VERSION: Final = 1
GENESIS_MPM_BACKEND_KIND: Final = "genesis-elastic-mpm-v1"
GENESIS_MPM_ENGINE_REPOSITORY: Final = "Genesis-Embodied-AI/Genesis"
GENESIS_MPM_RAW_ARRAY_NAMES: Final = frozenset(
    {
        "driven_particle_positions_m",
        "zero_action_particle_positions_m",
        "material_query_indices",
        "action_support",
    }
)
PHYSICAL_ARCHIVE_FILENAME: Final = "physical-prediction.npz"
RAW_ARCHIVE_FILENAME: Final = "genesis-particle-rollout.npz"
RUNTIME_FILENAME: Final = "genesis-runtime.json"
ARTIFACT_FILENAME: Final = "genesis-mpm-backend.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

GENESIS_MPM_CLAIM_BOUNDARY: Final = (
    "A synthetic fixed-material-identity compatibility smoke for the Genesis "
    "elastic MPM solver with compliant rigid attachments. It validates the "
    "engine-to-belief artifact path; it does not reproduce PhysWorld or "
    "DeformMaster, establish real-data transfer, calibrate an MPM twin, or "
    "support a state-of-the-art claim."
)

_RUNTIME_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_kind",
        "engine_repository",
        "engine_version",
        "torch_version",
        "python_version",
        "device",
        "device_name",
        "coordinate_frame",
        "position_units",
        "time_units",
        "frame_count",
        "particle_count",
        "query_count",
        "time_step_s",
        "simulation",
        "diagnostics",
        "implementation",
        "information_boundary",
        "raw_rollout_sha256",
        "runtime_id",
    }
)
_SIMULATION_FIELDS: Final = frozenset(
    {
        "scene",
        "beam_extents_m",
        "action_displacement_m",
        "gravity_m_s2",
        "density_kg_m3",
        "young_modulus_pa",
        "poisson_ratio",
        "elastic_model",
        "grid_density",
        "substeps",
        "attachment_stiffness",
        "solver",
    }
)
_DIAGNOSTIC_FIELDS: Final = frozenset(
    {
        "maximum_action_response_m",
        "maximum_particle_step_m",
        "response_to_action_ratio",
        "stability_cap_ratio",
        "stability_gate_passed",
    }
)
_IMPLEMENTATION_FIELDS: Final = frozenset(
    {"repository", "revision", "source_files_sha256"}
)
_IMPLEMENTATION_SOURCE_PATHS: Final = frozenset(
    {
        "src/bayesian_phystwin/_genesis_mpm_runtime.py",
        "src/bayesian_phystwin/genesis_mpm_backend_v1.py",
        "src/bayesian_phystwin/physical_rollout_v1.py",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "synthetic_scene",
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
        "backend_kind",
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
        "particle_count",
        "query_count",
        "material_identity_preserved",
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


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _finite_vector(value: object, *, name: str) -> list[float]:
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


def validate_genesis_mpm_runtime_manifest(
    value: Mapping[str, Any],
    *,
    raw_rollout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a content-addressed Genesis runtime record."""

    require_exact_fields(value, expected=_RUNTIME_FIELDS, name="runtime manifest")
    _require(
        value.get("schema") == GENESIS_MPM_RUNTIME_SCHEMA, "runtime schema changed"
    )
    _require(
        value.get("schema_version") == GENESIS_MPM_SCHEMA_VERSION,
        "runtime schema version changed",
    )
    _require(
        value.get("backend_kind") == GENESIS_MPM_BACKEND_KIND,
        "runtime backend kind changed",
    )
    _require(
        value.get("engine_repository") == GENESIS_MPM_ENGINE_REPOSITORY,
        "runtime engine repository changed",
    )
    simulation_raw = value.get("simulation")
    if not isinstance(simulation_raw, Mapping):
        raise ValueError("simulation must be a JSON object")
    simulation = cast(Mapping[str, Any], simulation_raw)
    require_exact_fields(simulation, expected=_SIMULATION_FIELDS, name="simulation")
    diagnostics_raw = value.get("diagnostics")
    if not isinstance(diagnostics_raw, Mapping):
        raise ValueError("diagnostics must be a JSON object")
    diagnostics = cast(Mapping[str, Any], diagnostics_raw)
    require_exact_fields(diagnostics, expected=_DIAGNOSTIC_FIELDS, name="diagnostics")
    implementation_raw = value.get("implementation")
    if not isinstance(implementation_raw, Mapping):
        raise ValueError("implementation must be a JSON object")
    implementation = cast(Mapping[str, Any], implementation_raw)
    require_exact_fields(
        implementation, expected=_IMPLEMENTATION_FIELDS, name="implementation"
    )
    _require(
        implementation.get("repository") == "IPS-Stuttgart/BayesianPhysTwin",
        "implementation repository changed",
    )
    revision = nonempty_string(
        implementation.get("revision"), name="implementation.revision"
    )
    _require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "implementation revision must be a lowercase Git SHA-1",
    )
    source_hashes_raw = implementation.get("source_files_sha256")
    if not isinstance(source_hashes_raw, Mapping):
        raise ValueError("implementation.source_files_sha256 must be a JSON object")
    source_hashes = cast(Mapping[str, Any], source_hashes_raw)
    _require(
        set(source_hashes) == set(_IMPLEMENTATION_SOURCE_PATHS),
        "implementation source-file roster changed",
    )
    for path, digest in source_hashes.items():
        sha256_digest(digest, name=f"implementation.source_files_sha256.{path}")
    boundary_raw = value.get("information_boundary")
    if not isinstance(boundary_raw, Mapping):
        raise ValueError("information_boundary must be a JSON object")
    boundary = cast(Mapping[str, Any], boundary_raw)
    require_exact_fields(
        boundary, expected=_BOUNDARY_FIELDS, name="information_boundary"
    )
    expected_boundary = {
        "synthetic_scene": True,
        "dataset_payload_read": False,
        "future_observations_read": False,
        "outcomes_read": False,
        "known_action_used": True,
    }
    _require(
        dict(boundary) == expected_boundary, "runtime information boundary changed"
    )
    frame_count = _positive_integer(value.get("frame_count"), name="frame_count")
    _require(frame_count >= 2, "frame_count must be at least two")
    _positive_integer(value.get("particle_count"), name="particle_count")
    _positive_integer(value.get("query_count"), name="query_count")
    _finite_positive(value.get("time_step_s"), name="time_step_s")
    for name in (
        "engine_version",
        "torch_version",
        "python_version",
        "device",
        "device_name",
    ):
        nonempty_string(value.get(name), name=name)
    _require(
        value.get("coordinate_frame") == "right-handed-z-up-world-v1",
        "coordinate frame changed",
    )
    _require(value.get("position_units") == "m", "position units must be metres")
    _require(value.get("time_units") == "s", "time units must be seconds")
    _require(
        simulation.get("scene") == "compliant-gripper-beam-bend-v1",
        "scene changed",
    )
    beam_extents = _finite_vector(
        simulation.get("beam_extents_m"), name="beam_extents_m"
    )
    _require(all(item > 0.0 for item in beam_extents), "beam extents must be positive")
    action = _finite_vector(
        simulation.get("action_displacement_m"), name="action_displacement_m"
    )
    _require(np.linalg.norm(action) > 0.0, "action displacement must be nonzero")
    _finite_vector(simulation.get("gravity_m_s2"), name="gravity_m_s2")
    for name in (
        "density_kg_m3",
        "young_modulus_pa",
        "attachment_stiffness",
    ):
        _finite_positive(simulation.get(name), name=name)
    poisson = simulation.get("poisson_ratio")
    _require(
        isinstance(poisson, (int, float))
        and not isinstance(poisson, bool)
        and np.isfinite(float(poisson))
        and 0.0 <= float(poisson) < 0.5,
        "poisson_ratio must be finite and in [0,0.5)",
    )
    _positive_integer(simulation.get("grid_density"), name="grid_density")
    _positive_integer(simulation.get("substeps"), name="substeps")
    _require(
        simulation.get("elastic_model") in {"corotation", "neohooken"},
        "elastic_model changed",
    )
    _require(simulation.get("solver") == "genesis-mpm", "solver changed")
    maximum_response = _finite_positive(
        diagnostics.get("maximum_action_response_m"),
        name="maximum_action_response_m",
    )
    maximum_step = _finite_positive(
        diagnostics.get("maximum_particle_step_m"),
        name="maximum_particle_step_m",
    )
    response_ratio = _finite_positive(
        diagnostics.get("response_to_action_ratio"),
        name="response_to_action_ratio",
    )
    stability_cap = _finite_positive(
        diagnostics.get("stability_cap_ratio"),
        name="stability_cap_ratio",
    )
    _require(
        diagnostics.get("stability_gate_passed") is True,
        "stability gate did not pass",
    )
    action_norm = float(np.linalg.norm(action))
    _require(
        response_ratio == maximum_response / action_norm,
        "response-to-action ratio changed",
    )
    _require(response_ratio <= stability_cap, "response exceeds stability cap")
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
        _, raw = load_genesis_particle_rollout(raw_path)
        driven = raw["driven_particle_positions_m"]
        zero = raw["zero_action_particle_positions_m"]
        measured_response = float(np.max(np.linalg.norm(driven - zero, axis=2)))
        measured_step = float(np.max(np.linalg.norm(np.diff(driven, axis=0), axis=2)))
        _require(measured_response == maximum_response, "response diagnostic changed")
        _require(measured_step == maximum_step, "particle-step diagnostic changed")
    return cast(dict[str, Any], plain_json(value))


def load_genesis_particle_rollout(
    path: str | Path,
) -> tuple[Path, dict[str, npt.NDArray[Any]]]:
    """Load a fixed-identity Genesis particle rollout without pickle."""

    source = _ordinary_file(path, name="raw rollout")
    try:
        with np.load(source, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(GENESIS_MPM_RAW_ARRAY_NAMES),
                "raw Genesis array roster changed",
            )
            arrays = {
                name: np.ascontiguousarray(np.asarray(stored[name])).copy()
                for name in stored.files
            }
    except (OSError, ValueError) as error:
        raise ValueError("cannot load raw Genesis rollout") from error

    driven = arrays["driven_particle_positions_m"]
    zero = arrays["zero_action_particle_positions_m"]
    indices = arrays["material_query_indices"]
    support = arrays["action_support"]
    _require(
        driven.ndim == 3
        and driven.shape[0] >= 2
        and driven.shape[1] >= 1
        and driven.shape[2] == 3,
        "Genesis particle positions must have shape (T,P,3)",
    )
    _require(zero.shape == driven.shape, "driven and zero-action shapes differ")
    _require(
        np.issubdtype(driven.dtype, np.floating)
        and np.issubdtype(zero.dtype, np.floating)
        and driven.dtype == zero.dtype,
        "Genesis particle positions must share a floating dtype",
    )
    _require(
        np.all(np.isfinite(driven)) and np.all(np.isfinite(zero)),
        "Genesis particle positions are non-finite",
    )
    _require(
        np.array_equal(driven[0], zero[0]), "Genesis rollouts differ at frame zero"
    )
    _require(
        indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer),
        "material_query_indices must be an integer vector",
    )
    _require(len(indices) >= 1, "material_query_indices must not be empty")
    _require(
        len(np.unique(indices)) == len(indices), "material_query_indices must be unique"
    )
    _require(
        np.all((indices >= 0) & (indices < driven.shape[1])),
        "material query index exceeds particle count",
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


def physical_rollout_from_genesis_particles(
    arrays: Mapping[str, npt.NDArray[Any]],
) -> dict[str, FloatArray]:
    """Project persistent MPM particle identities into the portable contract."""

    driven = np.asarray(arrays["driven_particle_positions_m"])
    zero = np.asarray(arrays["zero_action_particle_positions_m"])
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
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    record = cast(Mapping[str, Any], value)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    _require(record.get("path") == expected_path, f"{name} path changed")
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    byte_count = _positive_integer(record.get("byte_count"), name=f"{name}.byte_count")
    path = _ordinary_file(root / expected_path, name=name)
    _require(path.stat().st_size == byte_count, f"{name} byte count changed")
    _require(file_sha256(path) == digest, f"{name} SHA-256 changed")
    return path


def materialize_genesis_mpm_backend(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish one self-contained Genesis MPM compatibility bundle."""

    raw_source, raw = load_genesis_particle_rollout(raw_rollout_path)
    runtime_source = _ordinary_file(runtime_manifest_path, name="runtime manifest")
    runtime = validate_genesis_mpm_runtime_manifest(
        load_strict_json_object(runtime_source, label="Genesis runtime manifest"),
        raw_rollout_path=raw_source,
    )
    driven = raw["driven_particle_positions_m"]
    indices = np.asarray(raw["material_query_indices"], dtype=np.int64)
    _require(runtime["frame_count"] == driven.shape[0], "runtime frame count differs")
    _require(
        runtime["particle_count"] == driven.shape[1], "runtime particle count differs"
    )
    _require(runtime["query_count"] == len(indices), "runtime query count differs")
    physical = physical_rollout_from_genesis_particles(raw)

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
        mapping = {
            "frame_count": int(driven.shape[0]),
            "particle_count": int(driven.shape[1]),
            "query_count": int(len(indices)),
            "material_identity_preserved": True,
            "query_indices_sha256": array_sha256(indices),
            "coordinate_frame": runtime["coordinate_frame"],
            "position_units": "m",
        }
        identity = {
            "schema": GENESIS_MPM_ARTIFACT_SCHEMA,
            "schema_version": GENESIS_MPM_SCHEMA_VERSION,
            "backend_kind": GENESIS_MPM_BACKEND_KIND,
            "runtime_id": runtime["runtime_id"],
            "inputs": inputs,
            "output": _file_record(physical_target, relative_to=staging),
            "mapping": mapping,
            "information_boundary": runtime["information_boundary"],
            "claim_boundary": GENESIS_MPM_CLAIM_BOUNDARY,
        }
        artifact = {**identity, "artifact_id": content_id(identity)}
        artifact_path = staging / ARTIFACT_FILENAME
        write_atomic_json(artifact, artifact_path, overwrite=False)
        checksum_paths = [artifact_path, physical_target, raw_target, runtime_target]
        checksums = "".join(
            f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}\n"
            for path in sorted(checksum_paths, key=lambda item: item.as_posix())
        )
        (staging / CHECKSUMS_FILENAME).write_text(checksums, encoding="ascii")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_genesis_mpm_backend(output)


def validate_genesis_mpm_backend(output_dir: str | Path) -> dict[str, Any]:
    """Validate bundle custody and rederive its physical rollout exactly."""

    requested_root = Path(output_dir).absolute()
    _require(
        requested_root.is_dir()
        and not requested_root.is_symlink()
        and not any(parent.is_symlink() for parent in requested_root.parents),
        "backend bundle is not an ordinary non-symlink directory",
    )
    root = requested_root.resolve(strict=True)
    expected_roster = {
        ARTIFACT_FILENAME,
        CHECKSUMS_FILENAME,
        PHYSICAL_ARCHIVE_FILENAME,
        f"provenance/{RAW_ARCHIVE_FILENAME}",
        f"provenance/{RUNTIME_FILENAME}",
    }
    actual_roster = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(actual_roster == expected_roster, "backend bundle file roster changed")
    artifact = load_strict_json_object(
        root / ARTIFACT_FILENAME, label="Genesis artifact"
    )
    require_exact_fields(artifact, expected=_ARTIFACT_FIELDS, name="Genesis artifact")
    _require(
        artifact.get("schema") == GENESIS_MPM_ARTIFACT_SCHEMA,
        "artifact schema changed",
    )
    _require(artifact.get("schema_version") == 1, "artifact schema version changed")
    _require(
        artifact.get("backend_kind") == GENESIS_MPM_BACKEND_KIND, "backend kind changed"
    )
    inputs_raw = artifact.get("inputs")
    if not isinstance(inputs_raw, Mapping):
        raise ValueError("inputs must be a JSON object")
    inputs = cast(Mapping[str, Any], inputs_raw)
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
    runtime = validate_genesis_mpm_runtime_manifest(
        load_strict_json_object(runtime_path, label="Genesis runtime manifest"),
        raw_rollout_path=raw_path,
    )
    _require(
        artifact.get("runtime_id") == runtime["runtime_id"], "runtime binding changed"
    )
    _, raw = load_genesis_particle_rollout(raw_path)
    expected = physical_rollout_from_genesis_particles(raw)
    actual = load_physical_rollout_archive(physical_path)
    for name in sorted(PHYSICAL_ROLLOUT_ARRAY_NAMES):
        _require(
            actual[name].dtype == expected[name].dtype
            and actual[name].shape == expected[name].shape
            and actual[name].tobytes(order="C") == expected[name].tobytes(order="C"),
            f"physical rollout member {name} changed",
        )
    mapping_raw = artifact.get("mapping")
    if not isinstance(mapping_raw, Mapping):
        raise ValueError("mapping must be a JSON object")
    mapping = cast(Mapping[str, Any], mapping_raw)
    require_exact_fields(mapping, expected=_MAPPING_FIELDS, name="mapping")
    indices = np.asarray(raw["material_query_indices"], dtype=np.int64)
    expected_mapping = {
        "frame_count": int(raw["driven_particle_positions_m"].shape[0]),
        "particle_count": int(raw["driven_particle_positions_m"].shape[1]),
        "query_count": int(len(indices)),
        "material_identity_preserved": True,
        "query_indices_sha256": array_sha256(indices),
        "coordinate_frame": runtime["coordinate_frame"],
        "position_units": "m",
    }
    _require(dict(mapping) == expected_mapping, "material-query mapping changed")
    _require(
        artifact.get("information_boundary") == runtime["information_boundary"],
        "information boundary changed",
    )
    _require(
        artifact.get("claim_boundary") == GENESIS_MPM_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    identity = {key: item for key, item in artifact.items() if key != "artifact_id"}
    _require(
        artifact.get("artifact_id") == content_id(identity), "artifact identity changed"
    )
    expected_lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(
            (root / ARTIFACT_FILENAME, physical_path, raw_path, runtime_path),
            key=lambda item: item.as_posix(),
        )
    ]
    checksum_lines = (
        (root / CHECKSUMS_FILENAME).read_text(encoding="ascii").splitlines()
    )
    _require(checksum_lines == expected_lines, "SHA256SUMS changed")
    return cast(dict[str, Any], plain_json(artifact))
