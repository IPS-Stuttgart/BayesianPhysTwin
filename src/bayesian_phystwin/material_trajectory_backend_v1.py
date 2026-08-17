"""Publish external deformable simulators through one physical contract."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import plain_json
from ._material_trajectory_contract_v1 import (
    CANONICAL_COORDINATE_FRAME,
    MATERIAL_BACKEND_PROFILES,
    MATERIAL_TRAJECTORY_RAW_ARRAY_NAMES,
    MATERIAL_TRAJECTORY_RUNTIME_SCHEMA,
    MATERIAL_TRAJECTORY_SCHEMA_VERSION,
    MaterialBackendProfile,
    _mapping,
    _ordinary_file,
    _positive_integer,
    _require,
    array_sha256,
    file_sha256,
    get_material_backend_profile,
    load_material_trajectory_rollout,
    material_backend_profile_records,
    physical_rollout_from_material_trajectory,
    validate_material_runtime_manifest,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .physical_rollout_v1 import (
    PHYSICAL_ROLLOUT_ARRAY_NAMES,
    load_physical_rollout_archive,
    write_deterministic_npz,
)

MATERIAL_TRAJECTORY_ARTIFACT_SCHEMA: Final = (
    "bayesian-phystwin.material-trajectory-backend"
)
PHYSICAL_ARCHIVE_FILENAME: Final = "physical-prediction.npz"
RAW_ARCHIVE_FILENAME: Final = "material-trajectory-rollout.npz"
RUNTIME_FILENAME: Final = "material-runtime.json"
ARTIFACT_FILENAME: Final = "material-trajectory-backend.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

MATERIAL_TRAJECTORY_CLAIM_BOUNDARY: Final = (
    "A fixed-material-identity compatibility bridge for an external deformable "
    "simulator. It validates simulator-to-belief custody and exact replay into "
    "Bayesian-PhysTwin's portable physical-rollout contract; it does not by "
    "itself establish simulator fidelity, parameter identification, predictive "
    "calibration, real-data transfer, safety, or state of the art."
)

_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_kind",
        "runtime_id",
        "profile",
        "inputs",
        "output",
        "mapping",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_PROFILE_FIELDS: Final = frozenset(
    {"backend_kind", "engine_repository", "solver_family", "identity_kind"}
)
_FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
_INPUT_FIELDS: Final = frozenset({"raw_rollout", "runtime_manifest"})
_MAPPING_FIELDS: Final = frozenset(
    {
        "frame_count",
        "state_count",
        "query_count",
        "material_identity_preserved",
        "identity_kind",
        "solver_family",
        "query_indices_sha256",
        "coordinate_frame",
        "position_units",
    }
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
    byte_count = _positive_integer(record.get("byte_count"), name=f"{name}.byte_count")
    path = _ordinary_file(root / expected_path, name=name)
    _require(path.stat().st_size == byte_count, f"{name} byte count changed")
    _require(file_sha256(path) == digest, f"{name} SHA-256 changed")
    return path


def materialize_material_trajectory_backend(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish one self-contained external material-backend bundle."""

    raw_source, raw = load_material_trajectory_rollout(raw_rollout_path)
    runtime_source = _ordinary_file(runtime_manifest_path, name="runtime manifest")
    runtime = validate_material_runtime_manifest(
        load_strict_json_object(runtime_source, label="material runtime manifest"),
        raw_rollout_path=raw_source,
    )
    driven = raw["driven_material_positions_m"]
    indices = np.asarray(raw["material_query_indices"], dtype=np.int64)
    _require(runtime["frame_count"] == driven.shape[0], "runtime frame count differs")
    _require(runtime["state_count"] == driven.shape[1], "runtime state count differs")
    _require(runtime["query_count"] == len(indices), "runtime query count differs")
    profile = get_material_backend_profile(runtime["backend_kind"])
    physical = physical_rollout_from_material_trajectory(raw)

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
            "state_count": int(driven.shape[1]),
            "query_count": int(len(indices)),
            "material_identity_preserved": True,
            "identity_kind": profile.identity_kind,
            "solver_family": profile.solver_family,
            "query_indices_sha256": array_sha256(indices),
            "coordinate_frame": runtime["coordinate_frame"],
            "position_units": "m",
        }
        identity = {
            "schema": MATERIAL_TRAJECTORY_ARTIFACT_SCHEMA,
            "schema_version": MATERIAL_TRAJECTORY_SCHEMA_VERSION,
            "backend_kind": profile.backend_kind,
            "runtime_id": runtime["runtime_id"],
            "profile": profile.to_dict(),
            "inputs": inputs,
            "output": _file_record(physical_target, relative_to=staging),
            "mapping": mapping,
            "information_boundary": runtime["information_boundary"],
            "claim_boundary": MATERIAL_TRAJECTORY_CLAIM_BOUNDARY,
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
    return validate_material_trajectory_backend(output)


def validate_material_trajectory_backend(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate bundle custody and rederive every physical array exactly."""

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
        root / ARTIFACT_FILENAME, label="material backend artifact"
    )
    require_exact_fields(
        artifact, expected=_ARTIFACT_FIELDS, name="material backend artifact"
    )
    _require(
        artifact.get("schema") == MATERIAL_TRAJECTORY_ARTIFACT_SCHEMA,
        "artifact schema changed",
    )
    _require(
        artifact.get("schema_version") == MATERIAL_TRAJECTORY_SCHEMA_VERSION,
        "artifact schema version changed",
    )
    profile = get_material_backend_profile(artifact.get("backend_kind"))
    profile_record = _mapping(artifact.get("profile"), name="profile")
    require_exact_fields(profile_record, expected=_PROFILE_FIELDS, name="profile")
    _require(dict(profile_record) == profile.to_dict(), "backend profile changed")

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
    runtime = validate_material_runtime_manifest(
        load_strict_json_object(runtime_path, label="material runtime manifest"),
        raw_rollout_path=raw_path,
    )
    _require(
        runtime["backend_kind"] == profile.backend_kind,
        "artifact and runtime backend kinds differ",
    )
    _require(
        artifact.get("runtime_id") == runtime["runtime_id"],
        "runtime binding changed",
    )

    _, raw = load_material_trajectory_rollout(raw_path)
    expected = physical_rollout_from_material_trajectory(raw)
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
    indices = np.asarray(raw["material_query_indices"], dtype=np.int64)
    driven = raw["driven_material_positions_m"]
    expected_mapping = {
        "frame_count": int(driven.shape[0]),
        "state_count": int(driven.shape[1]),
        "query_count": int(len(indices)),
        "material_identity_preserved": True,
        "identity_kind": profile.identity_kind,
        "solver_family": profile.solver_family,
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
        artifact.get("claim_boundary") == MATERIAL_TRAJECTORY_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    identity = {key: item for key, item in artifact.items() if key != "artifact_id"}
    _require(
        artifact.get("artifact_id") == content_id(identity),
        "artifact identity changed",
    )

    checksum_paths = [
        root / ARTIFACT_FILENAME,
        physical_path,
        raw_path,
        runtime_path,
    ]
    expected_lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(checksum_paths, key=lambda item: item.as_posix())
    ]
    try:
        actual_lines = (
            (root / CHECKSUMS_FILENAME).read_text(encoding="ascii").splitlines()
        )
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read backend checksums") from error
    _require(actual_lines == expected_lines, "backend checksums changed")
    return cast(dict[str, Any], plain_json(artifact))


__all__ = [
    "ARTIFACT_FILENAME",
    "CANONICAL_COORDINATE_FRAME",
    "CHECKSUMS_FILENAME",
    "MATERIAL_BACKEND_PROFILES",
    "MATERIAL_TRAJECTORY_ARTIFACT_SCHEMA",
    "MATERIAL_TRAJECTORY_CLAIM_BOUNDARY",
    "MATERIAL_TRAJECTORY_RAW_ARRAY_NAMES",
    "MATERIAL_TRAJECTORY_RUNTIME_SCHEMA",
    "MATERIAL_TRAJECTORY_SCHEMA_VERSION",
    "PHYSICAL_ARCHIVE_FILENAME",
    "RAW_ARCHIVE_FILENAME",
    "RUNTIME_FILENAME",
    "MaterialBackendProfile",
    "array_sha256",
    "file_sha256",
    "get_material_backend_profile",
    "load_material_trajectory_rollout",
    "material_backend_profile_records",
    "materialize_material_trajectory_backend",
    "physical_rollout_from_material_trajectory",
    "validate_material_runtime_manifest",
    "validate_material_trajectory_backend",
]
