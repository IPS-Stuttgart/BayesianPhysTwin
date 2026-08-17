"""Deterministic publication and verification for Lagrangian backend exports."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._lagrangian_common_v1 import (
    _ARTIFACT_FIELDS,
    _FILE_FIELDS,
    _INPUT_FIELDS,
    _MAPPING_FIELDS,
    _PROVENANCE_ROSTER,
    _ROOT_ROSTER,
    ARTIFACT_FILENAME,
    CHECKSUMS_FILENAME,
    LAGRANGIAN_ARTIFACT_SCHEMA,
    LAGRANGIAN_BACKEND_CLAIM_BOUNDARY,
    LAGRANGIAN_BACKEND_PROFILES,
    LAGRANGIAN_SCHEMA_VERSION,
    PHYSICAL_ARCHIVE_FILENAME,
    RAW_ARCHIVE_FILENAME,
    RUNTIME_FILENAME,
    _mapping,
    _ordinary_file,
    _positive_integer,
    _require,
    array_sha256,
    file_sha256,
)
from ._lagrangian_runtime_v1 import (
    _validate_information_boundary,
    load_lagrangian_rollout,
    physical_rollout_from_lagrangian_points,
    validate_lagrangian_runtime_manifest,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .physical_rollout_v1 import (
    load_physical_rollout_archive,
    write_deterministic_npz,
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
    digest = sha256_digest(record.get("sha256"), name=f"{name} sha256")
    byte_count = _positive_integer(record.get("byte_count"), name=f"{name} byte_count")
    path = _ordinary_file(root / expected_path, name=name)
    _require(path.stat().st_size == byte_count, f"{name} byte count changed")
    _require(file_sha256(path) == digest, f"{name} SHA-256 changed")
    return path


def _write_checksums(root: Path, paths: list[Path]) -> Path:
    target = root / CHECKSUMS_FILENAME
    lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return target


def _validate_checksums(root: Path, paths: list[Path]) -> None:
    checksum_path = _ordinary_file(root / CHECKSUMS_FILENAME, name="checksum manifest")
    expected = "".join(
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    )
    try:
        actual = checksum_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read checksum manifest") from error
    _require(actual == expected, "checksum manifest changed")


def _validate_runtime_against_arrays(
    runtime: Mapping[str, Any], arrays: Mapping[str, npt.NDArray[Any]]
) -> None:
    driven = arrays["driven_point_positions_m"]
    indices = arrays["material_query_indices"]
    _require(
        runtime.get("frame_count") == driven.shape[0],
        "runtime frame_count changed",
    )
    _require(
        runtime.get("point_count") == driven.shape[1],
        "runtime point_count changed",
    )
    _require(runtime.get("query_count") == len(indices), "runtime query_count changed")
    expected_precision = "float32" if driven.dtype == np.dtype("float32") else "float64"
    metadata = _mapping(runtime.get("backend_metadata"), name="backend_metadata")
    _require(
        metadata.get("precision") == expected_precision,
        "runtime precision differs from rollout dtype",
    )


def _build_artifact(
    *,
    root: Path,
    runtime: Mapping[str, Any],
    raw_path: Path,
    runtime_path: Path,
    physical_path: Path,
    arrays: Mapping[str, npt.NDArray[Any]],
) -> dict[str, Any]:
    inputs = {
        "raw_rollout": _file_record(raw_path, relative_to=root),
        "runtime_manifest": _file_record(runtime_path, relative_to=root),
    }
    output = _file_record(physical_path, relative_to=root)
    driven = arrays["driven_point_positions_m"]
    indices = arrays["material_query_indices"]
    mapping = {
        "frame_count": int(driven.shape[0]),
        "point_count": int(driven.shape[1]),
        "query_count": int(len(indices)),
        "identity_kind": runtime["identity_kind"],
        "material_identity_preserved": True,
        "query_indices_sha256": array_sha256(indices),
        "coordinate_frame": runtime["coordinate_frame"],
        "position_units": runtime["position_units"],
        "step_axis": runtime["step_axis"],
        "step_units": runtime["step_units"],
    }
    identity = {
        "schema": LAGRANGIAN_ARTIFACT_SCHEMA,
        "schema_version": LAGRANGIAN_SCHEMA_VERSION,
        "backend_profile": runtime["backend_profile"],
        "runtime_id": runtime["runtime_id"],
        "inputs": inputs,
        "output": output,
        "mapping": mapping,
        "information_boundary": runtime["information_boundary"],
        "claim_boundary": LAGRANGIAN_BACKEND_CLAIM_BOUNDARY,
    }
    return {**identity, "artifact_id": content_id(identity)}


def materialize_lagrangian_backend(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish one deterministic portable bundle from an external solver export."""

    source_raw, arrays = load_lagrangian_rollout(raw_rollout_path)
    source_runtime = _ordinary_file(runtime_manifest_path, name="runtime manifest")
    runtime = validate_lagrangian_runtime_manifest(
        load_strict_json_object(source_runtime, label="runtime manifest"),
        raw_rollout_path=source_raw,
    )
    _validate_runtime_against_arrays(runtime, arrays)

    output = Path(output_dir).absolute()
    _require(
        not output.exists() and not output.is_symlink(),
        "output directory already exists",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging.",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary) / output.name
        provenance = staging / "provenance"
        provenance.mkdir(parents=True)
        raw_copy = provenance / RAW_ARCHIVE_FILENAME
        runtime_copy = provenance / RUNTIME_FILENAME
        shutil.copyfile(source_raw, raw_copy)
        shutil.copyfile(source_runtime, runtime_copy)
        physical_path = write_deterministic_npz(
            staging / PHYSICAL_ARCHIVE_FILENAME,
            physical_rollout_from_lagrangian_points(arrays),
        )
        artifact = _build_artifact(
            root=staging,
            runtime=runtime,
            raw_path=raw_copy,
            runtime_path=runtime_copy,
            physical_path=physical_path,
            arrays=arrays,
        )
        artifact_path = staging / ARTIFACT_FILENAME
        write_atomic_json(artifact, artifact_path, overwrite=False)
        _write_checksums(
            staging,
            [artifact_path, physical_path, raw_copy, runtime_copy],
        )
        validate_lagrangian_backend(staging)
        os.replace(staging, output)
    return validate_lagrangian_backend(output)


def validate_lagrangian_backend(output_dir: str | Path) -> dict[str, Any]:
    """Validate an already published external-backend bundle fail closed."""

    root = Path(output_dir).absolute()
    _require(
        root.is_dir() and not root.is_symlink(),
        "backend root must be an ordinary directory",
    )
    _require(
        {path.name for path in root.iterdir()} == set(_ROOT_ROSTER),
        "backend root roster changed",
    )
    provenance = root / "provenance"
    _require(
        provenance.is_dir()
        and not provenance.is_symlink()
        and {path.name for path in provenance.iterdir()} == set(_PROVENANCE_ROSTER),
        "backend provenance roster changed",
    )

    artifact_path = _ordinary_file(root / ARTIFACT_FILENAME, name="backend artifact")
    artifact_raw = load_strict_json_object(artifact_path, label="backend artifact")
    require_exact_fields(
        artifact_raw,
        expected=_ARTIFACT_FIELDS,
        name="backend artifact",
    )
    _require(
        artifact_raw.get("schema") == LAGRANGIAN_ARTIFACT_SCHEMA,
        "artifact schema changed",
    )
    _require(
        artifact_raw.get("schema_version") == LAGRANGIAN_SCHEMA_VERSION,
        "artifact schema version changed",
    )
    profile = nonempty_string(
        artifact_raw.get("backend_profile"), name="backend_profile"
    )
    _require(profile in LAGRANGIAN_BACKEND_PROFILES, "unknown backend_profile")
    sha256_digest(artifact_raw.get("runtime_id"), name="runtime_id")
    _require(
        artifact_raw.get("claim_boundary") == LAGRANGIAN_BACKEND_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    boundary = _validate_information_boundary(artifact_raw.get("information_boundary"))

    inputs = _mapping(artifact_raw.get("inputs"), name="inputs")
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
        artifact_raw.get("output"),
        root=root,
        expected_path=PHYSICAL_ARCHIVE_FILENAME,
        name="output",
    )

    _, arrays = load_lagrangian_rollout(raw_path)
    runtime = validate_lagrangian_runtime_manifest(
        load_strict_json_object(runtime_path, label="runtime manifest"),
        raw_rollout_path=raw_path,
    )
    _validate_runtime_against_arrays(runtime, arrays)
    _require(runtime["backend_profile"] == profile, "artifact backend profile changed")
    _require(
        runtime["runtime_id"] == artifact_raw["runtime_id"],
        "artifact runtime changed",
    )
    _require(
        runtime["information_boundary"] == boundary,
        "information boundary changed",
    )

    physical = load_physical_rollout_archive(
        physical_path,
        expected_frame_count=int(runtime["frame_count"]),
    )
    expected_physical = physical_rollout_from_lagrangian_points(arrays)
    for name, expected in expected_physical.items():
        _require(
            np.array_equal(physical[name], expected),
            f"physical array {name} changed",
        )

    mapping = _mapping(artifact_raw.get("mapping"), name="mapping")
    require_exact_fields(mapping, expected=_MAPPING_FIELDS, name="mapping")
    expected_mapping = {
        "frame_count": int(arrays["driven_point_positions_m"].shape[0]),
        "point_count": int(arrays["driven_point_positions_m"].shape[1]),
        "query_count": int(len(arrays["material_query_indices"])),
        "identity_kind": runtime["identity_kind"],
        "material_identity_preserved": True,
        "query_indices_sha256": array_sha256(arrays["material_query_indices"]),
        "coordinate_frame": runtime["coordinate_frame"],
        "position_units": runtime["position_units"],
        "step_axis": runtime["step_axis"],
        "step_units": runtime["step_units"],
    }
    _require(dict(mapping) == expected_mapping, "physical mapping changed")

    identity = {key: item for key, item in artifact_raw.items() if key != "artifact_id"}
    _require(
        artifact_raw.get("artifact_id") == content_id(identity),
        "artifact identity changed",
    )
    _validate_checksums(root, [artifact_path, physical_path, raw_path, runtime_path])
    return cast(dict[str, Any], plain_json(artifact_raw))
