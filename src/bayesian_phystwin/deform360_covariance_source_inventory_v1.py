"""Roster-bound, header-only inventory for the frozen covariance source study."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Final, cast

from ._canonical_contracts import plain_json
from ._portable_contracts import content_id, load_strict_json_object

INVENTORY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-source-input-inventory-v1"
)
INVENTORY_SCHEMA_VERSION: Final = 1
SOFTWARE_PROTOCOL_ID: Final = (
    "0f13d7a1f1610588ca9e7119f94814c99940fb31050419de16fa9cae06f683cc"
)
PAPER_PROTOCOL_ID: Final = (
    "fa16c105e6d535d1e229ccf086fd69d05b2be74592b5c4e3f6c5289b8915fee3"
)
CROSSREPO_BINDING_ID: Final = (
    "531123205959a3d3d0549d9256b6ec222dca636198bc1e93f1b468d1a77c8f33"
)
SELECTION_GIT_BLOB_SHA1: Final = "9c1cc1167339a45e3659a9ed6096c1af16f6f62d"
SELECTION_REPOSITORY_PATH: Final = (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
_ARRAY_SUFFIXES: Final = frozenset({".npy", ".npz"})
_JSON_LIMIT_BYTES: Final = 2_000_000
_BOUNDARY: Final = {
    "array_values_read": False,
    "confirmation_root_entered": False,
    "file_payloads_scored": False,
    "source_roots_only": True,
    "source_suffix_used_for_prediction": False,
    "target_outcomes_opened": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(requested.is_dir(), f"{name} must be an existing directory")
    _require(not requested.is_symlink(), f"{name} must not be a symlink")
    resolved = requested.resolve(strict=True)
    _require(resolved == requested, f"{name} must be a canonical absolute path")
    return resolved


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(requested.is_file(), f"{name} must be an existing file")
    _require(not requested.is_symlink(), f"{name} must not be a symlink")
    resolved = requested.resolve(strict=True)
    _require(resolved == requested, f"{name} must be a canonical absolute path")
    return resolved


def _identity(document: Mapping[str, Any], field: str, expected: str) -> None:
    _require(document.get(field) == expected, f"{field} changed")
    body = {key: value for key, value in document.items() if key != field}
    _require(content_id(body) == expected, f"{field} is not content-addressed")


def _registered_roster(
    *,
    protocol_path: str | Path,
    selection_path: str | Path,
    crossrepo_binding_path: str | Path,
) -> tuple[tuple[str, int, str], ...]:
    protocol_file = _ordinary_file(protocol_path, name="software protocol")
    selection_file = _ordinary_file(selection_path, name="selection lock")
    binding_file = _ordinary_file(
        crossrepo_binding_path,
        name="cross-repository binding",
    )
    protocol = load_strict_json_object(protocol_file, label="software protocol")
    selection = load_strict_json_object(selection_file, label="selection lock")
    binding = load_strict_json_object(binding_file, label="cross-repository binding")
    _identity(protocol, "protocol_id", SOFTWARE_PROTOCOL_ID)
    _identity(binding, "binding_id", CROSSREPO_BINDING_ID)
    cohort = cast(Mapping[str, Any], protocol.get("cohort"))
    _require(
        cohort.get("selection_path") == SELECTION_REPOSITORY_PATH
        and cohort.get("selection_git_blob_sha1") == SELECTION_GIT_BLOB_SHA1,
        "protocol selection binding changed",
    )
    _require(
        _git_blob_sha1(selection_file) == SELECTION_GIT_BLOB_SHA1,
        "selection lock bytes changed",
    )
    code_protocol = cast(Mapping[str, Any], binding.get("code_protocol"))
    paper_protocol = cast(Mapping[str, Any], binding.get("paper_protocol"))
    _require(
        code_protocol.get("protocol_id") == SOFTWARE_PROTOCOL_ID
        and paper_protocol.get("protocol_id") == PAPER_PROTOCOL_ID,
        "cross-repository protocol binding changed",
    )
    selection_groups = cast(Mapping[str, Any], selection.get("selection"))
    calibration = cast(Sequence[Mapping[str, Any]], selection_groups.get("calibration"))
    roster_rows: list[tuple[str, int, str]] = []
    for row in calibration:
        object_id = row.get("object_id")
        episode_id = row.get("episode_id")
        stratum = row.get("stratum")
        _require(
            type(object_id) is str
            and bool(object_id)
            and type(episode_id) is int
            and not isinstance(episode_id, bool)
            and type(stratum) is str
            and bool(stratum),
            "selection calibration row changed",
        )
        roster_rows.append(
            (
                cast(str, object_id),
                cast(int, episode_id),
                cast(str, stratum),
            )
        )
    roster = tuple(roster_rows)
    from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
        SOURCE_ROSTER,
    )

    _require(roster == SOURCE_ROSTER, "selection calibration roster changed")
    _require(
        len({object_id for object_id, _episode, _stratum in roster}) == len(roster),
        "selection calibration roster repeats an object",
    )
    return roster


def _npy_header(stream: IO[bytes]) -> dict[str, Any]:
    _require(stream.read(6) == b"\x93NUMPY", "array member is not NPY")
    major, minor = struct.unpack("BB", stream.read(2))
    if major == 1:
        header_length = struct.unpack("<H", stream.read(2))[0]
    elif major in {2, 3}:
        header_length = struct.unpack("<I", stream.read(4))[0]
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}")
    encoded = stream.read(header_length)
    parsed = ast.literal_eval(encoded.decode("latin1" if major < 3 else "utf-8"))
    _require(isinstance(parsed, dict), "NPY header is not a mapping")
    shape = parsed.get("shape")
    _require(isinstance(shape, tuple), "NPY shape is not a tuple")
    return {
        "dtype": parsed.get("descr"),
        "fortran_order": parsed.get("fortran_order"),
        "shape": list(shape),
        "version": [major, minor],
    }


def _header_record(path: Path, *, relative_path: str, root_name: str) -> dict[str, Any]:
    stat = path.stat()
    suffix = path.suffix.lower()
    record: dict[str, Any] = {
        "relative_path": relative_path,
        "root": root_name,
        "size_bytes": stat.st_size,
        "suffix": suffix,
    }
    if suffix == ".npy":
        with path.open("rb") as stream:
            record["array_header"] = _npy_header(stream)
    elif suffix == ".npz":
        members: list[dict[str, Any]] = []
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda value: value.filename):
                _require(not info.is_dir(), "NPZ member directories are forbidden")
                member: dict[str, Any] = {
                    "compressed_size": info.compress_size,
                    "name": info.filename,
                    "size_bytes": info.file_size,
                }
                if info.filename.endswith(".npy"):
                    with archive.open(info) as stream:
                        member["array_header"] = _npy_header(stream)
                members.append(member)
        record["npz_members"] = members
    elif suffix == ".json" and stat.st_size <= _JSON_LIMIT_BYTES:
        try:
            value = load_strict_json_object(
                path, label=f"inventory JSON {relative_path}"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            record["json_error"] = type(error).__name__
        else:
            record["json_top_level_keys"] = sorted(value)
            for field in (
                "artifact_kind",
                "episode_id",
                "object_id",
                "schema",
                "schema_version",
                "status",
            ):
                scalar = value.get(field)
                if scalar is None or type(scalar) in {bool, float, int, str}:
                    record[f"json_{field}"] = scalar
    return record


def _scan_root(
    root: Path,
    *,
    root_name: str,
    object_ids: tuple[str, ...],
    forbidden_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counts = {object_id: 0 for object_id in object_ids}
    total_files = 0
    total_bytes = 0
    suffix_counts: dict[str, int] = {}
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        _require(
            forbidden_root != current and forbidden_root not in current.parents,
            "inventory entered the forbidden confirmation root",
        )
        directory_names[:] = sorted(
            name for name in directory_names if not (current / name).is_symlink()
        )
        for file_name in sorted(file_names):
            path = current / file_name
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            _require(
                resolved == path
                and forbidden_root != resolved
                and forbidden_root not in resolved.parents,
                "inventory file escaped its admitted source root",
            )
            relative = path.relative_to(root).as_posix()
            matching = tuple(
                object_id
                for object_id in object_ids
                if object_id in relative.split("/")
            )
            if not matching:
                continue
            stat = path.stat()
            total_files += 1
            total_bytes += stat.st_size
            suffix = path.suffix.lower() or "<none>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            for object_id in matching:
                counts[object_id] += 1
            records.append(
                _header_record(
                    path,
                    relative_path=relative,
                    root_name=root_name,
                )
            )
    return (
        records,
        counts,
        {
            "byte_count": total_bytes,
            "file_count": total_files,
            "path": root.as_posix(),
            "suffix_counts": dict(sorted(suffix_counts.items())),
        },
    )


def build_covariance_source_inventory_v1(
    *,
    protocol_path: str | Path,
    selection_path: str | Path,
    crossrepo_binding_path: str | Path,
    calibration_source_root: str | Path,
    calibration_processed_root: str | Path,
    forbidden_confirmation_root: str | Path,
    implementation_revision: str,
) -> dict[str, Any]:
    """Build one source-roster-complete inventory without reading array values."""

    roster = _registered_roster(
        protocol_path=protocol_path,
        selection_path=selection_path,
        crossrepo_binding_path=crossrepo_binding_path,
    )
    revision = str(implementation_revision)
    _require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "implementation_revision must be a lowercase Git SHA-1",
    )
    source_root = _ordinary_directory(calibration_source_root, name="source root")
    processed_root = _ordinary_directory(
        calibration_processed_root,
        name="processed root",
    )
    forbidden_root = _ordinary_directory(
        forbidden_confirmation_root,
        name="forbidden confirmation root",
    )
    _require(
        len({source_root, processed_root, forbidden_root}) == 3,
        "source, processed, and forbidden roots must differ",
    )
    for admitted in (source_root, processed_root):
        _require(
            forbidden_root not in admitted.parents
            and admitted not in forbidden_root.parents,
            "an admitted source root overlaps the forbidden confirmation root",
        )
    object_ids = tuple(object_id for object_id, _episode, _stratum in roster)
    files: list[dict[str, Any]] = []
    counts = {object_id: 0 for object_id in object_ids}
    roots: dict[str, Any] = {}
    for name, root in (
        ("calibration_source", source_root),
        ("calibration_processed", processed_root),
    ):
        rows, local_counts, summary = _scan_root(
            root,
            root_name=name,
            object_ids=object_ids,
            forbidden_root=forbidden_root,
        )
        files.extend(rows)
        roots[name] = summary
        for object_id, count in local_counts.items():
            counts[object_id] += count
    missing = tuple(object_id for object_id in object_ids if counts[object_id] == 0)
    _require(not missing, "one or more frozen source objects are absent")
    protocol_file = _ordinary_file(protocol_path, name="software protocol")
    selection_file = _ordinary_file(selection_path, name="selection lock")
    binding_file = _ordinary_file(
        crossrepo_binding_path,
        name="cross-repository binding",
    )
    identity: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "software_protocol_id": SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "crossrepo_binding_id": CROSSREPO_BINDING_ID,
        "selection_git_blob_sha1": SELECTION_GIT_BLOB_SHA1,
        "implementation_revision": revision,
        "contract_files": {
            "crossrepo_binding_sha256": _sha256_file(binding_file),
            "protocol_sha256": _sha256_file(protocol_file),
            "selection_sha256": _sha256_file(selection_file),
        },
        "source_roster": [
            {"episode": episode, "object_id": object_id, "stratum": stratum}
            for object_id, episode, stratum in roster
        ],
        "roots": roots,
        "source_unit_path_counts": counts,
        "missing_source_units": list(missing),
        "files": sorted(
            files,
            key=lambda row: (str(row["root"]), str(row["relative_path"])),
        ),
        "information_boundary": dict(_BOUNDARY),
    }
    return cast(
        dict[str, Any], plain_json({**identity, "inventory_id": content_id(identity)})
    )


def validate_covariance_source_inventory_v1(value: object) -> dict[str, Any]:
    """Validate a previously generated target-closed inventory artifact."""

    _require(isinstance(value, Mapping), "source inventory must be a JSON object")
    document = cast(Mapping[str, Any], value)
    _require(
        document.get("schema") == INVENTORY_SCHEMA
        and document.get("schema_version") == INVENTORY_SCHEMA_VERSION,
        "source inventory schema changed",
    )
    _require(
        document.get("software_protocol_id") == SOFTWARE_PROTOCOL_ID
        and document.get("paper_protocol_id") == PAPER_PROTOCOL_ID
        and document.get("crossrepo_binding_id") == CROSSREPO_BINDING_ID
        and document.get("selection_git_blob_sha1") == SELECTION_GIT_BLOB_SHA1,
        "source inventory scientific identity changed",
    )
    _require(document.get("information_boundary") == _BOUNDARY, "boundary changed")
    _require(document.get("missing_source_units") == [], "source roster is incomplete")
    from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
        SOURCE_ROSTER,
    )

    expected_roster = [
        {"episode": episode, "object_id": object_id, "stratum": stratum}
        for object_id, episode, stratum in SOURCE_ROSTER
    ]
    _require(document.get("source_roster") == expected_roster, "source roster changed")
    declared = document.get("inventory_id")
    _require(isinstance(declared, str) and len(declared) == 64, "invalid inventory ID")
    identity = {key: item for key, item in document.items() if key != "inventory_id"}
    _require(content_id(identity) == declared, "source inventory identity changed")
    return cast(dict[str, Any], plain_json(document))


def publish_covariance_source_inventory_v1(
    value: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Publish one content-addressed inventory without replacing existing bytes."""

    document = validate_covariance_source_inventory_v1(value)
    target = Path(output_path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return target


__all__ = [
    "CROSSREPO_BINDING_ID",
    "INVENTORY_SCHEMA",
    "PAPER_PROTOCOL_ID",
    "SELECTION_GIT_BLOB_SHA1",
    "SOFTWARE_PROTOCOL_ID",
    "build_covariance_source_inventory_v1",
    "publish_covariance_source_inventory_v1",
    "validate_covariance_source_inventory_v1",
]
