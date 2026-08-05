"""Opaque, checksum-bound staging for the fresh PokeFlex target cohort."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .pokeflex_conservative_shrinkage_target import (
    ACTION_ROBUST_FRESH6_PUBLIC_TARGET_TAKE_IDS,
    ACTION_ROBUST_FRESH6_PUBLIC_ZIP_SHA256,
    FRESH12_PUBLIC_TARGET_TAKE_IDS,
    FRESH12_PUBLIC_ZIP_SHA256,
    INSTANCE_FRESH12_PUBLIC_TARGET_TAKE_IDS,
    INSTANCE_FRESH12_PUBLIC_ZIP_SHA256,
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
    TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
    TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
    canonical_payload_sha256,
    file_sha256,
    load_pokeflex_shrinkage_target_protocol,
)

STAGE_MANIFEST_KIND = "PokeFlexFresh12OpaqueStageManifest"
STAGE_MANIFEST_NAME = "source_stage_manifest.json"
_DEPTH_PATTERN = re.compile(r"^kinect/(?P<camera>[01])/depth/[0-9]{5}\.png$")
_CALIBRATION_PATTERN = re.compile(r"^kinect/(?P<camera>[01])/camera_parameters\.json$")
_MESH_PATTERN = re.compile(r"^meshes/mesh-f[0-9]{5}\.obj$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _registered_archives(
    protocol_id: str,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    if protocol_id == TARGET_PROTOCOL_FRESH12_PUBLIC_V1:
        return FRESH12_PUBLIC_TARGET_TAKE_IDS, FRESH12_PUBLIC_ZIP_SHA256
    if protocol_id == TARGET_PROTOCOL_INSTANCE_FRESH12_V2:
        return (
            INSTANCE_FRESH12_PUBLIC_TARGET_TAKE_IDS,
            INSTANCE_FRESH12_PUBLIC_ZIP_SHA256,
        )
    if protocol_id == TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3:
        return (
            ACTION_ROBUST_FRESH6_PUBLIC_TARGET_TAKE_IDS,
            ACTION_ROBUST_FRESH6_PUBLIC_ZIP_SHA256,
        )
    raise ValueError("stage protocol family changed")


def stage_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of one stage manifest."""

    return canonical_payload_sha256(payload, digest_field="stage_manifest_sha256")


def _authorized_kind(relative: PurePosixPath) -> tuple[str, str | None] | None:
    value = relative.as_posix()
    if value == "robot_data.json":
        return "robot", None
    match = _CALIBRATION_PATTERN.match(value)
    if match is not None:
        return "calibration", match.group("camera")
    match = _DEPTH_PATTERN.match(value)
    if match is not None:
        return "depth", match.group("camera")
    if _MESH_PATTERN.match(value) is not None:
        return "mesh", None
    return None


def _validate_file_inventory(files: object) -> tuple[dict[str, Any], ...]:
    _require(isinstance(files, list) and bool(files), "stage files are missing")
    records: list[dict[str, Any]] = []
    kinds: Counter[tuple[str, str | None]] = Counter()
    paths: set[str] = set()
    for raw in files:
        _require(isinstance(raw, Mapping), "stage file record is invalid")
        record = dict(raw)
        relative = PurePosixPath(str(record.get("path", "")))
        _require(bool(relative.parts), "stage file path is empty")
        _require(".." not in relative.parts, "stage file path escapes the take")
        authorized = _authorized_kind(relative)
        _require(authorized is not None, "stage includes an unauthorized member")
        _require(record.get("kind") == authorized[0], "stage file kind changed")
        _require(record.get("camera") == authorized[1], "stage camera changed")
        digest = str(record.get("sha256", ""))
        _require(
            _SHA256_PATTERN.fullmatch(digest) is not None,
            "stage file digest is invalid",
        )
        _require(int(record.get("byte_count", -1)) >= 0, "stage byte count is invalid")
        path = relative.as_posix()
        _require(path not in paths, "stage file path repeats")
        paths.add(path)
        kinds[authorized] += 1
        records.append(record)
    _require(
        [str(record["path"]) for record in records]
        == sorted(str(record["path"]) for record in records),
        "stage file inventory is not canonical",
    )
    _require(kinds[("robot", None)] == 1, "stage robot inventory changed")
    _require(
        kinds[("calibration", "0")] == kinds[("calibration", "1")] == 1,
        "stage calibration inventory changed",
    )
    _require(
        kinds[("depth", "0")] > 0 and kinds[("depth", "0")] == kinds[("depth", "1")],
        "stage depth inventory changed",
    )
    _require(kinds[("mesh", None)] > 0, "stage mesh inventory changed")
    return tuple(records)


def validate_pokeflex_fresh12_stage_manifest(
    path: str | Path,
    protocol: Mapping[str, Any],
    *,
    expected_take_id: str | None = None,
) -> dict[str, Any]:
    """Validate a byte-staging manifest without reading staged mesh content."""

    source = Path(path).resolve()
    _require(source.name == STAGE_MANIFEST_NAME, "stage manifest name changed")
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "stage schema changed")
    _require(payload.get("artifact_kind") == STAGE_MANIFEST_KIND, "stage kind changed")
    _require(
        payload.get("stage_manifest_sha256") == stage_manifest_sha256(payload),
        "stage manifest checksum mismatch",
    )
    target_take_ids, archive_sha256 = _registered_archives(
        str(protocol.get("protocol_id"))
    )
    _require(
        payload.get("protocol_sha256") == protocol.get("protocol_sha256"),
        "stage protocol changed",
    )
    take_id = str(payload.get("take_id", ""))
    _require(take_id in target_take_ids, "stage take is not registered")
    if expected_take_id is not None:
        _require(take_id == expected_take_id, "stage take changed")
    _require(payload.get("archive_name") == f"{take_id}.zip", "archive name changed")
    _require(
        payload.get("archive_sha256") == archive_sha256[take_id],
        "archive digest changed",
    )
    _require(int(payload.get("archive_byte_count", -1)) > 0, "archive size changed")
    _require(
        payload.get("target_mesh_geometry_decoded") is False,
        "stage decoded target geometry",
    )
    _require(
        payload.get("outcome_metric_computed") is False,
        "stage computed an outcome metric",
    )
    records = _validate_file_inventory(payload.get("files"))
    return {
        "path": source,
        "payload": payload,
        "take_id": take_id,
        "files_by_path": {str(row["path"]): row for row in records},
        "stage_manifest_sha256": payload["stage_manifest_sha256"],
    }


def validate_staged_file(
    path: str | Path,
    take_root: str | Path,
    files_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify one staged file against its opaque manifest record."""

    source = Path(path).resolve()
    root = Path(take_root).resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("staged file escapes the take root") from error
    record = files_by_path.get(relative)
    _require(record is not None, f"staged file is absent from manifest: {relative}")
    _require(source.is_file(), f"staged file is missing: {relative}")
    _require(
        source.stat().st_size == int(record["byte_count"]),
        f"staged file size changed: {relative}",
    )
    _require(
        file_sha256(source) == record["sha256"],
        f"staged file bytes changed: {relative}",
    )
    return dict(record)


def stage_pokeflex_fresh12_archive(
    archive_path: str | Path,
    destination_root: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Extract only registered input/score members without parsing their content."""

    archive = Path(archive_path).resolve()
    destination_root = Path(destination_root).resolve()
    protocol = load_pokeflex_shrinkage_target_protocol(Path(protocol_path))
    target_take_ids, archive_sha256 = _registered_archives(protocol["protocol_id"])
    take_id = archive.stem
    _require(take_id in target_take_ids, "archive is not registered")
    archive_digest = file_sha256(archive)
    _require(
        archive_digest == archive_sha256[take_id],
        "archive bytes changed",
    )
    destination = destination_root / take_id
    _require(not destination.exists(), "stage destination already exists")

    selected: list[tuple[zipfile.ZipInfo, PurePosixPath, str, str | None]] = []
    with zipfile.ZipFile(archive) as payload:
        for member in payload.infolist():
            path = PurePosixPath(member.filename)
            _require(".." not in path.parts, "archive member escapes its root")
            if member.is_dir() or len(path.parts) < 2 or path.parts[0] != take_id:
                continue
            relative = PurePosixPath(*path.parts[1:])
            authorized = _authorized_kind(relative)
            if authorized is not None:
                selected.append((member, relative, *authorized))
        selected = sorted(selected, key=lambda item: item[1].as_posix())
        preview = [
            {
                "path": relative.as_posix(),
                "kind": kind,
                "camera": camera,
                "byte_count": member.file_size,
                "sha256": "0" * 64,
            }
            for member, relative, kind, camera in selected
        ]
        _validate_file_inventory(preview)

        records = []
        for member, relative, kind, camera in selected:
            content = payload.read(member)
            target = destination / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            records.append(
                {
                    "path": relative.as_posix(),
                    "kind": kind,
                    "camera": camera,
                    "byte_count": len(content),
                    "sha256": _bytes_sha256(content),
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": STAGE_MANIFEST_KIND,
        "protocol_sha256": protocol["protocol_sha256"],
        "take_id": take_id,
        "archive_name": archive.name,
        "archive_sha256": archive_digest,
        "archive_byte_count": archive.stat().st_size,
        "authorized_member_policy": (
            "robot_data.json, Kinect 0/1 calibration and depth, and mesh OBJ bytes"
        ),
        "target_mesh_geometry_decoded": False,
        "outcome_metric_computed": False,
        "files": records,
    }
    manifest["stage_manifest_sha256"] = stage_manifest_sha256(manifest)
    manifest_path = destination / STAGE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest": str(manifest_path), **manifest}


__all__ = [
    "STAGE_MANIFEST_KIND",
    "STAGE_MANIFEST_NAME",
    "stage_manifest_sha256",
    "stage_pokeflex_fresh12_archive",
    "validate_staged_file",
    "validate_pokeflex_fresh12_stage_manifest",
]
