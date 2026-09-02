#!/usr/bin/env python3
"""Hash-lock PoseIt and inventory its ZIP structure without reading members."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    POSEIT_GELSIGHT_FILE_ID,
    POSEIT_REPOSITORY_REVISION,
    load_poseit_real_decision_protocol,
    poseit_protocol_config_sha256,
    poseit_protocol_file_sha256,
)

SCHEMA = "bayesian-phystwin.poseit-archive-structure-lock"
PRIVATE_MANIFEST_SCHEMA = "bayesian-phystwin.poseit-private-member-manifest"
SCHEMA_VERSION = 1
ARCHIVE_FILE_NAME = "gelsight.zip"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _safe_member_name(member: zipfile.ZipInfo) -> str:
    name = member.filename
    _require(bool(name), "ZIP member name is empty")
    _require("\x00" not in name, "ZIP member name contains NUL")
    _require("\\" not in name, "ZIP member name uses a backslash")
    _require(not name.startswith("/"), "ZIP member path is absolute")
    parts = PurePosixPath(name).parts
    _require(bool(parts), "ZIP member path is empty")
    _require(".." not in parts, "ZIP member path traverses its root")
    _require(not parts[0].endswith(":"), "ZIP member path has a drive prefix")
    normalized = PurePosixPath(*parts).as_posix()
    if member.is_dir() and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _require_regular_or_directory(member: zipfile.ZipInfo) -> None:
    mode = member.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if member.is_dir():
        _require(kind in (0, stat.S_IFDIR), "ZIP member is not a directory")
    else:
        _require(kind in (0, stat.S_IFREG), "ZIP member is a link or special file")


def _member_record(member: zipfile.ZipInfo, *, normalized_name: str) -> dict[str, Any]:
    return {
        "compressed_size": member.compress_size,
        "compression_method": member.compress_type,
        "crc32": f"{member.CRC:08x}",
        "encrypted": bool(member.flag_bits & 0x1),
        "is_directory": member.is_dir(),
        "name": normalized_name,
        "uncompressed_size": member.file_size,
    }


def _member_identity_line(record: dict[str, Any]) -> bytes:
    fields = (
        record["name"],
        record["crc32"],
        str(record["compressed_size"]),
        str(record["uncompressed_size"]),
        str(record["compression_method"]),
        "1" if record["is_directory"] else "0",
    )
    return ("\0".join(fields) + "\n").encode()


def _inventory_members(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(archive, "r") as bundle:
        members = bundle.infolist()
        archive_comment = bytes(bundle.comment)

    _require(bool(members), "PoseIt archive is empty")
    records: list[dict[str, Any]] = []
    normalized_names: set[str] = set()
    for member in members:
        name = _safe_member_name(member)
        _require(name not in normalized_names, "PoseIt archive has duplicate members")
        normalized_names.add(name)
        _require(member.flag_bits & 0x1 == 0, "PoseIt archive member is encrypted")
        _require_regular_or_directory(member)
        records.append(_member_record(member, normalized_name=name))

    records.sort(key=lambda record: str(record["name"]))
    regular = [record for record in records if not record["is_directory"]]
    _require(bool(regular), "PoseIt archive has no regular members")
    extensions = Counter(
        (PurePosixPath(str(record["name"])).suffix.casefold() or "<none>")
        for record in regular
    )
    depths = Counter(
        len(PurePosixPath(str(record["name"])).parts) for record in records
    )
    top_level = {PurePosixPath(str(record["name"])).parts[0] for record in records}
    identity_digest = hashlib.sha256()
    identity_digest.update(b"poseit-zip-central-directory-v1\0")
    names_digest = hashlib.sha256()
    names_digest.update(b"poseit-zip-member-names-v1\0")
    for record in records:
        identity_digest.update(_member_identity_line(record))
        names_digest.update((str(record["name"]) + "\n").encode())
    summary = {
        "archive_comment_length": len(archive_comment),
        "archive_comment_sha256": _sha256_bytes(archive_comment),
        "central_directory_identity_sha256": identity_digest.hexdigest(),
        "directory_member_count": len(records) - len(regular),
        "extension_counts": dict(sorted(extensions.items())),
        "member_depth_counts": {
            str(depth): count for depth, count in sorted(depths.items())
        },
        "member_names_sha256": names_digest.hexdigest(),
        "regular_member_count": len(regular),
        "top_level_component_count": len(top_level),
        "total_compressed_member_bytes": sum(
            int(record["compressed_size"]) for record in regular
        ),
        "total_uncompressed_member_bytes": sum(
            int(record["uncompressed_size"]) for record in regular
        ),
    }
    return records, summary


def _build_artifacts(
    archive: Path,
    protocol_path: Path,
    *,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    _require(archive.is_file() and not archive.is_symlink(), "archive path is invalid")
    _require(archive.name == ARCHIVE_FILE_NAME, "archive file name changed")
    _require(len(expected_protocol_sha256) == 64, "protocol SHA-256 is malformed")
    actual_protocol_sha256 = poseit_protocol_file_sha256(protocol_path)
    _require(
        actual_protocol_sha256 == expected_protocol_sha256,
        "protocol file SHA-256 changed",
    )
    protocol = load_poseit_real_decision_protocol(protocol_path)
    archive_sha256 = _sha256(archive)
    records, structure = _inventory_members(archive)
    private_identity = {
        "schema": PRIVATE_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "archive_file_name": archive.name,
        "archive_sha256": archive_sha256,
        "members": records,
        "member_payload_bytes_opened": False,
        "phase_labels_opened": False,
        "sensor_payloads_opened": False,
    }
    private_manifest = {
        **private_identity,
        "manifest_id": content_id(private_identity),
    }
    private_manifest_bytes = _canonical_bytes(private_manifest)
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "archive_file_id": POSEIT_GELSIGHT_FILE_ID,
        "archive_file_name": archive.name,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive.stat().st_size,
        "poseit_repository_revision": POSEIT_REPOSITORY_REVISION,
        "protocol_file_sha256": actual_protocol_sha256,
        "protocol_config_sha256": poseit_protocol_config_sha256(protocol),
        "private_member_manifest_sha256": _sha256_bytes(private_manifest_bytes),
        "structure": structure,
        "archive_byte_identity_recorded": True,
        "zip_central_directory_parsed": True,
        "member_payload_bytes_opened": False,
        "member_payload_integrity_verified": False,
        "phase_labels_opened": False,
        "sensor_payloads_opened": False,
        "object_roles_assigned": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    return {**identity, "lock_id": content_id(identity)}, private_manifest_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--private-member-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    private_output = arguments.private_member_manifest.resolve()
    _require(output != private_output, "public and private outputs must differ")
    _require(not output.exists(), "archive lock output already exists")
    _require(not private_output.exists(), "private member manifest already exists")
    lock, private_manifest_bytes = _build_artifacts(
        arguments.archive.resolve(strict=True),
        arguments.protocol.resolve(strict=True),
        expected_protocol_sha256=arguments.expected_protocol_sha256,
    )
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_bytes(private_manifest_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(lock))
    print(
        json.dumps(
            {
                "archive_sha256": lock["archive_sha256"],
                "lock_id": lock["lock_id"],
                "member_payload_bytes_opened": False,
                "output": str(output),
                "private_member_manifest": str(private_output),
                "private_member_manifest_sha256": lock[
                    "private_member_manifest_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
