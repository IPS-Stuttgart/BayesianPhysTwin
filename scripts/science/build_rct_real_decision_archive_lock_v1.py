#!/usr/bin/env python3
"""Hash-lock the public RCT archive without opening force outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rct_real_decision_protocol import (
    RCT_ARCHIVE_FILE_ID,
    RCT_ARCHIVE_SIZE_BYTES,
    RCT_CODE_REVISION,
    load_rct_preoutcome_amendment_v2,
    load_rct_preoutcome_clarification,
    load_rct_real_decision_protocol,
    protocol_config_sha256,
    protocol_file_sha256,
)

SCHEMA = "bayesian-phystwin.rct-real-decision-archive-lock"
SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _force_metadata_member(archive: Path) -> zipfile.ZipInfo:
    with zipfile.ZipFile(archive) as bundle:
        matches = [
            member
            for member in bundle.infolist()
            if Path(member.filename).name == "force_metadata.csv"
            and not member.is_dir()
        ]
    _require(len(matches) == 1, "archive must contain exactly one force_metadata.csv")
    member = matches[0]
    _require(member.flag_bits & 0x1 == 0, "force metadata member is encrypted")
    return member


def _force_metadata_header(
    archive: Path, member: zipfile.ZipInfo
) -> tuple[bytes, list[str]]:
    with zipfile.ZipFile(archive) as bundle, bundle.open(member, "r") as stream:
        header = stream.readline(4097)
    _require(header.endswith((b"\n", b"\r")), "force metadata header is too long")
    _require(len(header) <= 4096, "force metadata header is too long")
    _require(b'"' not in header, "quoted force metadata header is unsupported")
    columns = header.rstrip(b"\r\n").decode("ascii").split(",")
    _require(columns[0] == "material_id", "material_id is not the first CSV column")
    _require(
        {"material_id", "position", "sensor", "z_frame", "raw_fz"} <= set(columns),
        "force metadata schema is missing a registered column",
    )
    return header, columns


def _build_lock(
    archive: Path,
    protocol_path: Path,
    clarification_path: Path,
    amendment_v2_path: Path,
    *,
    expected_archive_size: int = RCT_ARCHIVE_SIZE_BYTES,
) -> dict[str, Any]:
    _require(archive.is_file() and not archive.is_symlink(), "archive path is invalid")
    _require(
        archive.stat().st_size == expected_archive_size,
        "archive byte size changed",
    )
    protocol = load_rct_real_decision_protocol(protocol_path)
    clarification = load_rct_preoutcome_clarification(clarification_path)
    amendment_v2 = load_rct_preoutcome_amendment_v2(amendment_v2_path)
    archive_sha256 = _sha256(archive)
    member = _force_metadata_member(archive)
    header, columns = _force_metadata_header(archive, member)
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "archive_file_id": RCT_ARCHIVE_FILE_ID,
        "archive_file_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "force_metadata_member": member.filename,
        "force_metadata_crc32": f"{member.CRC:08x}",
        "force_metadata_compressed_size": member.compress_size,
        "force_metadata_uncompressed_size": member.file_size,
        "force_metadata_header_columns": columns,
        "force_metadata_header_sha256": hashlib.sha256(header).hexdigest(),
        "rct_code_revision": RCT_CODE_REVISION,
        "protocol_file_sha256": protocol_file_sha256(protocol_path),
        "protocol_config_sha256": protocol_config_sha256(protocol),
        "clarification_file_sha256": protocol_file_sha256(clarification_path),
        "clarification_config_sha256": protocol_config_sha256(clarification),
        "amendment_v2_file_sha256": protocol_file_sha256(amendment_v2_path),
        "amendment_v2_config_sha256": protocol_config_sha256(amendment_v2),
        "archive_integrity_verified": True,
        "force_metadata_header_opened": True,
        "force_metadata_content_opened": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    return {**identity, "lock_id": content_id(identity)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--clarification", type=Path, required=True)
    parser.add_argument("--amendment-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    _require(not output.exists(), "archive lock output already exists")
    lock = _build_lock(
        arguments.archive.resolve(strict=True),
        arguments.protocol.resolve(strict=True),
        arguments.clarification.resolve(strict=True),
        arguments.amendment_v2.resolve(strict=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "archive_sha256": lock["archive_sha256"],
                "lock_id": lock["lock_id"],
                "force_metadata_content_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
