#!/usr/bin/env python3
"""Capture or verify a deterministic SHA-256 milestone file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Hash one regular file and return its digest and byte count."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def capture_manifest(specification: dict[str, Any]) -> dict[str, Any]:
    """Hash every source and verify any declared archive copy."""

    required = {"schema_version", "milestone", "captured_at", "host", "entries"}
    missing = required - set(specification)
    if missing:
        raise ValueError(f"manifest specification is missing: {sorted(missing)}")
    identifiers: set[str] = set()
    records = []
    for entry in specification["entries"]:
        identifier = str(entry["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate manifest id: {identifier}")
        identifiers.add(identifier)
        digest, byte_count = sha256_file(entry["source_path"])
        record = {
            "id": identifier,
            "category": str(entry["category"]),
            "source_path": str(Path(entry["source_path"])),
            "bytes": byte_count,
            "sha256": digest,
        }
        archive_path = entry.get("archive_path")
        if archive_path is not None:
            archive_digest, archive_bytes = sha256_file(archive_path)
            if archive_digest != digest or archive_bytes != byte_count:
                raise ValueError(f"archive copy differs for {identifier}")
            record["archive_path"] = str(Path(archive_path))
            record["archive_verified"] = True
        records.append(record)
    return {
        "schema_version": int(specification["schema_version"]),
        "milestone": str(specification["milestone"]),
        "captured_at": str(specification["captured_at"]),
        "host": str(specification["host"]),
        "entry_count": len(records),
        "entries": records,
    }


def verify_manifest(
    manifest: dict[str, Any],
    *,
    location: str,
) -> dict[str, Any]:
    """Verify source or archive paths against a captured manifest."""

    if location not in {"source", "archive"}:
        raise ValueError("location must be source or archive")
    checked = 0
    skipped = 0
    failures = []
    for entry in manifest["entries"]:
        path_key = "source_path" if location == "source" else "archive_path"
        if path_key not in entry:
            skipped += 1
            continue
        path = Path(entry[path_key])
        try:
            digest, byte_count = sha256_file(path)
        except FileNotFoundError:
            failures.append({"id": entry["id"], "reason": "missing", "path": str(path)})
            continue
        checked += 1
        if digest != entry["sha256"] or byte_count != entry["bytes"]:
            failures.append(
                {
                    "id": entry["id"],
                    "reason": "checksum_mismatch",
                    "path": str(path),
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": digest,
                    "expected_bytes": entry["bytes"],
                    "actual_bytes": byte_count,
                }
            )
    return {
        "milestone": manifest["milestone"],
        "location": location,
        "checked": checked,
        "skipped": skipped,
        "failures": failures,
        "passed": not failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("specification_json")
    capture.add_argument("output_json")
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest_json")
    verify.add_argument("--location", choices=("source", "archive"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        specification = json.loads(
            Path(args.specification_json).read_text(encoding="utf-8")
        )
        result = capture_manifest(specification)
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(output.resolve()), **result}, indent=2))
        return 0
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    result = verify_manifest(manifest, location=args.location)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
