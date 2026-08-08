#!/usr/bin/env python3
"""Stage verified calibration files from a runner-resident Deform360 snapshot.

The script consumes an already sealed calibration-source plan and stages only
its planned calibration files into the isolated calibration download root.
Local reuse is copy-on-write only: a reflink is attempted after exact byte
verification, and a failed reflink is left for the existing downloader to
obtain.  The immutable raw snapshot is never hard-linked or copied eagerly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "bayesian-phystwin/deform360-calibration-source-plan-v1"
MANIFEST_SCHEMA = "bayesian-phystwin/deform360-local-calibration-cache-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any], *, digest_key: str) -> str:
    payload = dict(value)
    payload.pop(digest_key, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(record: Mapping[str, Any]) -> str:
    value = record.get("path")
    _require(isinstance(value, str) and value, "planned file path is malformed")
    path = Path(value)
    _require(not path.is_absolute(), f"planned file path is absolute: {value}")
    _require(".." not in path.parts, f"planned file path escapes root: {value}")
    return value


def _planned_records(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    objects = plan.get("objects")
    _require(isinstance(objects, list), "plan objects are malformed")
    records: dict[str, dict[str, Any]] = {}
    for row in objects:
        _require(isinstance(row, Mapping), "plan object row is malformed")
        if row.get("status") != "planned":
            continue
        files = row.get("selected_files")
        _require(isinstance(files, list), "planned selected_files are malformed")
        for raw_record in files:
            _require(isinstance(raw_record, Mapping), "planned file is malformed")
            record = dict(raw_record)
            relative = _relative_path(record)
            previous = records.get(relative)
            if previous is not None:
                _require(previous == record, f"planned file changed: {relative}")
            records[relative] = record
    _require(records, "plan contains no calibration files")
    return tuple(records[path] for path in sorted(records))


def _verify_regular_file(
    path: Path,
    *,
    root: Path,
    record: Mapping[str, Any],
) -> tuple[int, bool]:
    relative = _relative_path(record)
    _require(path.exists(), f"file disappeared: {relative}")
    _require(not path.is_symlink(), f"symlink is forbidden: {relative}")
    resolved = path.resolve()
    _require(resolved.is_relative_to(root), f"file escaped root: {relative}")
    _require(path.is_file(), f"not a regular file: {relative}")

    size = path.stat().st_size
    declared_size = record.get("size")
    if isinstance(declared_size, int):
        _require(size == declared_size, f"size mismatch: {relative}")

    expected_sha256 = record.get("lfs_sha256")
    sha_verified = False
    if isinstance(expected_sha256, str):
        _require(len(expected_sha256) == 64, f"bad LFS digest: {relative}")
        _require(
            _sha256(path) == expected_sha256,
            f"LFS digest mismatch: {relative}",
        )
        sha_verified = True
    return size, sha_verified


def _try_reflink(source: Path, destination: Path) -> bool:
    """Create one CoW clone without falling back to a full byte copy."""

    result = subprocess.run(
        (
            "cp",
            "--reflink=always",
            "--preserve=mode,timestamps",
            "--",
            str(source),
            str(destination),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if destination.exists():
        destination.unlink()
    return False


def stage_local_calibration_cache(
    *,
    plan_path: Path,
    source_root: Path,
    destination_root: Path,
) -> dict[str, Any]:
    """Reflink locally available planned files into the calibration root."""

    plan_file = plan_path.expanduser().resolve()
    source = source_root.expanduser().resolve()
    destination = destination_root.expanduser().resolve()
    _require(source.is_dir(), f"source root is missing: {source}")
    _require(source != destination, "source and destination roots must differ")
    _require(
        not destination.is_relative_to(source),
        "destination root must not be inside the immutable source snapshot",
    )

    plan = _load_json(plan_file)
    _require(plan.get("schema") == PLAN_SCHEMA, "plan schema changed")
    plan_sha256 = plan.get("plan_sha256")
    _require(isinstance(plan_sha256, str), "plan digest is missing")
    _require(
        plan_sha256 == _canonical_sha256(plan, digest_key="plan_sha256"),
        "plan digest changed",
    )
    boundary = plan.get("information_boundary")
    _require(isinstance(boundary, Mapping), "plan information boundary is missing")
    _require(
        boundary.get("calibration_payloads_opened") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False,
        "plan information boundary changed",
    )

    destination.mkdir(parents=True, exist_ok=True)
    reflinked: list[str] = []
    reused: list[str] = []
    missing_in_source: list[str] = []
    reflink_unavailable: list[str] = []
    verified_sha256: list[str] = []
    staged_bytes = 0
    records = _planned_records(plan)

    for record in records:
        relative = _relative_path(record)
        source_path = source / relative
        destination_path = destination / relative
        if not source_path.exists():
            missing_in_source.append(relative)
            continue

        size, sha_verified = _verify_regular_file(
            source_path,
            root=source,
            record=record,
        )
        if sha_verified:
            verified_sha256.append(relative)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            _verify_regular_file(
                destination_path,
                root=destination,
                record=record,
            )
            reused.append(relative)
            staged_bytes += size
            continue

        if not _try_reflink(source_path, destination_path):
            reflink_unavailable.append(relative)
            continue
        _verify_regular_file(
            destination_path,
            root=destination,
            record=record,
        )
        reflinked.append(relative)
        staged_bytes += size

    download_fallback = sorted((*missing_in_source, *reflink_unavailable))
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "dataset_revision": plan.get("dataset_revision"),
        "source_root": str(source),
        "destination_root": str(destination),
        "planned_file_count": len(records),
        "local_source_file_count": len(records) - len(missing_in_source),
        "reflinked_file_count": len(reflinked),
        "reused_file_count": len(reused),
        "missing_in_source_count": len(missing_in_source),
        "reflink_unavailable_count": len(reflink_unavailable),
        "download_fallback_file_count": len(download_fallback),
        "sha256_verified_file_count": len(verified_sha256),
        "staged_bytes": staged_bytes,
        "missing_in_source_paths": missing_in_source,
        "reflink_unavailable_paths": reflink_unavailable,
        "download_fallback_paths": download_fallback,
        "information_boundary": {
            "official_raw_payloads_opened_for_hash_verification": True,
            "calibration_payloads_staged": bool(reflinked or reused),
            "confirmation_payloads_opened": False,
            "adaptive_confirmation_root_accessed": False,
            "target_outcomes_used": False,
            "hardlink_allowed": False,
            "full_copy_fallback_allowed": False,
            "reflink_only": True,
        },
    }
    payload["manifest_sha256"] = _canonical_sha256(
        payload,
        digest_key="manifest_sha256",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = stage_local_calibration_cache(
        plan_path=args.plan,
        source_root=args.source_root,
        destination_root=args.destination_root,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
