#!/usr/bin/env python3
"""Fail closed when a Deform360 runner lacks plan-derived writable capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "bayesian-phystwin.deform360-runner-capacity"
SCHEMA_VERSION = 1
DEFAULT_RESERVE_BYTES = 20 * 1024**3
DEFAULT_PROCESSED_MULTIPLIER = 2.0


class CapacityContractError(ValueError):
    """Raised when the plan or filesystem contract is malformed."""


@dataclass(frozen=True, slots=True)
class PlannedFile:
    path: PurePosixPath
    size: int


@dataclass(frozen=True, slots=True)
class FilesystemSnapshot:
    device: int
    available_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise CapacityContractError("selected file path must be a non-empty string")
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise CapacityContractError(f"selected file path is not portable: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise CapacityContractError(f"selected file path is not canonical: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise CapacityContractError(f"selected file path escapes its root: {value!r}")
    return path


def load_planned_files(plan_path: Path) -> tuple[list[PlannedFile], str]:
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityContractError(f"cannot read capacity plan: {error}") from error
    if not isinstance(payload, Mapping):
        raise CapacityContractError("capacity plan root must be a JSON object")
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise CapacityContractError(
            "capacity plan must contain a non-empty objects list"
        )

    unique: dict[PurePosixPath, PlannedFile] = {}
    for object_index, row in enumerate(objects):
        if not isinstance(row, Mapping):
            raise CapacityContractError(f"objects[{object_index}] must be an object")
        selected = row.get("selected_files")
        if not isinstance(selected, list):
            raise CapacityContractError(
                f"objects[{object_index}].selected_files must be a list"
            )
        for file_index, item in enumerate(selected):
            if not isinstance(item, Mapping):
                raise CapacityContractError(
                    f"objects[{object_index}].selected_files[{file_index}] "
                    "must be an object"
                )
            path = _canonical_relative_path(item.get("path"))
            size = item.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise CapacityContractError(
                    f"selected file {path.as_posix()!r} has an invalid size"
                )
            planned = PlannedFile(path=path, size=size)
            previous = unique.get(path)
            if previous is not None and previous != planned:
                raise CapacityContractError(
                    f"selected file {path.as_posix()!r} has conflicting sizes"
                )
            unique[path] = planned
    if not unique:
        raise CapacityContractError("capacity plan does not select any files")
    return sorted(unique.values(), key=lambda item: item.path.as_posix()), _sha256(
        plan_path
    )


def _reject_symlink_components(root: Path, relative: PurePosixPath) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CapacityContractError(
                f"writable data path contains a symbolic link: {relative.as_posix()!r}"
            )
        if not current.exists():
            return


def classify_existing_files(
    files: Sequence[PlannedFile], data_root: Path
) -> tuple[int, int, int, int]:
    exact_count = 0
    exact_bytes = 0
    missing_count = 0
    missing_bytes = 0
    for item in files:
        _reject_symlink_components(data_root, item.path)
        destination = data_root.joinpath(*item.path.parts)
        if destination.is_file() and destination.stat().st_size == item.size:
            exact_count += 1
            exact_bytes += item.size
        else:
            missing_count += 1
            missing_bytes += item.size
    return exact_count, exact_bytes, missing_count, missing_bytes


def _default_filesystem_probe(path: Path) -> FilesystemSnapshot:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    usage = os.statvfs(resolved)
    return FilesystemSnapshot(
        device=int(stat.st_dev),
        available_bytes=int(usage.f_bavail * usage.f_frsize),
    )


def build_capacity_report(
    *,
    plan_path: Path,
    data_root: Path,
    processed_root: Path,
    cache_root: Path,
    reserve_bytes: int,
    processed_multiplier: float,
    filesystem_probe: Callable[[Path], FilesystemSnapshot] | None = None,
) -> dict[str, Any]:
    if filesystem_probe is None:
        filesystem_probe = _default_filesystem_probe
    if reserve_bytes < 0:
        raise CapacityContractError("reserve bytes must be non-negative")
    if not math.isfinite(processed_multiplier) or processed_multiplier < 0:
        raise CapacityContractError(
            "processed multiplier must be finite and non-negative"
        )
    for role, root in (
        ("data", data_root),
        ("processed", processed_root),
        ("cache", cache_root),
    ):
        if not root.is_dir():
            raise CapacityContractError(f"{role} root is not a directory: {root}")
        if root.is_symlink():
            raise CapacityContractError(
                f"{role} root must not be a symbolic link: {root}"
            )

    files, plan_sha256 = load_planned_files(plan_path)
    exact_count, exact_bytes, missing_count, missing_bytes = classify_existing_files(
        files, data_root
    )
    planned_bytes = sum(item.size for item in files)
    processed_bytes = math.ceil(planned_bytes * processed_multiplier)

    snapshots = {
        "data": filesystem_probe(data_root),
        "processed": filesystem_probe(processed_root),
        "cache": filesystem_probe(cache_root),
    }
    workloads: dict[int, int] = defaultdict(int)
    roles_by_device: dict[int, list[str]] = defaultdict(list)
    available_by_device: dict[int, int] = {}
    workloads[snapshots["data"].device] += missing_bytes
    workloads[snapshots["cache"].device] += missing_bytes
    workloads[snapshots["processed"].device] += processed_bytes
    for role, snapshot in snapshots.items():
        roles_by_device[snapshot.device].append(role)
        previous = available_by_device.get(snapshot.device)
        if previous is None:
            available_by_device[snapshot.device] = snapshot.available_bytes
        else:
            available_by_device[snapshot.device] = min(
                previous, snapshot.available_bytes
            )

    filesystems: list[dict[str, Any]] = []
    passed = True
    for device in sorted(roles_by_device):
        workload = workloads[device]
        required = workload + reserve_bytes
        available = available_by_device[device]
        device_passed = available >= required
        passed = passed and device_passed
        filesystems.append(
            {
                "device": device,
                "roles": sorted(roles_by_device[device]),
                "available_bytes": available,
                "estimated_workload_bytes": workload,
                "reserve_bytes": reserve_bytes,
                "required_available_bytes": required,
                "passed": device_passed,
            }
        )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "selected_file_count": len(files),
        "planned_source_bytes": planned_bytes,
        "present_exact_file_count": exact_count,
        "present_exact_bytes": exact_bytes,
        "missing_file_count": missing_count,
        "missing_download_bytes": missing_bytes,
        "estimated_cache_download_bytes": missing_bytes,
        "processed_multiplier": processed_multiplier,
        "estimated_processed_bytes": processed_bytes,
        "reserve_bytes_per_filesystem": reserve_bytes,
        "filesystems": filesystems,
        "passed": passed,
        "information_boundary": {
            "file_contents_opened": False,
            "target_outcomes_used": False,
            "confirmation_payloads_opened": False,
            "adaptive_confirmation_payloads_opened": False,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument(
        "--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES
    )
    parser.add_argument(
        "--processed-multiplier",
        type=float,
        default=DEFAULT_PROCESSED_MULTIPLIER,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_capacity_report(
            plan_path=args.plan,
            data_root=args.data_root,
            processed_root=args.processed_root,
            cache_root=args.cache_root,
            reserve_bytes=args.reserve_bytes,
            processed_multiplier=args.processed_multiplier,
        )
    except CapacityContractError as error:
        print(f"capacity contract error: {error}", file=sys.stderr)
        return 1
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
