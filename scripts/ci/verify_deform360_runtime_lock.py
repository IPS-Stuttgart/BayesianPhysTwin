#!/usr/bin/env python3
"""Verify that an isolated Deform360 runtime exactly follows its version lock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin.deform360-runtime-lock-validation"
SCHEMA_VERSION = 1
_EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")


class RuntimeLockError(ValueError):
    """Raised when the runtime lock itself is malformed."""


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> tuple[dict[str, str], str]:
    locked: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_REQUIREMENT.fullmatch(line)
        if match is None:
            raise RuntimeLockError(
                f"lock line {line_number} is not an exact name==version pin"
            )
        name = canonical_name(match.group(1))
        version = match.group(2)
        previous = locked.get(name)
        if previous is not None and previous != version:
            raise RuntimeLockError(f"lock contains conflicting pins for {name}")
        locked[name] = version
    if not locked:
        raise RuntimeLockError("runtime lock does not contain any exact pins")
    return locked, _sha256(path)


def installed_distributions(site: Path) -> dict[str, str]:
    if not site.is_dir() or site.is_symlink():
        raise RuntimeLockError("runtime site must be a real directory")
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            raise RuntimeLockError(
                "installed distribution is missing its Name metadata"
            )
        name = canonical_name(raw_name)
        version = distribution.version
        previous = installed.get(name)
        if previous is not None and previous != version:
            raise RuntimeLockError(
                f"runtime contains conflicting installed versions for {name}"
            )
        installed[name] = version
    if not installed:
        raise RuntimeLockError("runtime site does not contain any distributions")
    return installed


def validate_runtime(
    *,
    lock_path: Path,
    site: Path,
    allowed_local_names: set[str],
    require_complete: bool,
) -> dict[str, Any]:
    locked, lock_sha256 = load_lock(lock_path)
    installed = installed_distributions(site)
    allowed = {canonical_name(name) for name in allowed_local_names}

    local = {name: version for name, version in installed.items() if name in allowed}
    third_party = {
        name: version for name, version in installed.items() if name not in allowed
    }
    unpinned = {
        name: version for name, version in third_party.items() if name not in locked
    }
    mismatched = {
        name: {"installed": version, "locked": locked[name]}
        for name, version in third_party.items()
        if name in locked and version != locked[name]
    }
    missing = (
        {name: version for name, version in locked.items() if name not in third_party}
        if require_complete
        else {}
    )
    missing_local = sorted(allowed - set(local))
    passed = not (unpinned or mismatched or missing or missing_local)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "lock_sha256": lock_sha256,
        "locked_package_count": len(locked),
        "installed_third_party_package_count": len(third_party),
        "local_packages": dict(sorted(local.items())),
        "missing_local_packages": missing_local,
        "unpinned_packages": dict(sorted(unpinned.items())),
        "version_mismatches": dict(sorted(mismatched.items())),
        "missing_locked_packages": dict(sorted(missing.items())),
        "require_complete": require_complete,
        "passed": passed,
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
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--allow-local", action="append", default=[])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_runtime(
            lock_path=args.lock,
            site=args.site,
            allowed_local_names=set(args.allow_local),
            require_complete=args.require_complete,
        )
    except (OSError, RuntimeLockError) as error:
        print(f"runtime lock error: {error}", file=sys.stderr)
        return 1
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
