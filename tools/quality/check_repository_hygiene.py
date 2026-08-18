#!/usr/bin/env python3
"""Reject temporary placeholder artifacts from maintained repository surfaces."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_BASENAMES = frozenset({".agent-placeholder"})
_FORBIDDEN_EXACT_TEXT = frozenset({"placeholder", "temporary placeholder"})
_MAINTAINED_PREFIXES = frozenset({".github", "api", "docs", "src", "tools"})
_MAX_TEXT_PROBE_BYTES = 4096


def tracked_repository_paths(root: Path) -> tuple[str, ...]:
    """Return canonical tracked paths without consulting untracked work products."""

    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(
        sorted(os.fsdecode(value) for value in completed.stdout.split(b"\0") if value)
    )


def find_repository_hygiene_violations(
    root: Path,
    paths: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return deterministic violations for tracked maintained files."""

    repository_root = root.resolve()
    candidates = tracked_repository_paths(repository_root) if paths is None else paths
    violations: list[str] = []
    for raw_path in candidates:
        if type(raw_path) is not str or not raw_path:
            violations.append(f"invalid tracked path: {raw_path!r}")
            continue
        relative = Path(raw_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            violations.append(f"noncanonical tracked path: {raw_path}")
            continue
        if relative.name in _FORBIDDEN_BASENAMES:
            violations.append(f"forbidden placeholder filename: {raw_path}")
            continue
        if not relative.parts or relative.parts[0] not in _MAINTAINED_PREFIXES:
            continue
        full_path = repository_root / relative
        if full_path.is_symlink() or not full_path.is_file():
            continue
        try:
            size = full_path.stat().st_size
        except OSError as error:
            violations.append(f"cannot inspect {raw_path}: {error}")
            continue
        if size > _MAX_TEXT_PROBE_BYTES:
            continue
        try:
            payload = full_path.read_bytes()
        except OSError as error:
            violations.append(f"cannot read {raw_path}: {error}")
            continue
        if b"\0" in payload:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text.strip().casefold() in _FORBIDDEN_EXACT_TEXT:
            violations.append(f"placeholder-only maintained file: {raw_path}")
    return tuple(sorted(violations))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject placeholder-only files from maintained repository paths."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository checkout to inspect",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    root = arguments.root.resolve()
    violations = find_repository_hygiene_violations(root)
    if violations:
        print("Repository hygiene violations:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("Repository hygiene policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
