#!/usr/bin/env python3
"""Run inexpensive source checks on Python files changed between two revisions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_CHUNK_SIZE = 100


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=None,
    )


def _resolve_revision(repository_root: Path, revision: str, *, name: str) -> str:
    if not revision or revision.strip() != revision:
        raise ValueError(f"{name} must be a nonempty canonical revision")
    try:
        result = _run_git(
            repository_root,
            ["rev-parse", "--verify", f"{revision}^{{commit}}"],
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{name} is not an available commit: {revision}") from error
    return result.stdout.decode("ascii").strip()


def changed_python_files(
    repository_root: Path,
    *,
    base_revision: str,
    head_revision: str,
) -> tuple[Path, ...]:
    """Return ordinary repository-local Python files changed in one exact diff."""

    root = repository_root.resolve(strict=True)
    base_sha = _resolve_revision(root, base_revision, name="base_revision")
    head_sha = _resolve_revision(root, head_revision, name="head_revision")
    checkout_sha = _resolve_revision(root, "HEAD", name="repository HEAD")
    if checkout_sha != head_sha:
        raise ValueError(
            "repository HEAD does not match head_revision; "
            "refusing to check files from a different tree"
        )
    result = _run_git(
        root,
        [
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            base_sha,
            head_sha,
            "--",
            "*.py",
        ],
        capture_output=True,
    )
    candidates: set[Path] = set()
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"git reported a nonlocal path: {name}")
        source = root / relative
        try:
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"changed path is not readable: {name}") from error
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"changed path escapes the repository: {name}") from error
        if source.is_symlink():
            raise ValueError(f"changed Python path must not be a symlink: {name}")
        if not source.is_file():
            raise ValueError(f"changed Python path is not an ordinary file: {name}")
        candidates.add(relative)
    return tuple(sorted(candidates, key=lambda path: path.as_posix()))


def _chunks(values: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _checked_paths(paths: Sequence[Path]) -> list[str]:
    return [f"./{path.as_posix()}" for path in paths]


def run_changed_python_checks(
    repository_root: Path,
    paths: Sequence[Path],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Run Ruff lint/format and byte compilation on the selected files."""

    root = repository_root.resolve(strict=True)
    if not paths:
        print("No changed Python files require the fast source preflight.")
        return
    print(f"Fast source preflight: {len(paths)} changed Python file(s)")
    for path in paths:
        print(f"  {path.as_posix()}")
    for chunk in _chunks(tuple(paths), chunk_size):
        rendered = _checked_paths(chunk)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--output-format=github",
                *rendered,
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "--diff",
                *rendered,
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "py_compile", *rendered],
            cwd=root,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Exact base commit SHA")
    parser.add_argument("--head", required=True, help="Exact head commit SHA")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository root; defaults to the current directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum files passed to each checker invocation",
    )
    args = parser.parse_args()
    try:
        paths = changed_python_files(
            args.repository_root,
            base_revision=args.base,
            head_revision=args.head,
        )
        run_changed_python_checks(
            args.repository_root,
            paths,
            chunk_size=args.chunk_size,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"fast source preflight failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
