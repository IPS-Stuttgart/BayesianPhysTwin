"""Run Ruff lint and formatting checks on Python files changed by a Git event."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

_ZERO_SHA = "0" * 40
_BATCH_SIZE = 128
_PYTHON_SUFFIXES = frozenset({".py", ".pyi"})


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository_root), *arguments],
        text=True,
    )


def _commit_exists(repository_root: Path, revision: str) -> bool:
    if not revision or revision == _ZERO_SHA:
        return False
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "cat-file",
            "-e",
            f"{revision}^{{commit}}",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _comparison_base(
    repository_root: Path,
    *,
    base: str,
    head: str,
) -> str | None:
    if _commit_exists(repository_root, base):
        merge_base = _git_output(
            repository_root,
            "merge-base",
            base,
            head,
        ).strip()
        if merge_base:
            return merge_base
    parent = f"{head}^"
    return parent if _commit_exists(repository_root, parent) else None


def changed_python_files(
    repository_root: Path,
    *,
    base: str = "",
    head: str = "HEAD",
) -> tuple[str, ...]:
    """Return existing changed Python paths relative to ``repository_root``."""

    resolved_root = repository_root.resolve()
    resolved_base = _comparison_base(
        resolved_root,
        base=base,
        head=head,
    )
    if resolved_base is None:
        output = _git_output(resolved_root, "ls-files")
    else:
        output = _git_output(
            resolved_root,
            "diff",
            "--name-only",
            "--diff-filter=ACMRT",
            resolved_base,
            head,
        )

    paths: set[str] = set()
    for raw_path in output.splitlines():
        relative = Path(raw_path)
        if relative.suffix not in _PYTHON_SUFFIXES or relative.is_absolute():
            continue
        candidate = (resolved_root / relative).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            continue
        paths.add(relative.as_posix())
    return tuple(sorted(paths))


def _batches(values: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), _BATCH_SIZE):
        yield values[start : start + _BATCH_SIZE]


def _run_ruff(
    repository_root: Path,
    files: Sequence[str],
    *arguments: str,
) -> None:
    for batch in _batches(files):
        subprocess.run(
            [sys.executable, "-m", "ruff", *arguments, *batch],
            cwd=repository_root,
            check=True,
        )


def check_changed_python_quality(
    repository_root: Path,
    *,
    base: str = "",
    head: str = "HEAD",
) -> tuple[str, ...]:
    """Run Ruff lint and format checks on changed Python files and return them."""

    files = changed_python_files(repository_root, base=base, head=head)
    if not files:
        print("No changed Python files require quality checks.")
        return files

    print("Checking changed Python files with Ruff:")
    for path in files:
        print(f"  {path}")
    _run_ruff(repository_root, files, "check")
    _run_ruff(repository_root, files, "format", "--check")
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Git worktree whose changed Python files should be checked",
    )
    parser.add_argument(
        "--base",
        default="",
        help="base revision; falls back to the head parent when unavailable",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="head revision to compare and quality-check",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_changed_python_quality(
        args.repository_root,
        base=args.base,
        head=args.head,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
