"""Reject repository-wide Ruff regressions relative to the Git merge base."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

_ZERO_SHA = "0" * 40


class RuffDiagnostic(dict[str, object]):
    """Validated JSON object emitted by Ruff for one diagnostic."""


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository_root), *arguments],
        text=True,
    ).strip()


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
) -> str:
    if _commit_exists(repository_root, base):
        merge_base = _git_output(repository_root, "merge-base", base, head)
        if merge_base:
            return merge_base
    parent = f"{head}^"
    if _commit_exists(repository_root, parent):
        return _git_output(repository_root, "rev-parse", parent)
    raise ValueError("a comparison base is required for repository-wide Ruff gating")


def _relative_path(repository_root: Path, filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Ruff reported a path outside the repository: {path}"
            ) from error
    return path.as_posix()


def _existing_paths(repository_root: Path, paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(path for path in paths if (repository_root / path).exists())


def _run_ruff(
    repository_root: Path,
    paths: Sequence[str],
) -> list[RuffDiagnostic]:
    existing_paths = _existing_paths(repository_root, paths)
    if not existing_paths:
        return []
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *existing_paths,
            "--output-format=json",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"Ruff failed before producing diagnostics (exit {completed.returncode})"
        )

    raw_payload: object = json.loads(completed.stdout or "[]")
    if not isinstance(raw_payload, list) or any(
        not isinstance(item, dict) for item in raw_payload
    ):
        raise ValueError("Ruff JSON output has an unexpected shape")
    return [RuffDiagnostic(cast(dict[str, object], item)) for item in raw_payload]


def _diagnostic_counts(
    repository_root: Path,
    diagnostics: Sequence[RuffDiagnostic],
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], str]]:
    counts: Counter[tuple[str, str]] = Counter()
    messages: dict[tuple[str, str], str] = {}
    for diagnostic in diagnostics:
        filename = diagnostic.get("filename")
        code = diagnostic.get("code")
        message = diagnostic.get("message")
        if not isinstance(filename, str) or not isinstance(code, str):
            raise ValueError("Ruff diagnostic is missing filename or code")
        key = (_relative_path(repository_root, filename), code)
        counts[key] += 1
        if isinstance(message, str):
            messages.setdefault(key, message)
    return counts, messages


@contextmanager
def _temporary_worktree(
    repository_root: Path,
    revision: str,
    *,
    label: str,
) -> Iterator[Path]:
    with TemporaryDirectory(prefix=f"bpt-ruff-{label}-") as directory:
        worktree = Path(directory) / "worktree"
        subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "worktree",
                "add",
                "--detach",
                "--force",
                str(worktree),
                revision,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        try:
            yield worktree
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "-C", str(repository_root), "worktree", "prune"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def _write_report(path: Path | None, diagnostics: Sequence[RuffDiagnostic]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_ruff_regression(
    repository_root: Path,
    *,
    base: str = "",
    head: str = "HEAD",
    paths: Sequence[str] = ("src", "tests", "tools"),
    baseline_report_path: Path | None = None,
    current_report_path: Path | None = None,
) -> int:
    """Return zero when no path/code diagnostic count exceeds the merge base."""

    resolved_root = repository_root.resolve()
    baseline_revision = _comparison_base(resolved_root, base=base, head=head)
    head_revision = _git_output(resolved_root, "rev-parse", head)
    with _temporary_worktree(
        resolved_root,
        baseline_revision,
        label="baseline",
    ) as baseline_root:
        baseline_diagnostics = _run_ruff(baseline_root, paths)
        baseline_counts, _ = _diagnostic_counts(
            baseline_root,
            baseline_diagnostics,
        )
    with _temporary_worktree(
        resolved_root,
        head_revision,
        label="current",
    ) as current_root:
        current_diagnostics = _run_ruff(current_root, paths)
        current_counts, messages = _diagnostic_counts(
            current_root,
            current_diagnostics,
        )

    resolved_baseline_report = (
        None
        if baseline_report_path is None
        else (
            baseline_report_path
            if baseline_report_path.is_absolute()
            else resolved_root / baseline_report_path
        )
    )
    resolved_current_report = (
        None
        if current_report_path is None
        else (
            current_report_path
            if current_report_path.is_absolute()
            else resolved_root / current_report_path
        )
    )
    _write_report(resolved_baseline_report, baseline_diagnostics)
    _write_report(resolved_current_report, current_diagnostics)

    unexpected = current_counts - baseline_counts
    removed = baseline_counts - current_counts
    print(
        "Repository-wide Ruff comparison: "
        f"base={baseline_revision[:12]}, "
        f"head={head_revision[:12]}, "
        f"baseline={sum(baseline_counts.values())}, "
        f"current={sum(current_counts.values())}, "
        f"removed={sum(removed.values())}, "
        f"unexpected={sum(unexpected.values())}"
    )
    if unexpected:
        print("New or increased Ruff diagnostics:", file=sys.stderr)
        for (path, code), count in sorted(unexpected.items()):
            message = messages.get((path, code), "")
            suffix = f": {message}" if message else ""
            print(f"  {path}: {code} x{count}{suffix}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository whose Ruff diagnostics should be compared",
    )
    parser.add_argument(
        "--base",
        default="",
        help="base revision; falls back to the head parent when unavailable",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="head revision whose repository-wide diagnostics should be checked",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="optional JSON report for merge-base diagnostics",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON report for current diagnostics",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src", "tests", "tools"],
        help="repository paths passed to Ruff in both revisions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return check_ruff_regression(
        cast(Path, args.repository_root),
        base=cast(str, args.base),
        head=cast(str, args.head),
        paths=cast(list[str], args.paths),
        baseline_report_path=cast(Path | None, args.baseline_report),
        current_report_path=cast(Path | None, args.report),
    )


if __name__ == "__main__":
    raise SystemExit(main())
