#!/usr/bin/env python3
"""Run the fast local checks that mirror the pull-request quality gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Revisions:
    """Exact revisions used by one local pull-request readiness run."""

    base: str
    head: str


def _git_text(
    repository_root: Path,
    *arguments: str,
    check: bool = True,
) -> str | None:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if check:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(
                f"git {' '.join(arguments)} failed"
                + (f": {detail}" if detail else "")
            )
        return None
    return completed.stdout.strip()


def _resolve_commit(repository_root: Path, revision: str, *, name: str) -> str:
    if not revision or revision.strip() != revision:
        raise ValueError(f"{name} must be a nonempty canonical revision")
    resolved = _git_text(
        repository_root,
        "rev-parse",
        "--verify",
        f"{revision}^{{commit}}",
        check=False,
    )
    if resolved is None:
        raise ValueError(f"{name} is not an available commit: {revision}")
    return resolved


def _default_base(repository_root: Path, head_sha: str) -> str:
    for candidate in ("origin/main", "main"):
        candidate_sha = _git_text(
            repository_root,
            "rev-parse",
            "--verify",
            f"{candidate}^{{commit}}",
            check=False,
        )
        if candidate_sha is None or candidate_sha == head_sha:
            continue
        merge_base = _git_text(
            repository_root,
            "merge-base",
            head_sha,
            candidate_sha,
            check=False,
        )
        if merge_base is not None and merge_base != head_sha:
            return merge_base

    parent_sha = _git_text(
        repository_root,
        "rev-parse",
        "--verify",
        f"{head_sha}^{{commit}}",
        check=False,
    )
    if parent_sha is not None:
        return parent_sha
    raise ValueError(
        "no comparison commit is available; pass --base or fetch origin/main"
    )


def resolve_revisions(
    repository_root: Path,
    *,
    base_revision: str | None,
    head_revision: str,
) -> Revisions:
    """Resolve and validate the exact local comparison revisions."""

    root = repository_root.resolve(strict=True)
    head_sha = _resolve_commit(root, head_revision, name="head revision")
    checkout_sha = _resolve_commit(root, "HEAD", name="repository HEAD")
    if checkout_sha != head_sha:
        raise ValueError(
            "repository HEAD does not match the requested head revision; "
            "refusing to check a different tree"
        )
    if base_revision is None:
        base_sha = _default_base(root, head_sha)
    else:
        base_sha = _resolve_commit(root, base_revision, name="base revision")
    if base_sha == head_sha:
        raise ValueError("base and head revisions must differ")
    return Revisions(base=base_sha, head=head_sha)


def command_plan(revisions: Revisions) -> tuple[tuple[str, ...], ...]:
    """Return the ordered commands executed by the readiness gate."""

    python = sys.executable
    return (
        (
            python,
            "-m",
            "pre_commit",
            "validate-config",
            ".pre-commit-config.yaml",
        ),
        ("git", "diff", "--check", revisions.base, revisions.head, "--"),
        (
            python,
            "scripts/ci/check_changed_python.py",
            "--base",
            revisions.base,
            "--head",
            revisions.head,
        ),
        (
            python,
            "tools/quality/changed_python_quality.py",
            "--base",
            revisions.base,
            "--head",
            revisions.head,
        ),
        (
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_changed_python_preflight.py",
            "tests/test_pr_ready.py",
        ),
    )


def run_pr_ready(repository_root: Path, revisions: Revisions) -> None:
    """Execute the fast readiness commands in their registered order."""

    for command in command_plan(revisions):
        print(f"\n==> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=repository_root, check=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=None,
        help="comparison revision; defaults to merge-base with origin/main or main",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="reviewed head revision; must match the checked-out HEAD",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root; defaults to the checkout containing this script",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        revisions = resolve_revisions(
            arguments.repository_root,
            base_revision=arguments.base,
            head_revision=arguments.head,
        )
        print(
            f"PR-ready comparison: {revisions.base}..{revisions.head}",
            flush=True,
        )
        run_pr_ready(arguments.repository_root, revisions)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"PR-ready checks failed: {error}", file=sys.stderr)
        return 1
    print("\nPR-ready checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
