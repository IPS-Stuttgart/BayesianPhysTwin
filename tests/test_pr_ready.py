from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.pr_ready import Revisions, command_plan, resolve_revisions


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repository: Path, name: str, content: str) -> str:
    (repository / name).write_text(content, encoding="utf-8")
    _git(repository, "add", name)
    _git(repository, "commit", "-m", f"Write {name}")
    return _git(repository, "rev-parse", "HEAD")


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Readiness Test")
    _git(repository, "config", "user.email", "readiness@example.invalid")
    return repository


def test_default_base_is_merge_base_with_local_main(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    base_sha = _commit(repository, "base.txt", "base\n")
    _git(repository, "switch", "-c", "feature")
    head_sha = _commit(repository, "feature.txt", "feature\n")

    revisions = resolve_revisions(
        repository,
        base_revision=None,
        head_revision="HEAD",
    )

    assert revisions == Revisions(base=base_sha, head=head_sha)


def test_explicit_base_is_resolved_to_an_exact_commit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    base_sha = _commit(repository, "base.txt", "base\n")
    head_sha = _commit(repository, "head.txt", "head\n")

    revisions = resolve_revisions(
        repository,
        base_revision=base_sha[:12],
        head_revision="HEAD",
    )

    assert revisions == Revisions(base=base_sha, head=head_sha)


def test_requested_head_must_match_checked_out_tree(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    base_sha = _commit(repository, "base.txt", "base\n")
    _commit(repository, "head.txt", "head\n")

    with pytest.raises(ValueError, match="does not match"):
        resolve_revisions(
            repository,
            base_revision=base_sha,
            head_revision=base_sha,
        )


def test_command_plan_matches_fast_ci_boundaries() -> None:
    revisions = Revisions(base="a" * 40, head="b" * 40)

    commands = command_plan(revisions)

    assert commands[0] == (
        sys.executable,
        "-m",
        "pre_commit",
        "validate-config",
        ".pre-commit-config.yaml",
    )
    assert commands[1] == (
        "git",
        "diff",
        "--check",
        revisions.base,
        revisions.head,
        "--",
    )
    assert "scripts/ci/check_changed_python.py" in commands[2]
    assert "tools/quality/changed_python_quality.py" in commands[3]
    assert commands[4][-2:] == (
        "tests/test_changed_python_preflight.py",
        "tests/test_pr_ready.py",
    )
