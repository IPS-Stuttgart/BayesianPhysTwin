"""Regression tests for source-version and Git-tag identity."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/release/check_version_identity.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_version_identity", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_versions(
    root: Path,
    *,
    project: str,
    citation: str | None = None,
) -> None:
    (root / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools"]\n\n'
        "[project]\n"
        'name = "bayesian-phystwin"\n'
        f'version = "{project}"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        f'version: "{citation or project}"\n',
        encoding="utf-8",
    )


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "pyproject.toml", "CITATION.cff")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _repository(tmp_path: Path, *, version: str = "0.4.0") -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.name", "Release Test")
    _git(root, "config", "user.email", "release-test@example.invalid")
    _write_versions(root, project=version)
    _commit(root, "initial release candidate")
    return root


def test_untagged_final_version_is_a_release_candidate(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    report = tool.validate_version_identity(
        root,
        require_complete_history=True,
    )

    assert report["status"] == "untagged-release-candidate"
    assert report["release_tag"] == "v0.4.0"
    assert report["tagged_revision"] is None
    assert report["final_version"] is True


def test_exact_tagged_release_is_accepted(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    _git(root, "tag", "-a", "v0.4.0", "-m", "BayesianPhysTwin 0.4.0")

    report = tool.validate_version_identity(
        root,
        expected_tag="v0.4.0",
        require_complete_history=True,
    )

    assert report["status"] == "exact-tagged-release"
    assert report["tagged_revision"] == head
    assert report["head_revision"] == head
    assert report["tag_required"] is True


def test_reusing_a_tagged_version_on_a_new_commit_is_rejected(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    tagged = _git(root, "rev-parse", "HEAD")
    _git(root, "tag", "v0.4.0")
    (root / "README.md").write_text("changed after release\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "post-release change")

    with pytest.raises(tool.VersionIdentityError, match="reuses 'v0.4.0'"):
        tool.validate_version_identity(root, require_complete_history=True)

    assert _git(root, "rev-parse", "v0.4.0^{commit}") == tagged


def test_post_release_development_version_has_distinct_identity(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    _git(root, "tag", "v0.4.0")
    _write_versions(root, project="0.4.1.dev0")
    head = _commit(root, "open the next development line")

    report = tool.validate_version_identity(
        root,
        require_complete_history=True,
    )

    assert report["project_version"] == "0.4.1.dev0"
    assert report["release_tag"] == "v0.4.1.dev0"
    assert report["tagged_revision"] is None
    assert report["head_revision"] == head
    assert report["status"] == "development-version"
    assert report["final_version"] is False


def test_project_and_citation_versions_must_match(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _write_versions(root, project="0.4.1.dev0", citation="0.4.0")

    with pytest.raises(tool.VersionIdentityError, match="does not match"):
        tool.validate_version_identity(root)


def test_expected_tag_must_be_canonical_and_available(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    with pytest.raises(tool.VersionIdentityError, match="must be 'v0.4.0'"):
        tool.validate_version_identity(root, expected_tag="v0.4.1")
    with pytest.raises(tool.VersionIdentityError, match="is not available"):
        tool.validate_version_identity(root, expected_tag="v0.4.0")


def test_duplicate_literal_project_versions_are_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8") + 'version = "0.4.1"\n',
        encoding="utf-8",
    )

    with pytest.raises(tool.VersionIdentityError, match="exactly one literal"):
        tool.validate_version_identity(root)


def test_quality_ratchet_runs_release_identity_check() -> None:
    quality = (ROOT / "tools/quality/changed_python_quality.py").read_text(
        encoding="utf-8"
    )

    assert '"tools/release/check_version_identity.py"' in quality
    assert '"--head",\n            checkout_head' in quality
    assert '"--require-complete-history"' in quality
    assert "Release version and Git tag identity" in quality
