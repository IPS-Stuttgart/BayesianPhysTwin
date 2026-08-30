"""Tests for exact, inactive retired GitHub Actions archives."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/quality/check_retired_workflow_archive.py"
MANIFEST_PATH = Path("archive/github-actions/retired-one-shot-v1/manifest.json")


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_retired_workflow_archive",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    archive = root / "archive/github-actions/retired-one-shot-v1"
    contract_tests = archive / "contract-tests"
    archive.mkdir(parents=True)
    contract_tests.mkdir()
    workflow = b"name: retired\n"
    historical_test = b"def test_historical():\n    assert True\n"
    workflow_path = archive / "launch-source-once.yml"
    test_path = contract_tests / "test_launch_source.py"
    workflow_path.write_bytes(workflow)
    test_path.write_bytes(historical_test)
    manifest = {
        "schema": "bayesian-phystwin.retired-github-actions",
        "schema_version": 1,
        "retired_from_revision": "a" * 40,
        "retired_workflow_count": 1,
        "retired_workflow_bytes": len(workflow),
        "scientific_boundary": "archive-only test fixture",
        "workflows": [
            {
                "original_path": ".github/workflows/launch-source-once.yml",
                "archived_path": (
                    "archive/github-actions/retired-one-shot-v1/launch-source-once.yml"
                ),
                "git_blob_sha1": _git_blob_sha1(workflow),
                "byte_count": len(workflow),
            }
        ],
        "contract_tests": [
            {
                "original_path": "tests/test_launch_source.py",
                "archived_path": (
                    "archive/github-actions/retired-one-shot-v1/"
                    "contract-tests/test_launch_source.py"
                ),
                "git_blob_sha1": _git_blob_sha1(historical_test),
            }
        ],
    }
    (archive / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def test_exact_inactive_archive_passes(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    report = tool.validate_repository(root)

    assert report == {
        "schema": "bayesian-phystwin.retired-github-actions",
        "schema_version": 1,
        "retired_from_revision": "a" * 40,
        "retired_workflow_count": 1,
        "retired_workflow_bytes": 14,
        "archived_contract_test_count": 1,
        "active_original_path_count": 0,
        "status": "exact-inactive-archive",
    }


def test_changed_archived_workflow_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    source = root / "archive/github-actions/retired-one-shot-v1/launch-source-once.yml"
    source.write_text("name: changed\n", encoding="utf-8")

    with pytest.raises(
        tool.RetiredWorkflowArchiveError,
        match="archived workflow.*changed",
    ):
        tool.validate_repository(root)


def test_reactivated_original_path_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    active = root / ".github/workflows/launch-source-once.yml"
    active.parent.mkdir(parents=True)
    active.write_text("name: reactivated\n", encoding="utf-8")

    with pytest.raises(
        tool.RetiredWorkflowArchiveError,
        match="became active again",
    ):
        tool.validate_repository(root)


def test_changed_archived_contract_test_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    source = (
        root / "archive/github-actions/retired-one-shot-v1/"
        "contract-tests/test_launch_source.py"
    )
    source.write_text("def test_changed():\n    assert False\n", encoding="utf-8")

    with pytest.raises(
        tool.RetiredWorkflowArchiveError,
        match="contract-test Git blob changed",
    ):
        tool.validate_repository(root)


def test_checked_in_archive_is_exact_and_inactive() -> None:
    report = tool.validate_repository(ROOT, MANIFEST_PATH)

    assert report["retired_workflow_count"] == 15
    assert report["retired_workflow_bytes"] == 126_762
    assert report["archived_contract_test_count"] == 10
    assert report["active_original_path_count"] == 0
    assert report["status"] == "exact-inactive-archive"
