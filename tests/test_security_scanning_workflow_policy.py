from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "security-scanning.yml"
ACTION_REFERENCE = re.compile(r"^\s*uses:\s+([^\s@]+)@([^\s#]+)", re.MULTILINE)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_security_scanning_has_read_only_triggers_and_permissions() -> None:
    text = _workflow_text()

    assert "push:" in text
    assert "pull_request:" in text
    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "persist-credentials: false" in text
    assert "continue-on-error: true" not in text
    assert "${{ secrets." not in text


def test_codeql_scans_python_and_workflow_sources() -> None:
    text = _workflow_text()

    assert "language: [python, actions]" in text
    assert "build-mode: none" in text
    assert "queries: security-extended" in text
    assert "security-events: write" in text
    assert "github/codeql-action/init@" in text
    assert "github/codeql-action/analyze@" in text
    assert 'category: "/language:${{ matrix.language }}"' in text


def test_dependency_audit_is_strict_pinned_and_archived() -> None:
    text = _workflow_text()

    assert 'python -m pip install "pip-audit==2.10.1"' in text
    assert "python -m pip_audit" in text
    assert "--strict" in text
    assert "--progress-spinner off" in text
    assert "--format json" in text
    assert "--output pip-audit.json" in text
    assert "if: always()" in text
    assert "actions/upload-artifact@" in text
    assert "if-no-files-found: warn" in text


def test_all_third_party_actions_are_pinned_to_full_commit_shas() -> None:
    references = ACTION_REFERENCE.findall(_workflow_text())

    assert references
    for action, revision in references:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{action} must use a full lowercase commit SHA, got {revision!r}"
        )
