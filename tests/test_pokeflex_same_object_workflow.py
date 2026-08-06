from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/pokeflex-same-object-paper-artifacts.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _artifact_job(text: str) -> str:
    marker = "\n  artifacts:\n"
    assert marker in text
    return text.split(marker, maxsplit=1)[1]


def test_hosted_source_preflight_stays_separate() -> None:
    text = _workflow_text()
    hosted = text.split("\n  artifacts:\n", maxsplit=1)[0]
    assert "runs-on: ubuntu-latest" in hosted
    assert "uses: actions/setup-python@" in hosted
    assert "contents: read" in text


def test_self_hosted_runtime_uses_a_fresh_virtual_environment() -> None:
    artifacts = _artifact_job(_workflow_text())
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in artifacts
    assert "uses: actions/setup-python@" not in artifacts
    assert 'echo "POKEFLEX_VENV=${root}/venv"' in artifacts
    assert '"${POKEFLEX_BOOTSTRAP_PYTHON}" -m venv --clear' in artifacts
    assert '"${POKEFLEX_VENV}/bin/python" -m ensurepip --upgrade' in artifacts
    assert "--no-cache-dir" in artifacts
    assert re.search(r"\n\s+python -m pip install", artifacts) is None


def test_input_custody_is_checked_before_heavy_runtime_installation() -> None:
    artifacts = _artifact_job(_workflow_text())
    preflight = artifacts.index(
        "Validate frozen input custody before runtime installation"
    )
    runtime = artifacts.index("Create isolated released-checkpoint runtime")
    assert preflight < runtime
    assert "selected_archive_size_matches" in artifacts
    assert "selected_extracted_take" in artifacts
    assert "input-preflight.json" in artifacts


def test_failure_reporting_does_not_require_a_completed_runtime() -> None:
    artifacts = _artifact_job(_workflow_text())
    assert 'if [[ -n "${BPT_PYTHON:-}" && -x "${BPT_PYTHON}" ]]' in artifacts
    assert "if-no-files-found: warn" in artifacts
    assert "${{ env.POKEFLEX_EVIDENCE }}/input-preflight.json" in artifacts
