from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "deform360-covariance-residual-adapter-v1.yml"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_is_hosted_read_only_and_target_inaccessible() -> None:
    text = _workflow_text()

    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: [self-hosted" not in text
    assert "permissions:\n  contents: read" in text
    assert "${{ secrets." not in text
    assert "${{ vars." not in text
    assert "workstation2" in text
    assert 'test "${RUNNER_NAME}" != "workstation2"' in text
    assert "test ! -e /mnt/lexar4tb" in text
    assert "unopened-candidate-target" not in text


def test_workflow_checks_out_exact_source_and_runs_locked_surface() -> None:
    text = _workflow_text()

    assert "workflow_dispatch:" in text
    assert "github.event.pull_request.head.sha || github.sha" in text
    assert "ref: ${{ env.SOURCE_SHA }}" in text
    assert "fetch-depth: 1" in text
    assert "persist-credentials: false" in text
    assert "clean: true" in text
    assert "deform360_covariance_residual_adapter_dry_run_v1.json" in text
    assert "run_deform360_covariance_residual_adapter_dry_run_v1.py" in text
    assert "python -m ruff check" in text
    assert "python -m ruff format --check" in text
    assert "python -m mypy" in text
    assert "python -m pytest -q" in text


def test_workflow_pins_actions_and_uploads_only_compact_evidence() -> None:
    text = _workflow_text()

    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert "target_payload_opened" in text
    assert "target_outcomes_opened" in text
    assert "claim_authorized" in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 30" in text
