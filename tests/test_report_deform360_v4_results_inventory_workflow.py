from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/report-deform360-v4-results-inventory-once.yml"


def test_receipt_workflow_is_push_only_and_write_bounded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request:" not in text
    assert "pull_request_target" not in text
    assert "push:\n    branches: [main]\n    paths:" in text
    assert "workflow_dispatch:" in text
    assert "permissions:\n  contents: read\n  issues: write" in text
    assert "contents: write" not in text
    assert "actions: write" not in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text


def test_receipt_workflow_uses_only_results_tree_on_workstation2() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: self-hosted" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert '--root "${DEFORM360_RESULTS_ROOT}"' in text
    assert '--forbidden-root "${DEFORM360_OFFICIAL_RAW_ROOT}"' in text
    assert '--forbidden-root "${DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT}"' in text
    assert "binary_scientific_payloads_loaded=false" in text
    assert "raw_dataset_payloads_opened=false" in text
    assert "adaptive_confirmation_payloads_opened=false" in text
    assert "confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "replacement_allowed=false" in text


def test_receipt_workflow_does_not_modify_runner_environment() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pip install" not in text
    assert "actions/setup-python" not in text
    assert "command -v python3 || command -v python" in text
    assert "scripts/ci/inventory_deform360_v4_results.py" in text
    assert "maximum-candidates 50000" in text
    assert "len(shortlist) >= 40" in text
    assert "len(body.encode(\"utf-8\")) >= 60_000" in text


def test_receipt_workflow_publishes_traceable_compact_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/upload-artifact@v7" in text
    assert "deform360-v4-results-inventory-receipt-${GITHUB_RUN_ID}" in text
    assert 'ISSUE_NUMBER: "148"' in text
    assert "workflow_run_id" in text
    assert "source_revision" in text
    assert "inventory_id" in text
    assert "receipt_id" in text
    assert "candidate_schema_counts" in text
    assert "observation_artifact_id" in text
    assert "linearization_artifact_id" in text
    assert "gauge_tree_prior_artifact_id" in text
    assert "urllib.request.Request" in text
    assert "issues/{issue}/comments" in text
    assert "response.status != 201" in text
