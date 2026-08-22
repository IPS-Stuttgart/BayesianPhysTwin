from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)
ROUTER_WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-receipt-routing.yml"
ROUTER_TOOL = ROOT / "tools/quality/route_deform360_v6_source_receipt.py"
CURRENT_ACTIONS = ROOT / "api/ecosystem-current-actions-v1.json"
ARCHIVED_RUNNER = (
    ROOT / "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
ARCHIVED_RUNNER_BLOB_SHA = "42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
SOURCE_ACTION_ID = "covariance-only-independent-confirmation"


def _workflow_payload() -> dict[str, Any]:
    return dict(yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def _router_payload() -> dict[str, Any]:
    return dict(yaml.safe_load(ROUTER_WORKFLOW.read_text(encoding="utf-8")))


def _workflow_steps() -> list[dict[str, Any]]:
    return list(_workflow_payload()["jobs"]["evidence"]["steps"])


def _current_source_action() -> dict[str, Any]:
    payload = json.loads(CURRENT_ACTIONS.read_text(encoding="utf-8"))
    matches = [
        action
        for action in payload["actions"]
        if action["action_id"] == SOURCE_ACTION_ID
    ]
    assert len(matches) == 1
    return dict(matches[0])


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def test_dual_runtime_source_execution_exports_archived_inline_roots() -> None:
    matches = [
        step
        for step in _workflow_steps()
        if step.get("name") == "Generate real prefix-only source prediction evidence"
    ]
    assert len(matches) == 1
    environment = matches[0]["env"]
    run_root = (
        "${{ env.RESULTS_ROOT }}/bayesian-phystwin/"
        "deform360-v6-source-prediction/${{ env.AMENDMENT_ID }}/"
        "${{ github.sha }}"
    )

    assert environment["RUN_ROOT"] == run_root
    assert environment["PREDICTION_ROOT"] == f"{run_root}/prediction-panel"
    assert _git_blob_sha(ARCHIVED_RUNNER) == ARCHIVED_RUNNER_BLOB_SHA


def test_dual_runtime_source_receipt_routes_to_current_owning_issue() -> None:
    router = _router_payload()
    action = _current_source_action()

    assert action["owning_repository"] == "IPS-Stuttgart/BayesianPhysTwin"
    assert action["status"] == "source-gate-pending"
    assert action["target_access"] == "closed"
    issue_number = str(action["owning_issue"])
    assert router["env"]["ISSUE_NUMBER"] == issue_number
    assert router["permissions"] == {
        "actions": "read",
        "contents": "read",
        "issues": "write",
    }
    assert ROUTER_WORKFLOW.read_text(encoding="utf-8").startswith(
        "# workflow-lifecycle: permanent\n"
        "# workflow-owner: IPS-Stuttgart maintainers\n"
    )

    route_job = router["jobs"]["route"]
    condition = route_job["if"]
    assert "head_branch == 'main'" in condition
    assert "workflow_run.event == 'push'" in condition
    assert "workflow_run.event == 'workflow_dispatch'" in condition

    steps = route_job["steps"]
    checkouts = [
        step
        for step in steps
        if step.get("name") == "Check out trusted router revision"
    ]
    assert len(checkouts) == 1
    checkout = checkouts[0]
    assert checkout["uses"] == (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    )
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["persist-credentials"] is False

    downloads = [
        step
        for step in steps
        if step.get("name") == "Download exact compact source artifact"
    ]
    assert len(downloads) == 1
    download = downloads[0]["with"]
    assert "github.event.workflow_run.id" in download["name"]
    assert "github.event.workflow_run.run_attempt" in download["name"]
    assert download["run-id"] == "${{ github.event.workflow_run.id }}"

    route_steps = [
        step
        for step in steps
        if step.get("name")
        == f"Verify and route bounded receipt to issue {issue_number}"
    ]
    assert len(route_steps) == 1
    route_step = route_steps[0]
    assert route_step["run"] == (
        "set -euo pipefail\n"
        "python tools/quality/route_deform360_v6_source_receipt.py\n"
    )
    assert route_step["env"]["SOURCE_HEAD_SHA"] == (
        "${{ github.event.workflow_run.head_sha }}"
    )
    assert ROUTER_TOOL.is_file()
