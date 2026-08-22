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
CURRENT_ACTIONS = ROOT / "api/ecosystem-current-actions-v1.json"
ARCHIVED_RUNNER = (
    ROOT / "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
ARCHIVED_RUNNER_BLOB_SHA = "42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
SOURCE_ACTION_ID = "covariance-only-independent-confirmation"


def _workflow_payload() -> dict[str, Any]:
    return dict(yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


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
    workflow = _workflow_payload()
    action = _current_source_action()

    assert action["owning_repository"] == "IPS-Stuttgart/BayesianPhysTwin"
    assert action["status"] == "source-gate-pending"
    assert action["target_access"] == "closed"
    issue_number = str(action["owning_issue"])
    assert workflow["env"]["ISSUE_NUMBER"] == issue_number

    publish_steps = [
        step
        for step in _workflow_steps()
        if step.get("name") == f"Publish bounded source receipt to issue {issue_number}"
    ]
    assert len(publish_steps) == 1
    publish_script = publish_steps[0]["run"]
    assert 'os.environ["ISSUE_NUMBER"]' in publish_script
    assert "development suffix opened: `false`" in publish_script
    assert "v6 target payloads opened: `false`" in publish_script
    assert "v6 target outcomes used: `false`" in publish_script
