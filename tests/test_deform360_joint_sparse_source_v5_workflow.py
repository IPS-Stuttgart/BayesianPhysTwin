from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / (
    ".github/workflows/deform360-joint-sparse-source-v5-contracts.yml"
)
DOCUMENT = ROOT / "docs/deform360_joint_sparse_source_execution_v5.md"
MANIFEST = ROOT / "MANIFEST.in"


def test_source_contract_workflow_is_hosted_and_data_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["contracts"]["runs-on"] == "ubuntu-latest"
    assert "workflow_dispatch:" not in text
    assert "self-hosted" not in text
    assert "/mnt/lexar4tb" not in text
    assert "confirmation payload" not in text.lower()
    assert "target outcome" not in text.lower()
    assert "actions/upload-artifact" not in text


def test_document_states_public_measurement_and_approval_boundary() -> None:
    text = " ".join(DOCUMENT.read_text(encoding="utf-8").split())

    assert "These are real-world recordings" in text
    assert "requires no new recording" in text
    assert "or manually supplied approval" in text
    assert "A person cannot waive a failed check" in text
    assert "passing source result authorizes evaluation" in text


def test_source_execution_artifacts_are_in_source_distribution() -> None:
    lines = set(MANIFEST.read_text(encoding="utf-8").splitlines())

    assert "include docs/deform360_joint_sparse_source_execution_v5.md" in lines
    assert (
        "include protocols/locks/"
        "deform360_official_hub_joint_sparse_source_execution_v5.json" in lines
    )
    assert (
        "include scripts/science/evaluate_deform360_joint_sparse_source_v5.py"
        in lines
    )
