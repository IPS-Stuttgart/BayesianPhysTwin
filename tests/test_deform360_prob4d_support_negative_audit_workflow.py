from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(
    ".github/workflows/revalidate-deform360-prob4d-support-negative-once.yml"
)
AUDITOR = Path("scripts/science/audit_deform360_prob4d_support_negative.py")


def test_support_negative_auditor_is_exact_artifact_bound_and_target_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert document["on"] == {
        "push": {
            "branches": ["main"],
            "paths": [
                ".github/workflows/"
                "revalidate-deform360-prob4d-support-negative-once.yml",
                "scripts/science/audit_deform360_prob4d_support_negative.py",
            ],
        },
        "workflow_dispatch": "",
    }
    assert 'SOURCE_RUN_ID: "31297018948"' in text
    assert 'SOURCE_ARTIFACT_ID: "9033414269"' in text
    assert "SOURCE_ARTIFACT_DIGEST: sha256:7247a2a26050" in text
    assert "SOURCE_HEAD_SHA: ded8910becbb" in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: self-hosted" not in text
    assert "actions: read" in text
    assert "contents: read" in text
    assert "issues: write" in text
    assert "/mnt/lexar4tb/datasets/deform360/adaptive-confirmation" not in text


def test_support_negative_auditor_reconstructs_the_early_terminal() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    auditor = AUDITOR.read_text(encoding="utf-8")

    locate = workflow.index("Locate the exact source run and compact artifact")
    download = workflow.index("Download only the immutable compact source artifact")
    validate = workflow.index(
        "Independently validate the early support-negative terminal"
    )
    upload = workflow.index("Upload the independent support-negative audit")
    publish = workflow.index("Publish the corrected terminal decision")
    enforce = workflow.index("Enforce the independently validated terminal")
    assert locate < download < validate < upload < publish < enforce
    assert "continue-on-error: true" in workflow
    assert "if: steps.validate.outcome == 'success'" in workflow
    assert 'test "${{ steps.validate.outcome }}" = "success"' in workflow

    assert '"metric-support/metric-batch-result.json"' in auditor
    assert '"metric-support/support-receipt.json"' in auditor
    assert '"support_gate": "failure"' in auditor
    assert '"samples": "skipped"' in auditor
    assert '"calibration": "skipped"' in auditor
    assert '"source_gate": "skipped"' in auditor
    assert '"supported_stream_count": 313' in auditor
    assert '"support_negative_stream_count": 11' in auditor
    assert '"technical_failure_stream_count": 0' in auditor
    assert '"audit_status": "validated-negative"' in auditor
    assert '"terminal_stage": "support-gate"' in auditor
    assert '"source_gate_passed": False' in auditor
    assert '"confirmation_access_authorized": False' in auditor
    assert "source file roster changed" in auditor
    assert "result ID changed" in auditor
    assert "support-negative roster changed" in auditor


def test_support_negative_auditor_publishes_once_and_never_authorizes_confirmation(
) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "deform360-prob4d-support-negative-independent-audit-v1" in text
    assert "The immutable support-negative audit is already published." in text
    assert "overall source gate passed: \\`false\\`" in text
    assert "confirmation access authorized: \\`false\\`" in text
    assert "No stream was replaced" in text
    assert "adaptive-confirmation or target payload was opened" in text
    expected_enforcement = (
        'test "$(jq -r \' .audit_status\' "${receipt}")" = "validated-negative"'
    ).replace("' .audit", "'.audit")
    assert expected_enforcement in text
