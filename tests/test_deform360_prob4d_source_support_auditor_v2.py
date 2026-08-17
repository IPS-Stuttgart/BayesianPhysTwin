from __future__ import annotations

from pathlib import Path

import yaml

AUDITOR = Path(
    ".github/workflows/revalidate-deform360-prob4d-source-support-negative-v2.yml"
)


def _text() -> str:
    return AUDITOR.read_text(encoding="utf-8")


def test_v2_auditor_is_exact_artifact_bound_and_target_closed() -> None:
    text = _text()
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" in text
    assert "branches: [main]" in text
    assert 'SOURCE_RUN_ID: "31297018948"' in text
    assert 'SOURCE_RUN_ATTEMPT: "1"' in text
    assert "SOURCE_RUN_CONCLUSION: failure" in text
    assert "SOURCE_HEAD_SHA: ded8910becbb" in text
    assert 'SOURCE_ARTIFACT_ID: "9033414269"' in text
    assert "SOURCE_ARTIFACT_DIGEST: sha256:7247a2a26050" in text
    assert "VALIDATOR_REVISION: 94913b23c31e" in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: self-hosted" not in text
    assert "actions: read" in text
    assert "contents: read" in text
    assert "issues: write" in text
    assert '"confirmation_payloads_opened": False' in text
    assert '"target_outcomes_used": False' in text
    assert '"future_frames_used": False' in text
    assert '"replacement_allowed": False' in text


def test_v2_auditor_reconstructs_the_early_support_terminal() -> None:
    text = _text()

    verify = text.index("Verify the exact source run and compact artifact")
    download = text.index("Download only the exact compact source artifact")
    validate = text.index("Reconstruct the frozen early support decision")
    upload = text.index("Upload the corrected independent support audit")
    publish = text.index("Publish the corrected independently reconstructed terminal")
    enforce = text.index("Enforce completion of the corrected audit")
    assert verify < download < validate < upload < publish < enforce

    assert "compact source artifact roster changed" in text
    assert "validate_deform360_prob4d_metric_batch" in text
    assert '"metric_batch": "success"' in text
    assert '"support_gate": "failure"' in text
    assert '"samples": "skipped"' in text
    assert '"calibration": "skipped"' in text
    assert '"source_gate": "skipped"' in text
    assert "source_gate_evaluated" in text
    assert '"audit_status": "validated-support-negative"' in text
    assert '"terminal_stage": "metric-support"' in text
    assert '"terminal_status": metric["status"]' in text
    assert 'metric["admitted_stream_count"] == 324' in text
    assert 'metric["supported_stream_count"] == 313' in text
    assert 'metric["support_negative_stream_count"] == 11' in text
    assert 'metric["technical_failure_stream_count"] == 0' in text
    assert 'metric["supported_object_count"] == 10' in text
    assert 'metric["plan_emitted"] is False' in text
    assert 'metric["status"] == "support-negatives-retained"' in text
    assert "released-robot-geometry-outside-" in text
    assert 'Counter({"sheet": 4, "volumetric": 7})' in text
    assert "len(objects) != 6 or len(cameras) != 5" in text


def test_v2_auditor_corrects_only_the_audit_interpretation() -> None:
    text = _text()
    normalized = " ".join(text.split())

    assert "The earlier v1 audit reported" in normalized
    assert "every terminal source artifact must contain a later" in normalized
    assert "This v2 audit independently validates" in normalized
    assert "It changes no source result, camera roster, threshold" in normalized
    assert "validated-support-negative" in text
    assert ".source_gate_evaluated" in text
    assert ".confirmation_access_authorized" in text
    assert "retention-days: 180" in text
    assert "continue-on-error: true" in text
    assert "sha256sum --check SHA256SUMS" in text


def test_v2_auditor_runs_contracts_before_the_one_shot_audit() -> None:
    text = _text()

    assert "needs: contracts" in text
    assert "github.event_name != 'pull_request'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "tests/test_deform360_prob4d_source_support_auditor_v2.py" in text
    assert "tests/test_deform360_prob4d_source_workflow.py" in text
    assert "tests/test_deform360_prob4d_metric_batch.py" in text
    assert "tests/test_deform360_prob4d_metric_batch_regressions.py" in text
