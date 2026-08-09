from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/deform360-prob4d-source-gate.yml")


def test_public_source_gate_workflow_is_contract_only_on_pull_requests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "workflow_call:" in text
    assert "workflow_dispatch:" not in text
    assert "push:" not in text
    assert "inputs.execute_authorized == true" in text
    assert "github.event_name == 'push'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "runs-on: self-hosted" in text
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert "launch-deform360-prob4d-source-gate-once.yml@refs/heads/main" in text
    assert "cancel-in-progress: false" in text


def test_public_source_gate_workflow_pins_the_complete_real_data_lineage() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'PRODUCTION_RUN_ID: "31279398563"' in text
    assert 'PRODUCTION_ARTIFACT_ID: "9031215572"' in text
    assert "PRODUCTION_RESULT_ID: 146f885351b2af" in text
    assert "PRODUCTION_ADMISSION_ID: 715ab8479bad" in text
    assert "PROB4D_REVISION: 25d90ef7f78b" in text
    assert "MOTIONCRAFTER_REVISION: 9cb4e9679f5f" in text
    assert "SOURCE_GATE_LOCK_ID: cc96d2cc03a5" in text
    assert "repository: FlorianPfaff/Prob4D" in text
    assert 'test "$(git -C "${PROB4D_CHECKOUT}" rev-parse HEAD)"' in text
    assert "cmp -s" in text
    assert '"${VISUAL_PRODUCTION_ROOT}/visual-production-result.json"' in text
    assert '"public_released_measurements_used": True' in text


def test_public_source_gate_workflow_preserves_custody_and_exact_fallback() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "adaptive-confirmation" not in text
    assert "held-v8" not in text
    assert "rendered_depth.h5" not in text
    assert "calibration_tactile_payloads" not in text
    assert '"confirmation_payloads_opened": False' in text
    assert '"target_outcomes_used": False' in text
    assert '"future_frames_used": False' in text
    assert '"replacement_allowed": False' in text
    assert '"human_approval_required": False' in text
    assert '"new_measurements_required": False' in text
    assert 'test ! -e "${source}"' in text
    assert "Commit the durable no-overwrite source root" in text
    assert "find . -type f ! -path './SHA256SUMS'" in text
    assert "continue-on-error: true" in text
    assert "support-negatives-retained" not in text
    assert '"status": "all-streams-supported"' in text
    assert '"supported_stream_count": 324' in text
    assert '"support_negative_stream_count": 0' in text
    assert '"technical_failure_stream_count": 0' in text


def test_public_source_gate_uploads_both_negative_and_positive_decisions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    support_publish = text.index("Publish compact support evidence before enforcement")
    support_enforce = text.index("Enforce complete frozen stream support")
    decision_collect = text.index("Collect compact source decision before enforcement")
    decision_upload = text.index("Upload compact public source-only decision")
    decision_enforce = text.index("Enforce the frozen automated source decision")
    assert support_publish < support_enforce
    assert decision_collect < decision_upload < decision_enforce
    assert "steps.metric_batch.outcome" in text
    assert "steps.support_gate.outcome" in text
    assert "steps.samples.outcome" in text
    assert "steps.calibration.outcome" in text
    assert "steps.gate.outcome" in text
    assert 'result["confirmation_access_authorized"] is True' in text
    assert 'result["gate_passed"] is True' in text
