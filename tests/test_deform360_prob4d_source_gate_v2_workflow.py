from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(
    ".github/workflows/continue-deform360-prob4d-source-gate-v2.yml"
)


def _block(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def test_v2_workflow_is_hosted_only_on_pull_requests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)
    contracts = _block(text, "  contracts:", "  continue-source-gate:")
    empirical = _block(text, "  continue-source-gate:", "  receipt:")

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" not in text
    assert "runs-on: ubuntu-latest" in contracts
    assert "runs-on: self-hosted" not in contracts
    assert "runs-on: self-hosted" in empirical
    assert "github.event_name == 'push'" in empirical
    assert "github.ref == 'refs/heads/main'" in empirical
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in empirical
    assert (
        "continue-deform360-prob4d-source-gate-v2.yml@refs/heads/main"
        in empirical
    )
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text


def test_v2_workflow_binds_the_immutable_predecessor() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "LEGACY_SOURCE_REVISION: ded8910becbb" in text
    assert 'LEGACY_SOURCE_RUN_ID: "31297018948"' in text
    assert "LEGACY_METRIC_BATCH_RESULT_ID: f246394c84fd" in text
    assert "LEGACY_METRIC_BATCH_RESULT_SHA256: 679550aff53d" in text
    assert "PRODUCTION_RESULT_ID: 146f885351b2" in text
    assert "PRODUCTION_ADMISSION_ID: 715ab8479bad" in text
    assert "PRODUCTION_REVISION: c4e68bf54aa4" in text
    assert "PROB4D_REVISION: 25d90ef7f78b" in text
    assert "SOURCE_GATE_LOCK_ID: cc96d2cc03a5" in text
    assert "deform360-prob4d-source-gate-v1" in text
    assert "deform360-prob4d-source-gate-v2" in text
    assert 'test ! -e "${source}"' in text


def test_v2_applies_only_the_frozen_object_support_minimum() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    admission = _block(
        text,
        "      - name: Admit supported streams under the frozen object minimum",
        "      - name: Enforce only the preregistered per-object support rule",
    )
    enforcement = _block(
        text,
        "      - name: Enforce only the preregistered per-object support rule",
        "      - name: Materialize correlation-aware source samples",
    )

    assert "admit_deform360_prob4d_metric_support.py" in admission
    assert '"supported_stream_count": 313' in text
    assert '"support_negative_stream_count": 11' in text
    assert '"technical_failure_stream_count": 0' in text
    assert '"minimum_supported_streams_per_object": 2' in enforcement
    assert '"status": "admitted-with-retained-support-negatives"' in enforcement
    assert "retained support-negative accounting changed" in enforcement
    assert 'result["information_boundary"]["replacement_allowed"] is not False' in enforcement
    assert '"supported_stream_count": 324' not in text
    assert '"support_negative_stream_count": 0' not in text


def test_v2_corrects_metric_root_and_preserves_stage_order() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    support = text.index("Admit supported streams under the frozen object minimum")
    samples = text.index("Materialize correlation-aware source samples")
    calibration = text.index("Fit object-balanced source calibration")
    gate = text.index("Evaluate the unchanged frozen source gate")

    assert support < samples < calibration < gate
    assert '--metric-root "${LEGACY_METRIC_BATCH_ROOT}/metrics"' in text
    assert '--metric-root "${LEGACY_METRIC_BATCH_ROOT}/metric-prefix"' not in text
    assert '--plan "${SUPPORT_ADMISSION_ROOT}/metric-prefix-plan.json"' in text
    assert "deform360_official_hub_prob4d_source_gate_v1.json" in text
    assert "continue-on-error: true" in text


def test_v2_uploads_compact_evidence_before_enforcement() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    collect = text.index("Collect compact decision before enforcement")
    upload = text.index("Upload compact source-only continuation evidence")
    enforce = text.index("Validate and expose the registered result")
    upload_block = _block(
        text,
        "      - name: Upload compact source-only continuation evidence",
        "      - name: Validate and expose the registered result",
    )

    assert collect < upload < enforce
    assert "legacy-metric-batch-result.json" in text
    assert "metric-support-admission-result.json" in text
    assert "source-calibration-result.json" in text
    assert "pipeline-receipt.json" in text
    assert "SHA256SUMS" in text
    assert "${{ env.COMPACT_ROOT }}" in upload_block
    assert "samples.npz" not in upload_block
    assert "retention-days: 90" in upload_block


def test_v2_keeps_confirmation_and_future_inputs_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "adaptive-confirmation" not in text
    assert "held-v8" not in text
    assert "rendered_depth.h5" not in text
    assert "confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "future_frames_used=false" in text
    assert "replacement_allowed=false" in text
    assert '"confirmation_payloads_opened": False' in text
    assert '"target_outcomes_used": False' in text
    assert '"future_frames_used": False' in text
    assert '"replacement_allowed": False' in text
    assert '"new_measurements_required": False' in text
    assert '"human_approval_required": False' in text


def test_v2_accepts_a_valid_negative_as_complete_scientific_output() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    final = _block(
        text,
        "      - name: Validate and expose the registered result",
        "      - name: Remove run-local runtime and compact staging",
    )

    assert "validate_source_gate_result" in final
    assert "result['gate_passed']" in final
    assert "result['confirmation_access_authorized']" in final
    assert 'test "${SOURCE_GATE_OUTCOME}" = "success"' in final
    assert 'test "${GATE_PASSED}" = "true"' not in final
    assert "a valid negative" in text.lower()
    assert '"repos/${GITHUB_REPOSITORY}/issues/148/comments"' in text
