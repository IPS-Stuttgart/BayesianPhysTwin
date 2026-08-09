"""Static contracts for the protected Deform360 Prob4D source gate."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-prob4d-source-gate.yml")
LAUNCHER = Path(
    ".github/workflows/launch-deform360-prob4d-source-gate-once.yml"
)
STORAGE_ROOT = "/mnt/lexar4tb/datasets/deform360"
ADAPTIVE_ROOT = (
    "/mnt/lexar4tb/datasets/deform360/"
    "adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370"
)
PROB4D_REVISION = "25d90ef7f78ba4307f4555cb636d666004e1bf66"
PROCESSING_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"


def _block(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def test_pull_request_validation_is_hosted_and_payload_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    contracts = _block(text, "  contracts:", "  source-gate:")
    source_gate = text[text.index("  source-gate:") :]

    assert "pull_request:" in text
    assert "workflow_call:" in text
    assert "pull_request_target:" not in text
    assert "runs-on: ubuntu-latest" in contracts
    assert "runs-on: self-hosted" not in contracts
    assert "runs-on: self-hosted" in source_gate
    assert "inputs.execute_authorized == true" in source_gate
    assert "github.event_name == 'push'" in source_gate
    assert "github.ref == 'refs/heads/main'" in source_gate
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in source_gate
    assert (
        "launch-deform360-prob4d-source-gate-once.yml@refs/heads/main"
        in source_gate
    )
    assert "contents: write" not in text
    assert "git push" not in text


def test_source_gate_binds_exact_runner_roots_and_revisions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert f"DEFORM360_STORAGE_ROOT: {STORAGE_ROOT}" in text
    assert f"DEFORM360_ADAPTIVE_CONFIRMATION_ROOT: {ADAPTIVE_ROOT}" in text
    assert f"PROB4D_REVISION: {PROB4D_REVISION}" in text
    assert f"PROCESSING_REVISION: {PROCESSING_REVISION}" in text
    assert (
        "FROZEN_VISUAL_IMPLEMENTATION_REVISION: "
        "c4e68bf54aa4f039a1bed04cd4f2cdcc3eedfe4c"
        in text
    )
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert "AUTHORITATIVE_ADMISSION_RUN_ID: \"31272512658\"" in text
    assert "AUTHORITATIVE_ADMISSION_ARTIFACT_ID: \"9026043628\"" in text
    assert (
        "AUTHORITATIVE_ADMISSION_ID: "
        "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15"
        in text
    )


def test_pipeline_runs_the_registered_stages_in_order() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    stages = [
        "scripts/science/materialize_deform360_prob4d_metric_batch.py",
        "scripts/science/materialize_deform360_prob4d_calibration_samples.py",
        "scripts/science/fit_deform360_prob4d_source_calibration.py fit",
        "scripts/science/evaluate_deform360_prob4d_source_gate.py",
    ]
    positions = [text.index(stage, text.index("  source-gate:")) for stage in stages]

    assert positions == sorted(positions)
    assert (
        "protocols/locks/deform360_official_hub_prob4d_robot_metric_gauge_v1.json"
        in text
    )
    assert (
        "protocols/locks/deform360_official_hub_prob4d_source_gate_v1.json"
        in text
    )
    assert "validate-bundle" in text
    assert "Frozen visual-production root escaped its reviewed namespace" in text
    assert "FROZEN_VISUAL_IMPLEMENTATION_REVISION" in text
    assert "all-jobs-succeeded" in text
    assert "all-streams-supported" in text


def test_confirmation_and_adaptive_payloads_remain_outside_execution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    source_gate = text[text.index("  source-gate:") :]
    resolve = _block(
        source_gate,
        "      - name: Resolve protected roots and isolate the source-gate runtime",
        "      - name: Verify authoritative admission artifact metadata",
    )

    assert "adaptive_confirmation_payloads_opened=false" in source_gate
    assert "confirmation_payloads_opened=false" in source_gate
    assert "target_outcomes_used=false" in source_gate
    assert "future_frames_used=false" in source_gate
    assert "replacement_allowed=false" in source_gate
    assert "realpath -e \"${DEFORM360_ADAPTIVE_CONFIRMATION_ROOT}\"" not in text
    assert "find \"${DEFORM360_ADAPTIVE_CONFIRMATION_ROOT}\"" not in text
    assert "adaptive=\"${DEFORM360_ADAPTIVE_CONFIRMATION_ROOT}\"" in resolve
    assert "is_within \"${adaptive}\" \"${candidate}\"" in resolve


def test_outputs_are_no_overwrite_and_compact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    upload = _block(
        text,
        "      - name: Upload compact source-gate evidence",
        "      - name: Fail after retaining technical metric failures",
    )
    receipt = _block(
        text,
        "      - name: Publish compact source-gate receipt",
        "      - name: Upload compact source-gate evidence",
    )

    assert "PIPELINE_BASE_ROOT" in text
    assert "${result_id}/${GITHUB_SHA}" in text
    assert 'if [[ -d "${METRIC_BATCH_ROOT}" ]]' in text
    assert 'if [[ -d "${SAMPLES_ROOT}" ]]' in text
    assert 'if [[ ! -d "${SOURCE_CALIBRATION_ROOT}" ]]' in text
    assert 'if [[ -d "${SOURCE_GATE_ROOT}" ]]' in text
    assert "${{ env.COMPACT_ROOT }}" in upload
    assert "samples.npz" not in upload
    assert "VISUAL_OUTPUT_ROOT" not in upload
    assert "source-gate-receipt.json" in receipt
    assert "SHA256SUMS" in receipt
    assert "retention-days: 30" in upload


def test_support_negative_and_gate_failure_are_complete_results() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "continue_pipeline=" in text
    assert "technical-failures-retained" in text
    assert "Fail after retaining technical metric failures" in text
    assert "gate_passed" in text
    assert "confirmation_access_authorized" in text
    assert "A passing source gate authorizes only separately locked" in text


def test_one_shot_launcher_posts_the_bounded_decision() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "push:" in text
    assert "branches: [main]" in text
    assert (
        "uses: ./.github/workflows/deform360-prob4d-source-gate.yml" in text
    )
    assert "execute_authorized: true" in text
    assert "needs.execute.outputs.status" in text
    assert "needs.execute.outputs.result_id" in text
    assert "needs.execute.outputs.gate_passed" in text
    assert "needs.execute.outputs.confirmation_access_authorized" in text
    assert '"repos/${GITHUB_REPOSITORY}/issues/148/comments"' in text
    assert "It is not confirmation evidence" in text
    assert "replacement allowed: \\`false\\`" in text
