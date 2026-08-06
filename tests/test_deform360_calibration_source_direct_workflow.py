"""Static safety contracts for the direct Deform360 calibration workflow."""

from __future__ import annotations

from pathlib import Path

REUSABLE = Path(
    ".github/workflows/"
    "deform360-official-hub-calibration-source-reusable.yml"
)
DISPATCHER = Path(
    ".github/workflows/dispatch-deform360-calibration-source-pr-target.yml"
)
DIRECT_SCRIPT = Path("scripts/ci/run_deform360_calibration_source_direct.sh")


def test_reusable_workflow_initializes_evidence_before_checkout() -> None:
    text = REUSABLE.read_text(encoding="utf-8")

    initialize = text.index("- name: Initialize isolated evidence paths")
    source_checkout = text.index(
        "- name: Check out exact reviewed BayesianPhysTwin source"
    )
    processing_checkout = text.index(
        "- name: Check out exact official Deform360 processing source"
    )

    assert "workflow_call:" in text
    assert initialize < source_checkout < processing_checkout
    assert 'echo "EVIDENCE_ROOT=${evidence_root}"' in text
    assert 'echo "PYTHON_SITE=${site_root}"' in text
    assert "ref: ${{ inputs.source_sha }}" in text
    assert text.count("persist-credentials: false") == 2
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text


def test_reusable_workflow_publishes_one_non_sensitive_completion_receipt() -> None:
    text = REUSABLE.read_text(encoding="utf-8")
    completion = text.index("- name: Publish non-sensitive completion receipt")
    upload = text.index("- name: Publish compact calibration-source evidence")
    completion_block = text[completion:upload]

    assert completion < upload
    assert "if: always()" in completion_block
    assert "continue-on-error: true" in completion_block
    assert "execution-manifest.json" in completion_block
    assert "completion-receipt.json" in completion_block
    assert "deform360-calibration-source-run" in completion_block
    assert "/issues/148/comments" in completion_block
    assert "local paths, object identities, or target outcomes" in completion_block
    assert "DATA_ROOT" not in completion_block
    assert "PROCESSED_ROOT" not in completion_block


def test_direct_script_records_every_exit_after_boundary_verification() -> None:
    text = DIRECT_SCRIPT.read_text(encoding="utf-8")
    finalize = text[text.index("finalize() {") : text.index("trap finalize EXIT")]

    assert "local workload_status=$?" in finalize
    assert "verify_confirmation_boundary" in finalize
    assert "local boundary_status=$?" in finalize
    assert 'local manifest="${EVIDENCE_ROOT}/execution-manifest.json"' in finalize
    assert "deform360_calibration_source_run_record" in finalize
    assert '--output "${manifest}"' in finalize
    assert '--workload-exit-code "${workload_status}"' in finalize
    assert (
        '--confirmation-boundary-exit-code "${boundary_status}"' in finalize
    )
    assert 'if [[ -f "${manifest}" ]]; then' in finalize
    assert 'exit "${record_status}"' in finalize
    assert "tests/test_deform360_calibration_source_direct_workflow.py" in text
    assert "tests/test_deform360_calibration_source_run_record.py" in text
    assert "src/bayesian_phystwin/deform360_calibration_source_run_record.py" in text


def test_pull_request_target_dispatcher_does_not_execute_head_code() -> None:
    text = DISPATCHER.read_text(encoding="utf-8")

    assert "pull_request_target:" in text
    assert "head.repo.full_name == github.repository" in text
    assert "head.ref == 'agent/calibration-dispatch-trigger-v1'" in text
    assert "changed_files == 1" in text
    assert "additions == 1" in text
    assert "deletions == 0" in text
    assert "source_sha: ${{ github.sha }}" in text
    assert "actions/checkout" not in text
    assert "run:" not in text
