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
RUN_RECORD_CI = Path(
    ".github/workflows/deform360-calibration-run-record-ci.yml"
)


def test_reusable_workflow_beacons_before_checkout_and_initializes_evidence() -> None:
    text = REUSABLE.read_text(encoding="utf-8")

    beacon = text.index(
        "- name: Publish non-sensitive run beacon before checkout"
    )
    source_checkout = text.index(
        "- name: Check out exact reviewed BayesianPhysTwin source"
    )
    processing_checkout = text.index(
        "- name: Check out exact official Deform360 processing source"
    )

    assert "workflow_call:" in text
    assert beacon < source_checkout < processing_checkout
    assert 'echo "EVIDENCE_ROOT=${evidence_root}"' in text
    assert 'echo "PYTHON_SITE=${site_root}"' in text
    assert "ref: ${{ inputs.source_sha }}" in text
    assert text.count("persist-credentials: false") == 2
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "group: deform360-official-calibration-source-direct" in text
    assert "cancel-in-progress: true" in text


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
    assert "render_deform360_calibration_source_run_record.py" in completion_block
    assert 'python3 "${renderer}" issue' in completion_block
    assert '--source-revision "${BPT_SOURCE_SHA}"' in completion_block
    assert '--workflow-run-id "${GITHUB_RUN_ID}"' in completion_block
    assert '--workflow-run-attempt "${GITHUB_RUN_ATTEMPT}"' in completion_block
    assert "/issues/148/comments" in completion_block
    assert "local paths, object identities, or target outcomes" in completion_block
    assert "hashlib" not in completion_block
    assert "DATA_ROOT" not in completion_block
    assert "PROCESSED_ROOT" not in completion_block


def test_reusable_summary_uses_the_same_strict_renderer() -> None:
    text = REUSABLE.read_text(encoding="utf-8")
    summary = text.index("- name: Publish compact job summary")
    cleanup = text.index("- name: Remove isolated runtime and processing checkout")
    summary_block = text[summary:cleanup]

    assert "render_deform360_calibration_source_run_record.py" in summary_block
    assert 'python3 "${renderer}" summary' in summary_block
    assert '--source-revision "${BPT_SOURCE_SHA}"' in summary_block
    assert '--workflow-run-id "${GITHUB_RUN_ID}"' in summary_block
    assert '--workflow-run-attempt "${GITHUB_RUN_ATTEMPT}"' in summary_block
    assert "json.loads" not in summary_block


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
    assert (
        '--plan-json "${EVIDENCE_ROOT}/calibration-source-plan.json"'
        in finalize
    )
    assert (
        '--download-json "${EVIDENCE_ROOT}/calibration-download-manifest.json"'
        in finalize
    )
    assert (
        '--result-json "${EVIDENCE_ROOT}/calibration-source-result.json"'
        in finalize
    )
    assert 'if [[ -f "${manifest}" ]]; then' in finalize
    assert 'exit "${record_status}"' in finalize
    assert "tests/test_deform360_calibration_source_direct_workflow.py" in text
    assert "tests/test_deform360_calibration_source_run_record.py" in text
    assert "tests/test_deform360_calibration_source_run_record_validation.py" in text
    assert "tests/test_render_deform360_calibration_source_run_record.py" in text
    assert "scripts/ci/render_deform360_calibration_source_run_record.py" in text
    assert "src/bayesian_phystwin/_deform360_calibration_artifact_chain.py" in text
    assert "src/bayesian_phystwin/_deform360_calibration_run_common.py" in text
    assert (
        "src/bayesian_phystwin/"
        "_deform360_calibration_source_run_record_impl.py" in text
    )
    assert (
        "src/bayesian_phystwin/"
        "_deform360_calibration_source_run_record_validation.py" in text
    )
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


def test_focused_run_record_ci_is_exact_head_and_read_only() -> None:
    text = RUN_RECORD_CI.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "workflow_dispatch:" in text
    assert "contents: read" in text
    assert "runs-on: ubuntu-latest" in text
    assert 'python-version: ["3.10", "3.12"]' in text
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in text
    assert "persist-credentials: false" in text
    assert "ruff check" in text
    assert "ruff format --check" in text
    assert "bash -n scripts/ci/run_deform360_calibration_source_direct.sh" in text
    assert "test_deform360_calibration_source_run_record.py" in text
    assert "test_deform360_calibration_source_run_record_validation.py" in text
    assert "test_render_deform360_calibration_source_run_record.py" in text
    assert "render_deform360_calibration_source_run_record.py" in text
    assert "_deform360_calibration_artifact_chain.py" in text
    assert "_deform360_calibration_run_common.py" in text
    assert "_deform360_calibration_source_run_record_impl.py" in text
    assert "_deform360_calibration_source_run_record_validation.py" in text
    assert "self-hosted" not in text
