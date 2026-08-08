from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(
    ".github/workflows/diagnose-deform360-visual-production-failure-once.yml"
)


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_failure_diagnostic_is_valid_and_runs_once_after_reviewed_merge() -> None:
    text = _workflow()
    parsed = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(parsed, dict)
    assert "pull_request:" in text
    assert "push:" in text
    assert "branches: [main]" in text
    assert "workflow_dispatch:" not in text
    assert "cancel-in-progress: false" in text
    assert "github.event_name == 'push'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "runs-on: self-hosted" in text
    assert "2026-08-09-common-traceback-v1" in text


def test_failure_diagnostic_binds_exact_retained_execution() -> None:
    text = _workflow()

    expected = (
        'FAILED_PRODUCTION_RUN_ID: "31277475724"',
        "FAILED_PRODUCTION_ATTEMPT_ID: 31277475724-1",
        "FAILED_IMPLEMENTATION_REVISION: "
        "312d4a46545b547efcdab79d4f285473077887cf",
        "ADMISSION_ID: "
        "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15",
        'EXPECTED_JOB_COUNT: "324"',
        "EXPECTED_STDERR_SHA256: "
        "0ceb1aa1732f8efd2a75eac1c008c6afd5fc1e80349e01d3c86f09706ff52fba",
        'EXPECTED_STDERR_BYTES: "812"',
        "VISUAL_OUTPUT_ROOT: /mnt/lexar4tb/datasets/deform360/results/"
        "bayesian-phystwin/calibration-visual-production",
    )
    for value in expected:
        assert value in text
    assert "expected {expected_count} failure receipts" in text
    assert "retained failures do not share the expected traceback" in text
    assert "representative traceback digest changed" in text


def test_failure_diagnostic_reads_only_receipts_and_retained_technical_log() -> None:
    text = _workflow()
    diagnostic = text[text.index("  diagnose:") :]

    assert 'failure_root = run_root / "failures"' in diagnostic
    assert 'stderr_path = safe_member(run_root, str(stderr_record["path"]))' in diagnostic
    assert "stable_read(stderr_path, expected_size=expected_bytes)" in diagnostic
    assert "retained_technical_logs_opened" in diagnostic
    assert "retained_calibration_camera_payloads_opened" in diagnostic
    assert "calibration_tactile_payloads_opened" in diagnostic
    assert "calibration_robot_state_opened" in diagnostic
    assert "reserved_evaluation_frames_opened" in diagnostic
    assert "adaptive_confirmation_payloads_opened" in diagnostic
    assert "confirmation_payloads_opened" in diagnostic
    assert "target_outcomes_used" in diagnostic
    assert "replacement_allowed" in diagnostic
    assert "/data-7fea8e2" not in diagnostic
    assert "adaptive-confirmation-download-" not in diagnostic
    assert "undistorted.mp4" not in diagnostic
    assert "aligned_timestamps.txt" not in diagnostic
    assert "np.load" not in diagnostic
    assert "VideoReader" not in diagnostic


def test_failure_diagnostic_rejects_symlinks_and_path_escape() -> None:
    text = _workflow()

    assert "path contains a symbolic link" in text
    assert "retained path contains a symlink" in text
    assert 'os.O_NOFOLLOW' in text
    assert 'any(part in {"", ".", ".."} for part in pure.parts)' in text
    assert "file changed while being read" in text
    assert "not a regular file" in text


def test_failure_diagnostic_sanitizes_paths_and_uploads_only_compact_text() -> None:
    text = _workflow()

    assert '"<GITHUB_WORKSPACE>"' in text
    assert '"<RUNNER_TEMP>"' in text
    assert '"<HOME>"' in text
    assert '"<DEFORM360_STORAGE>"' in text
    assert '"<ABSOLUTE_PATH>/"' in text
    assert "sanitized-traceback.txt" in text
    assert "diagnostic.json" in text
    assert "SHA256SUMS" in text
    assert "actions/upload-artifact@v7" in text
    assert "*.bin" not in text
    assert "*.npz" not in text
    assert "retention-days: 90" in text
