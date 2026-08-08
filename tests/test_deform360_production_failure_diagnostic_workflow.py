from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(
    ".github/workflows/diagnose-deform360-visual-production-failure-once.yml"
)
SCRIPT = Path("scripts/ci/diagnose_deform360_visual_production_failure.py")
UNIT_TEST = Path("tests/test_diagnose_deform360_visual_production_failure.py")
STATIC_TEST = Path("tests/test_deform360_production_failure_diagnostic_workflow.py")


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
    assert "2026-08-09-common-traceback-v2" in text


def test_failure_diagnostic_binds_exact_retained_execution() -> None:
    text = _workflow()

    expected = (
        'FAILED_PRODUCTION_RUN_ID: "31277475724"',
        "FAILED_PRODUCTION_ATTEMPT_ID: 31277475724-1",
        "FAILED_IMPLEMENTATION_REVISION: 312d4a46545b547efcdab79d4f285473077887cf",
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
    for option in (
        "--visual-output-root",
        "--admission-id",
        "--implementation-revision",
        "--attempt-id",
        "--failed-workflow-run-id",
        "--expected-job-count",
        "--expected-stderr-sha256",
        "--expected-stderr-bytes",
        "--output-dir",
    ):
        assert option in text


def test_failure_diagnostic_validates_before_self_hosted_access() -> None:
    text = _workflow()
    contracts = text[text.index("  contracts:") : text.index("  diagnose:")]
    diagnose = text[text.index("  diagnose:") :]

    for path in (SCRIPT, UNIT_TEST, STATIC_TEST):
        assert str(path) in contracts
    assert "python -m ruff check" in contracts
    assert "python -m ruff format --check" in contracts
    assert "python -m mypy" in contracts
    assert "python -m pytest" in contracts
    assert "needs: contracts" in diagnose
    runner_check = 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"'
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert runner_check in diagnose
    assert diagnose.index(runner_check) < diagnose.index(
        "python scripts/ci/diagnose_deform360_visual_production_failure.py"
    )


def test_failure_diagnostic_wrapper_opens_no_dataset_payload() -> None:
    text = _workflow()
    diagnose = text[text.index("  diagnose:") :]

    assert "/data-7fea8e2" not in diagnose
    assert "adaptive-confirmation-download-" not in diagnose
    assert "undistorted.mp4" not in diagnose
    assert "aligned_timestamps.txt" not in diagnose
    assert "np.load" not in diagnose
    assert "VideoReader" not in diagnose
    assert "motioncrafter" not in diagnose.lower()
    assert "--visual-output-root" in diagnose
    assert 'mkdir -p -- "${parent}"' in diagnose
    assert 'echo "DIAGNOSTIC_ROOT=${diagnostic_root}" >> "${GITHUB_ENV}"' in diagnose


def test_failure_diagnostic_uploads_only_compact_sanitized_evidence() -> None:
    text = _workflow()
    upload = text[text.index("Upload compact failure diagnostic") :]

    assert "diagnostic.json" in text
    assert "sanitized-traceback.txt" in text
    assert "SHA256SUMS" in text
    assert "actions/upload-artifact@v7" in upload
    assert "*.bin" not in upload
    assert "*.npz" not in upload
    assert "retention-days: 90" in upload
    assert "if-no-files-found: error" in upload
