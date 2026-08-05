"""Workflow contracts for the locked Deform360 calibration-source stage."""

from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(
    ".github/workflows/deform360-official-hub-calibration-source.yml"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_pull_requests_validate_without_opening_payloads() -> None:
    text = _workflow_text()
    contracts = text.index("  contracts:")
    empirical = text.index("  prepare-calibration-source:")
    empirical_block = text[empirical:]

    assert contracts < empirical
    assert "runs-on: ubuntu-latest" in text[contracts:empirical]
    assert "if: github.event_name != 'pull_request'" in empirical_block
    assert "needs: contracts" in empirical_block


def test_self_hosted_job_uses_a_fresh_runner_temp_virtual_environment() -> None:
    text = _workflow_text()
    empirical = text[text.index("  prepare-calibration-source:") :]
    job_header = empirical[: empirical.index("    steps:")]

    assert "actions/setup-python" not in empirical
    assert "${{ runner.temp }}" not in job_header
    assert "${RUNNER_TEMP}/deform360-calibration-venv-" in empirical
    assert 'echo "VENV_ROOT=${venv_root}"' in empirical
    assert '"${BASE_PYTHON}" -m venv --copies "${VENV_ROOT}"' in empirical
    assert '"${python}" -m ensurepip --upgrade' in empirical
    assert "--no-cache-dir" in empirical


def test_failure_reporting_does_not_require_the_virtual_environment() -> None:
    text = _workflow_text()
    confirmation = text[text.index("      - name: Verify the confirmation cohort") :]
    summary = confirmation[
        confirmation.index("      - name: Publish job summary") :
    ]

    assert "${VENV_ROOT}/bin/python" not in confirmation
    assert "${VENV_ROOT}/bin/python" not in summary
    assert "if-no-files-found: warn" in confirmation
    assert 'for path in "${VENV_ROOT:-}" "${PROCESSING_REPO:-}"' in summary
