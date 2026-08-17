"""Workflow contracts for the locked Deform360 calibration-source stage."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-official-hub-calibration-source.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_has_only_yaml_keys_at_top_level() -> None:
    top_level_key = re.compile(r"^[A-Za-z0-9_-]+:")
    invalid = [
        (line_number, line)
        for line_number, line in enumerate(_workflow_text().splitlines(), start=1)
        if line
        and not line.startswith((" ", "#"))
        and line not in {"---", "..."}
        and top_level_key.match(line) is None
    ]

    assert not invalid, (
        f"dedented non-key content makes workflow YAML invalid: {invalid}"
    )


def test_processing_checkout_exclusion_is_one_shell_line() -> None:
    text = _workflow_text()

    assert "printf '%s\\n' '/_deform360_processing/' >> .git/info/exclude" in text
    assert "printf '/_deform360_processing/\n' >>" not in text


def test_trusted_contracts_use_isolated_self_hosted_execution() -> None:
    text = _workflow_text()
    contracts = text.index("  contracts:")
    empirical = text.index("  prepare-calibration-source:")
    contract_block = text[contracts:empirical]
    empirical_block = text[empirical:]

    assert contracts < empirical
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in contract_block
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in contract_block
    )
    assert "actions/setup-python" not in contract_block
    assert "${RUNNER_TEMP}/deform360-calibration-contracts-" in contract_block
    assert 'echo "CONTRACT_SITE=${contract_site}"' in contract_block
    assert '--target "${CONTRACT_SITE}"' in contract_block
    assert "--break-system-packages" in contract_block
    assert "--no-cache-dir" in contract_block
    assert " -m venv" not in contract_block
    assert "ensurepip" not in contract_block
    assert "DATA_ROOT:" not in contract_block
    assert "PROCESSED_ROOT:" not in contract_block
    assert "HF_TOKEN" not in contract_block
    assert (
        " scripts/science/run_deform360_official_hub_calibration_source.py plan "
        not in contract_block
    )
    assert "if: github.event_name != 'pull_request'" in empirical_block
    assert "needs: contracts" in empirical_block


def test_empirical_job_uses_a_fresh_runner_temp_target_site() -> None:
    text = _workflow_text()
    empirical = text[text.index("  prepare-calibration-source:") :]
    job_header = empirical[: empirical.index("    steps:")]

    assert "actions/setup-python" not in empirical
    assert "${{ runner.temp }}" not in job_header
    assert "${RUNNER_TEMP}/deform360-calibration-site-" in empirical
    assert 'echo "PYTHON_SITE=${site_root}"' in empirical
    assert '--target "${PYTHON_SITE}"' in empirical
    assert "--break-system-packages" in empirical
    assert "--no-cache-dir" in empirical
    assert " -m venv" not in empirical
    assert "ensurepip" not in empirical
    assert "VENV_ROOT" not in empirical


def test_failure_reporting_does_not_require_the_target_site() -> None:
    text = _workflow_text()
    confirmation = text[text.index("      - name: Verify the confirmation cohort") :]
    summary = confirmation[confirmation.index("      - name: Publish job summary") :]

    assert "${PYTHON_SITE}/" not in confirmation
    assert "${PYTHON_SITE}/" not in summary
    assert "if-no-files-found: warn" in confirmation
    assert 'for path in "${PYTHON_SITE:-}" "${PROCESSING_REPO:-}"' in summary
