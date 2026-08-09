from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/deform360-provider-failure-census-v1.yml")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_provider_failure_census_is_manual_and_contract_only_on_pull_requests() -> None:
    text = _text()
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "workflow_dispatch:" in text
    assert "runs-on: self-hosted" in text
    assert "inputs.execute_authorized == true" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "persist-credentials: false" in text
    assert "cancel-in-progress: false" in text
    assert "contents: read" in text


def test_provider_failure_census_binds_exact_storage_and_input_identity() -> None:
    text = _text()

    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert "runs-on: self-hosted" in text
    assert "/mnt/lexar4tb/datasets/deform360/data-7fea8e2" in text
    assert (
        "adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370"
        in text
    )
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert "expected_input_sha256" in text
    assert '[[ "${expected_sha}" =~ ^[0-9a-f]{64}$ ]]' in text
    assert 'test "${actual_sha}" = "${expected_sha}"' in text
    assert "evidence must remain below the Deform360 results tree" in text
    assert "evidence must not resolve below a raw-data root" in text
    assert "provider-failure input exceeds its byte budget" not in text
    assert 'test "${input_bytes}" -le 67108864' in text


def test_provider_failure_census_keeps_confirmation_and_targets_closed() -> None:
    text = _text()

    assert "raw_tree_traversal_allowed=false" in text
    assert "source_only=true" in text
    assert "confirmation_payloads_opened=false" in text
    assert "adaptive_confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "future_frames_used=false" in text
    assert "replacement_allowed=false" in text
    assert '"split": "source-only"' in text
    assert '"confirmation_payloads_opened": False' in text
    assert '"adaptive_confirmation_payloads_opened": False' in text
    assert '"target_outcomes_used": False' in text
    assert '"future_frames_used": False' in text
    assert '"replacement_allowed": False' in text
    assert 'find "${DEFORM360_OFFICIAL_RAW_ROOT}"' not in text
    assert 'find "${DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT}"' not in text
    assert "du -" not in text
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_provider_failure_census_publishes_compact_equal_case_evidence() -> None:
    text = _text()

    assert "bpt diagnostic run diagnose-provider-failures" in text
    assert ".equal_case_weighting == true" in text
    assert "metadata.statistical_unit must be explicit" in text
    assert "dense measurements must not be independent cases" in text
    assert "provider-failure-report.json" in text
    assert "execution-receipt.json" in text
    assert "summary.md" in text
    assert "SHA256SUMS" in text
    assert "sha256sum --check SHA256SUMS" in text
    assert "actions/upload-artifact@v7" in text
