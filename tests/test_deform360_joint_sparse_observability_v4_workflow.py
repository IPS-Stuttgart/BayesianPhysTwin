from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/deform360-joint-sparse-observability-v4.yml")


def test_v4_workflow_is_contract_only_on_pull_requests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
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


def test_v4_workflow_binds_runner_storage_and_closed_information_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert "/mnt/lexar4tb/datasets/deform360/data-7fea8e2" in text
    assert (
        "adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370"
        in text
    )
    assert "DEFORM360_JOINT_SPARSE_V4_MANIFEST" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert "v4 manifest must be below the Deform360 results tree" in text
    assert "development_cohort_only=true" in text
    assert "prediction_point_values_used=false" in text
    assert "prediction_residuals_used=false" in text
    assert "confirmation_payloads_opened=false" in text
    assert "adaptive_confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "future_frames_used=false" in text
    assert "replacement_allowed=false" in text


def test_v4_workflow_accepts_support_negative_as_complete_scientific_result() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'test "${status}" = "0" -o "${status}" = "3"' in text
    assert '"development-design-supported"' in text
    assert '"development-design-not-supported"' in text
    assert '"development-technical-failures-retained"' in text
    assert "actions/upload-artifact@v7" in text
