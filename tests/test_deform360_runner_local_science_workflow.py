"""Static safety contracts for runner-local Deform360 scientific execution."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-runner-local-science.yml")
OFFICIAL_ROOT = "/mnt/lexar4tb/datasets/deform360/data-7fea8e2"
ADAPTIVE_ROOT = (
    "/mnt/lexar4tb/datasets/deform360/"
    "adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370"
)


def _block(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def test_workflow_keeps_pull_request_validation_hosted_and_data_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    contracts = _block(text, "  contracts:", "  science:")
    science = text[text.index("  science:") :]

    assert "pull_request:" in text
    assert "workflow_dispatch:" in text
    assert "pull_request_target:" not in text
    assert "permissions:\n  contents: read" in text
    assert "cancel-in-progress: false" in text
    assert "runs-on: ubuntu-latest" in contracts
    assert "runs-on: self-hosted" not in contracts
    assert "runs-on: self-hosted" in science
    assert "github.event_name == 'workflow_dispatch'" in science
    assert "github.ref == 'refs/heads/main'" in science
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in science
    assert text.count("persist-credentials: false") >= 3
    assert "contents: write" not in text
    assert "git push" not in text


def test_workflow_binds_exact_runner_roots_and_frozen_revisions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert f"OFFICIAL_RAW_ROOT: {OFFICIAL_ROOT}" in text
    assert f"ADAPTIVE_CONFIRMATION_ROOT: {ADAPTIVE_ROOT}" in text
    assert (
        "OFFICIAL_RAW_SOURCE_REVISION: 7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
    ) in text
    assert "DATASET_REVISION: f804696d7a133908c7497ffdab43819d879b5cbc" in text
    assert "PROCESSING_REVISION: d8522a4403b766aeb387510c04e89032a56fdf35" in text
    assert (
        "LOCAL_SCIENCE_ROOT: "
        "/mnt/lexar4tb/datasets/deform360/bpt-runner-local-science-f804696d7a13"
    ) in text


def test_adaptive_confirmation_root_is_names_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    inventory = _block(
        text,
        "      - name: Build names-only inventories of both runner-resident roots",
        "      - name: Compare runner-resident roots without opening dataset payloads",
    )
    reuse = _block(
        text,
        "      - name: Reuse only exact calibration bytes from the official raw snapshot",
        "      - name: Download and hash only missing frozen calibration files",
    )

    assert "${RESOLVED_OFFICIAL_RAW_ROOT}" in inventory
    assert "${RESOLVED_ADAPTIVE_CONFIRMATION_ROOT}" in inventory
    assert "stage_deform360_local_calibration_cache.py" in reuse
    assert '--source-root "${RESOLVED_OFFICIAL_RAW_ROOT}"' in reuse
    assert "RESOLVED_ADAPTIVE_CONFIRMATION_ROOT" not in reuse
    assert 'boundary["adaptive_confirmation_root_accessed"] is not False' in reuse
    assert 'boundary["confirmation_payloads_opened"] is not False' in reuse


def test_science_path_uses_existing_frozen_calibration_contracts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    science = text[text.index("  science:") :]

    assert "run_deform360_official_hub_calibration_source.py plan" in science
    assert "run_deform360_official_hub_calibration_source.py download" in science
    assert "run_deform360_official_hub_calibration_source.py prepare" in science
    assert "protocols/deform360_official_hub_calibration_source_v1.json" in science
    assert (
        "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
        in science
    )
    assert '--data-root "${DATA_ROOT}"' in science
    assert '--processing-repository "${PROCESSING_REPO}"' in science
    assert "--workers 2" in science
    assert "confirmation_payloads_opened=false" in science
    assert "adaptive_confirmation_payloads_opened=false" in science
    assert "target_outcomes_used=false" in science
    assert "command -v nvidia-smi" in science


def test_raw_snapshot_reuse_is_copy_on_write_and_artifacts_are_compact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    upload = _block(
        text,
        "      - name: Upload compact runner-local science evidence",
        "      - name: Remove isolated runtime and processing checkout",
    )

    assert "local-cache-staging.json" in text
    assert 'boundary["hardlink_allowed"] is not False' in text
    assert "${{ env.EVIDENCE_ROOT }}" in upload
    assert "${{ env.DATA_ROOT }}" not in upload
    assert "${{ env.PROCESSED_ROOT }}" not in upload
    assert "retention-days: 30" in upload
