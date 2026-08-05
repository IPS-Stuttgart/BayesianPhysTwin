from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-stage1-control.yml"
MAIN_TEST_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def test_workflow_uses_registered_command_and_isolated_self_hosted_roots() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "bpt experiment run prepare-deform360-stage1" in workflow
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in workflow
    assert "${RUNNER_TEMP}/deform360-stage1-${GITHUB_RUN_ID}" in workflow
    assert "${RUNNER_TEMP}/deform360-stage1-seal-${GITHUB_RUN_ID}" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "persist-credentials: false" in workflow


def test_workflow_exposes_only_contract_prepare_and_seal_transitions() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    options = workflow.split("options:", 1)[1].split(
        "provider_attestation_path:",
        1,
    )[0]

    assert "- contracts" in options
    assert "- prepare" in options
    assert "- seal" in options
    assert "confirmation" not in options
    assert "target" not in options
    assert "evaluation" not in options


def test_workflow_never_downloads_or_opens_deform360_payloads() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    forbidden = (
        "huggingface_hub",
        "hf_hub_download",
        "brownu/deform360",
        "DEFORM360_DATA_ROOT",
        "dataset_root",
        "download-selected-object",
        "confirmation-opening",
    )

    for marker in forbidden:
        assert marker not in workflow
    assert "selected_raw_payloads_opened" not in workflow
    assert "target_outcomes" not in workflow


def test_workflow_uploads_only_compact_control_plane_artifacts() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    prepare_upload = workflow.split(
        "name: Upload target-blind prepare evidence",
        1,
    )[1].split("  seal:", 1)[0]
    seal_upload = workflow.split(
        "name: Upload compact calibration-seal evidence",
        1,
    )[1]

    assert "visual-provider-lock.json" in prepare_upload
    assert "stage1-plan.json" in prepare_upload
    assert "visual-calibration-lock.json" in seal_upload
    assert "stage1-seal-summary.json" in seal_upload
    assert "*.npz" not in workflow
    assert "*.pkl" not in workflow
    assert "*.mp4" not in workflow
    assert "actions/cache" not in workflow


def test_workflow_pins_every_third_party_action_to_a_full_sha() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for line in workflow.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        reference = stripped.split("@", 1)[1].split()[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)


def test_main_coverage_lane_runs_stage1_adversarial_suite() -> None:
    workflow = MAIN_TEST_WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_deform360_stage1_coverage.py" in workflow
