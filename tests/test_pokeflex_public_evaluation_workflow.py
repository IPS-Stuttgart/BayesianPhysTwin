from __future__ import annotations

from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "pokeflex-public-evaluation.yml"
)


def test_workflow_uses_registry_command_and_isolated_self_hosted_run() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "bpt experiment run evaluate-pokeflex-public" in workflow
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in workflow
    assert (
        "${RUNNER_TEMP}/pokeflex-public-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
        in workflow
    )
    assert "cancel-in-progress: false" in workflow
    assert "python scripts/remote/run_pokeflex" not in workflow


def test_workflow_exposes_no_reserved_or_prospective_stage() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    option_block = workflow.split("options:", 1)[1].split("dataset_root:", 1)[0]
    assert "- contracts" in option_block
    assert "- source-validation" in option_block
    assert "prospective" not in option_block
    assert "calibration" not in option_block
    assert "replacement_allowed" in workflow
    assert "retrospective/exploratory" in workflow


def test_workflow_uploads_only_compact_evidence() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    upload = workflow.split("name: Upload compact evaluation evidence", 1)[1]
    assert "execution_manifest.json" in upload
    assert "source_validation_progress_v2.json" in upload
    assert "evaluation_summary.json" in upload
    assert "source_validation_analysis.json" in upload
    assert "*_full_template15mm_v2.json" not in upload
    assert "actions/cache" not in workflow
