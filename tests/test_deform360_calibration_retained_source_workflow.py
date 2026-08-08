"""Static contracts for the reviewed Deform360 retained-source execution."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REUSABLE = ROOT / ".github/workflows/deform360-calibration-prepared-inventory.yml"
LAUNCHER = (
    ROOT / ".github/workflows/launch-deform360-calibration-retained-source-once.yml"
)
INVENTORY_GUIDE = ROOT / "docs/deform360_calibration_prepared_inventory.md"

AUTHORITATIVE_RUN_ID = "31236564360"
AUTHORITATIVE_ARTIFACT = "deform360-official-calibration-source-31236564360-1"
AUTHORITATIVE_ARTIFACT_DIGEST = (
    "866c3f05e733e0cd6548e97ea4134476a37c7e01d09614cda2e86b3cb59d97d2"
)
SUPERSEDED_RUN_ID = "31236230283"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reusable_workflow_binds_the_authoritative_source_execution() -> None:
    source = _source(REUSABLE)

    assert f'SOURCE_RUN_ID: "{AUTHORITATIVE_RUN_ID}"' in source
    assert f"SOURCE_ARTIFACT_NAME: {AUTHORITATIVE_ARTIFACT}" in source
    assert f"SOURCE_ARTIFACT_DIGEST: {AUTHORITATIVE_ARTIFACT_DIGEST}" in source
    assert SUPERSEDED_RUN_ID not in source
    assert "run-id: ${{ env.SOURCE_RUN_ID }}" in source
    assert "name: ${{ env.SOURCE_ARTIFACT_NAME }}" in source


def test_reusable_workflow_is_reviewed_main_only_and_read_only() -> None:
    source = _source(REUSABLE)

    assert "workflow_call:" in source
    assert "workflow_dispatch:" in source
    assert "github.event_name != 'pull_request'" in source
    assert "github.ref == 'refs/heads/main'" in source
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in source
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in source
    assert "persist-credentials: false" in source
    assert "pull_request_target:" not in source
    assert "contents: write" not in source
    assert "issues: write" not in source
    assert "secrets: inherit" not in source


def test_reusable_workflow_materializes_the_complete_metadata_chain() -> None:
    source = _source(REUSABLE)
    execution = source.split(
        "      - name: Inventory retained bytes and freeze visual admission\n",
        maxsplit=1,
    )[1]

    inventory = execution.index(
        "scripts/science/inventory_deform360_calibration_prepared_source.py"
    )
    plan = execution.index(
        "scripts/science/build_deform360_calibration_visual_production_plan.py"
    )
    admission = execution.index(
        "scripts/science/admit_deform360_calibration_visual_execution.py"
    )
    assert inventory < plan < admission

    for output in (
        "prepared-source-inventory.json",
        "calibration-visual-production-plan.json",
        "calibration-visual-execution-admission.json",
        "receipt.json",
        "SHA256SUMS",
    ):
        assert output in execution

    for output_name in (
        "artifact_id",
        "artifact_url",
        "artifact_digest",
        "inventory_id",
        "plan_id",
        "admission_id",
        "camera_view_count",
    ):
        assert f"{output_name}:" in source


def test_one_shot_launcher_calls_only_the_reviewed_reusable_workflow() -> None:
    source = _source(LAUNCHER)

    assert "on:\n  push:\n    branches: [main]" in source
    assert (
        '      - ".github/workflows/'
        'launch-deform360-calibration-retained-source-once.yml"'
    ) in source
    assert (
        "uses: ./.github/workflows/deform360-calibration-prepared-inventory.yml"
    ) in source
    assert "if: always()" in source
    assert "issues: write" in source
    assert '"repos/${GITHUB_REPOSITORY}/issues/148/comments"' in source
    assert AUTHORITATIVE_RUN_ID in source
    assert AUTHORITATIVE_ARTIFACT in source
    assert AUTHORITATIVE_ARTIFACT_DIGEST in source
    assert "actions/checkout" not in source
    assert "pull_request_target:" not in source
    assert "workflow_dispatch:" not in source
    assert "secrets: inherit" not in source


def test_inventory_guide_distinguishes_authoritative_and_superseded_runs() -> None:
    source = _source(INVENTORY_GUIDE)

    assert AUTHORITATIVE_RUN_ID in source
    assert AUTHORITATIVE_ARTIFACT in source
    assert AUTHORITATIVE_ARTIFACT_DIGEST in source
    assert SUPERSEDED_RUN_ID in source
    assert "predates the final step-scoped optional" in source
    assert "is not substituted into this admission" in source
