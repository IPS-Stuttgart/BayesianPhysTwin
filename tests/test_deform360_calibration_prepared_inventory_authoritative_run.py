from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / ("deform360-calibration-prepared-inventory.yml")
)
GUIDE = ROOT / "docs" / "deform360_calibration_prepared_inventory.md"

AUTHORITATIVE_SOURCE_REVISION = "0f403cbed8b5fc9ac585b5f7c237106809207b3f"
AUTHORITATIVE_RUN_ID = "31236564360"
AUTHORITATIVE_ARTIFACT = "deform360-official-calibration-source-31236564360-1"
AUTHORITATIVE_RECORD_ID = (
    "edf3692d88fed3c011ee44da2508b39e4755a0e97a83a26a0391fcfe433d7b74"
)
SUPERSEDED_RUN_ID = "31236230283"


def test_inventory_workflow_uses_the_authoritative_source_execution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert f'SOURCE_RUN_ID: "{AUTHORITATIVE_RUN_ID}"' in text
    assert f"SOURCE_ARTIFACT_NAME: {AUTHORITATIVE_ARTIFACT}" in text
    assert f'SOURCE_RUN_ID: "{SUPERSEDED_RUN_ID}"' not in text
    assert "name: ${{ env.SOURCE_ARTIFACT_NAME }}" in text
    assert "run-id: ${{ env.SOURCE_RUN_ID }}" in text


def test_inventory_guide_names_the_authoritative_evidence_chain() -> None:
    text = GUIDE.read_text(encoding="utf-8")

    assert AUTHORITATIVE_SOURCE_REVISION in text
    assert f"workflow run `{AUTHORITATIVE_RUN_ID}`" in text
    assert AUTHORITATIVE_ARTIFACT in text
    assert AUTHORITATIVE_RECORD_ID in text
    assert "authoritative least-privilege" in text
    assert "predates the final step-scoped optional" in text


def test_authoritative_binding_contract_is_permanently_collected() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    relative = (
        "tests/test_deform360_calibration_prepared_inventory_authoritative_run.py"
    )

    assert f'- "{relative}"' in text
    assert relative in text
