from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-v61-candidate-producer.yml"
RETIREMENT = (
    ROOT / "results/diagnostics/deform360_v61_one_shot_retirement_v1/retirement.json"
)


def _retirement_record() -> dict[str, object]:
    return json.loads(RETIREMENT.read_text(encoding="utf-8"))


def test_candidate_producer_is_retired_after_one_terminal_success() -> None:
    assert not WORKFLOW.exists()

    record = _retirement_record()
    producer = record["producer"]
    assert isinstance(producer, dict)
    assert producer["tracking_issue"] == 642
    assert producer["workflow_run_id"] == 31647329129
    assert producer["workflow_run_attempt"] == 1
    assert producer["source_revision"] == ("2eb8d12e2120d58d0d678c3771d29faaeb765497")
    assert producer["workflow_conclusion"] == "success"
    assert producer["artifact_id"] == 9161411983
    assert producer["artifact_sha256"] == (
        "03065bdecf9dd5906e70d5722f8d5f1608ae4968dccda585e13e732d5b7a9849"
    )
    assert producer["execution_receipt_id"] == (
        "65747822fa8380296a572811772fce88b9275a7e1148a8015e1156f520f7e369"
    )
    assert producer["candidate_panel_receipt_id"] == (
        "db3cc4351436492db5962bc1e99f516adc38a5031140b675b45dc6d752b7559a"
    )
    assert producer["raw_prediction_batch_id"] == (
        "d27674518f523db4fddb9cc108dd3d77321dddefeccc866b2b81044bf44ebee8"
    )
    assert producer["prediction_record_count"] == 100
    assert producer["technical_failure_record_count"] == 0
    assert producer["source_suffix_opened"] is False


def test_retirement_record_is_content_addressed_and_fail_closed() -> None:
    record = _retirement_record()
    declared = record.pop("record_id")
    assert declared == content_id(record)
    assert declared == (
        "ea4f31a025ec714ce8f13025a1af899028fac0a4ad12d3cf3bfd305c957bd168"
    )

    retirement = record["retirement"]
    assert isinstance(retirement, dict)
    assert retirement == {
        "confirmation_authorized": False,
        "producer_workflow_deleted": True,
        "rerun_authorized": False,
        "scorer_workflow_deleted": True,
        "target_access_authorized": False,
    }
