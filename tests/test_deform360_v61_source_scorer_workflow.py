from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v61-source-scorer-endpoint-gsplat-base.yml"
)
RETIREMENT = (
    ROOT
    / "results/diagnostics/deform360_v61_one_shot_retirement_v1/retirement.json"
)
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_1_source_scoring.json"
)
RUNTIME_LOCK = ROOT / (
    "requirements/locks/deform360-v61-source-scorer-pt24cu121-py310.txt"
)


def _retirement_record() -> dict[str, object]:
    return json.loads(RETIREMENT.read_text(encoding="utf-8"))


def test_source_scorer_is_retired_after_terminal_retained_failure() -> None:
    assert not WORKFLOW.exists()

    record = _retirement_record()
    scorer = record["scorer"]
    assert isinstance(scorer, dict)
    assert scorer["tracking_issue"] == 645
    assert scorer["workflow_run_id"] == 31669176135
    assert scorer["workflow_run_attempt"] == 1
    assert scorer["source_revision"] == (
        "74e556d6f9b503409f3b163ef27ccb7a17c61d85"
    )
    assert scorer["workflow_conclusion"] == "failure"
    assert scorer["artifact_id"] == 9169119864
    assert scorer["artifact_sha256"] == (
        "b1a8ea2e4d3952af4b446fa4d420ea23f310b4e43396cc8505978302fcd4e42f"
    )
    assert scorer["source_scoring_receipt_id"] == (
        "f284be9c6a83afe5688030cfec466f0bbe2f2a24d7ce0aa13eac272d9763742c"
    )
    assert scorer["terminal_status"] == (
        "source-scoring-technical-failure-retained"
    )
    assert scorer["terminal_stage"] == "endpoint-processing"
    assert scorer["exit_code"] == 2
    assert scorer["source_suffix_opened"] is True
    assert scorer["source_gate_evaluated"] is False
    assert scorer["source_gate_passed"] is None
    assert scorer["source_continuation_authorized"] is False
    assert scorer["replacement_allowed"] is False
    assert scorer["confirmation_payloads_opened"] is False
    assert scorer["target_outcomes_opened"] is False
    assert scorer["held_v8_artifacts_accessed"] is False


def test_retirement_preserves_frozen_reproduction_inputs() -> None:
    record = _retirement_record()
    declared = record.pop("record_id")
    assert declared == content_id(record)

    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == (
        "c616fe1fbe19785452535772adfa937501a0fa35ab41b3c2fc995a968e60a8f1"
    )
    assert hashlib.sha256(RUNTIME_LOCK.read_bytes()).hexdigest() == (
        "e46e32b809fd9438437cf0ff4138dccb119904b5f1d9f90900df99603f278af3"
    )

    claim_boundary = record["claim_boundary"]
    assert isinstance(claim_boundary, str)
    assert "source gate was not evaluated" in claim_boundary
    assert "no continuation or confirmation was authorized" in claim_boundary
