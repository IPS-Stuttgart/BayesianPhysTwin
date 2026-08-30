"""Frozen result contracts for independent native Slingshot execution."""

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_independent_native_qualification_v1/summary.json"
)


def test_compact_result_is_content_bound_and_scoped():
    value = json.loads(SUMMARY.read_text())
    identity = value.pop("artifact_id")
    assert identity == content_id(value)
    assert value["status"] == "passed"
    assert value["ordinary_processes"] == value["planned_processes"] == 64
    assert value["failed_processes"] == 0
    assert value["qualified_worlds"] == 8
    assert value["qualification_passed"] is True
    assert value["v3_protocol_freeze_authorized"] is True
    assert value["v3_scientific_execution_authorized"] is False
    assert value["scientific_policy_value_scored"] is False
    assert value["retry_authorized"] is False
    assert value["replacement_authorized"] is False
    assert value["protected_data_read"] is False


def test_result_note_preserves_the_claim_boundary():
    text = (
        ROOT / "docs/dlolab_slingshot_independent_native_qualification_v1_result.md"
    ).read_text()
    assert "does not recover or score the incomplete" in text
    assert "new disjoint calibration" in text
    assert "establish policy value" in text


def test_raw_verifier_covers_every_process_and_world():
    text = (
        ROOT
        / "scripts/remote/verify_dlolab_slingshot_independent_native_qualification_v1.py"
    ).read_text()
    assert "range(ACTION_COUNT)" in text
    assert "runner.load_task" in text
    assert "independent_world_qa" in text
    assert "_verify_source(lock)" in text
    assert "raw qualification tree identity changed" in text
