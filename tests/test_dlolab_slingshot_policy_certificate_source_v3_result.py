"""Frozen result contracts for the terminal Slingshot v3 source run."""

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v3/summary.json"
)


def test_compact_v3_result_is_content_bound_and_terminal() -> None:
    value = json.loads(SUMMARY.read_text())
    identity = value.pop("artifact_id")
    assert identity == content_id(value)
    assert value["status"] == "retained_calibration_world_qa_failure"
    assert value["stage"]["ordinary_calibration_future_actions"] == 1024
    assert value["stage"]["calibration_action_failures"] == 0
    assert value["stage"]["evaluation_prefixes"] == 0
    assert value["registered_failure"]["failed_check"] == "duplicate_positions"
    assert value["post_terminal_read_only_diagnostic"]["passing_worlds"] == 127
    assert value["complete_288_world_evaluation_scored"] is False
    assert value["source_gate_passed"] is False
    assert value["retry_authorized"] is False
    assert value["replacement_authorized"] is False
    assert value["partial_score_authorized"] is False


def test_result_note_preserves_claim_and_roster_boundaries() -> None:
    text = (
        ROOT / "docs/dlolab_slingshot_policy_certificate_source_v3_result.md"
    ).read_text()
    assert "No evaluation prefix" in text
    assert "does not authorize\nloosening this one" in text
    assert "no Slingshot policy-value" in text
    assert "V3's roster is closed" in text


def test_raw_verifier_rederives_every_action_and_world() -> None:
    text = (
        ROOT
        / "scripts/remote/verify_dlolab_slingshot_policy_certificate_source_v3_failure.py"
    ).read_text()
    assert "range(runner.ACTION_COUNT)" in text
    assert "runner.load_future_action" in text
    assert "independent_world_qa" in text
    assert "_verify_source(lock)" in text
    assert "raw v3 tree identity changed" in text
