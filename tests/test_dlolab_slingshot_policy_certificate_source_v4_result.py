"""Frozen result contracts for the positive Slingshot v4 source run."""

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v4/summary.json"
)


def test_compact_v4_result_is_content_bound_and_positive() -> None:
    value = json.loads(SUMMARY.read_text())
    identity = value.pop("artifact_id")
    assert identity == content_id(value)
    assert value["status"] == "source_gate_passed"
    assert value["denominator"]["ordinary_action_processes_total"] == 3328
    assert value["denominator"]["technical_failures"] == 0
    assert value["pre_future"]["accepted_worlds"] == 36
    assert value["arms"]["policy_gain_guard"]["mean_gain_ci95"][0] > 0.0
    assert value["arms"]["policy_gain_guard"]["harm_probability_upper95"] < 0.05
    assert value["comparisons"][
        "policy_guard_paired_gain_vs_simultaneous_guard_ci95"
    ][0] > 0.0
    assert value["coverage"]["marginal_policy_gain"] >= 0.85
    assert value["source_gate_passed"] is True
    assert value["retry_authorized"] is False
    assert value["new_recordings"] is False


def test_result_note_states_decision_value_and_claim_boundary() -> None:
    text = (
        ROOT / "docs/dlolab_slingshot_policy_certificate_source_v4_result.md"
    ).read_text()
    assert "prospective source gate passed" in text
    assert "36/288 worlds updated" in text
    assert "an equal-data simultaneous-action uncertainty guard" in text
    assert "not an official benchmark or SOTA claim" in text
    assert "does not relax or rescore v3" in text


def test_raw_verifier_replays_complete_denominator_and_score() -> None:
    text = (
        ROOT
        / "scripts/remote/verify_dlolab_slingshot_policy_certificate_source_v4_result.py"
    ).read_text()
    assert "runner.verify_result(raw_root)" in text
    assert "_verify_frozen_source(lock)" in text
    assert '!= 1024' in text
    assert '!= 2304' in text
    assert "raw v4 tree identity changed" in text
