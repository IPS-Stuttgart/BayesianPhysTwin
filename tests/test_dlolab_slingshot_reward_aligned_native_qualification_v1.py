"""Contracts for the derived reward-aligned native qualification."""

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_reward_aligned_native_qualification_v1/summary.json"
)


def test_reward_aligned_qualification_is_content_bound_and_scoped() -> None:
    value = json.loads(SUMMARY.read_text())
    identity = value.pop("artifact_id")
    assert identity == content_id(value)
    assert value["denominator"]["worlds"] == 128
    assert value["denominator"]["ordinary_action_processes"] == 1024
    assert value["denominator"]["reward_aligned_qualified_worlds"] == 128
    assert value["denominator"]["position_deterministic_worlds"] == 127
    assert value["source"]["new_native_execution"] is False
    assert value["source"]["v3_rescored"] is False
    assert value["v4_protocol_freeze_authorized"] is True
    assert value["v4_scientific_execution_authorized"] is False
    assert value["retry_authorized"] is False


def test_qualification_note_preserves_the_v3_failure() -> None:
    text = (
        ROOT / "docs/dlolab_slingshot_reward_aligned_native_qualification_v1.md"
    ).read_text()
    assert "remains part of v3" in text
    assert "does not score v3" in text
    assert "new disjoint worlds" in text
