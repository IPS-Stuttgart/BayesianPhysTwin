"""Contracts for the Slingshot-v4 query competence atlas release."""

from pathlib import Path

from bayesian_phystwin.query_competence_atlas_v2 import (
    load_query_competence_atlas,
)
from scripts.build_dlolab_query_competence_atlas_v5 import build_atlas

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results/source/dlolab_query_competence_atlas_v5/atlas.json"


def test_atlas_v5_rebuilds_exactly() -> None:
    committed = load_query_competence_atlas(ATLAS)
    rebuilt = build_atlas()
    assert rebuilt.to_record() == committed.to_record()
    assert committed.metadata["atlas_release"] == 5
    assert len(committed.entries) == 6
    assert len(committed.certified_query_ids) == 2
    assert len(committed.rejected_query_ids) == 4


def test_slingshot_v4_is_certified_without_rewriting_v2() -> None:
    atlas = load_query_competence_atlas(ATLAS)
    slingshot = [
        entry
        for entry in atlas.entries
        if entry.query_scope.metadata.get("task") == "slingshot"
    ]
    assert len(slingshot) == 2
    by_version = {
        entry.query_scope.metadata.get("version", "v2"): entry
        for entry in slingshot
    }
    assert by_version["v2"].decision == "rejected"
    certified = by_version["reward-aligned-v4"]
    assert certified.decision == "certified"
    assert certified.independent_group_count == 288
    assert certified.metadata["accepted_worlds"] == 36
    assert certified.metadata["harm_risk_upper"] < 0.05
    assert certified.metadata["paired_gain_ci95"][0] > 0.0
    assert certified.metadata["paired_gain_vs_matched_guard_ci95"][0] > 0.0
    assert certified.query_scope.query_id != by_version["v2"].query_scope.query_id


def test_paper_synthesis_keeps_both_slingshot_queries_and_claim_boundary() -> None:
    text = (ROOT / "docs/query_conditional_simulator_competence_v2.md").read_text()
    assert "Slingshot v2" in text
    assert "Slingshot reward-aligned v4" in text
    assert "V2 remains rejected. V4 is certified" in text
    assert "3,328 one-action native" in text
    assert "not an official benchmark or SOTA" in text
