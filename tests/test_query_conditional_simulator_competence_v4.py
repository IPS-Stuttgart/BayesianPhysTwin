from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v4_paper_synthesis_matches_prospective_joint_contract() -> None:
    text = (
        ROOT / "docs/query_conditional_simulator_competence_v4.md"
    ).read_text(encoding="utf-8")
    assert "confidence at least `0.95`" in text
    assert "320 fresh evaluation worlds" in text
    assert "Rewards are never pooled across" in text
    assert "tasks: all means" in text
    assert "both empirical runs are still active" in text
    assert "no portfolio result is asserted" in text
    assert "post-hoc" in text
    assert "atlas synthesis" in text
