from __future__ import annotations

from scripts.science.run_interventional_cause_adequacy_controlled_v1 import run


def test_controlled_incomplete_family_study_passes_registered_checks() -> None:
    result = run(trials=1_200, seed=20260902)

    assert result["decision"] == (
        "cause-family-adequacy-and-none-of-the-above-supported"
    )
    assert all(result["checks"].values())
    metrics = result["metrics"]
    assert metrics["unknown_detection_recall"] >= 0.99
    assert metrics["forced_unknown_false_physical_promotion"] >= 0.99
    assert metrics["adequacy_unknown_false_physical_promotion"] <= 0.01
    assert metrics["registered_false_unknown_rate"] <= 0.01
    assert metrics["broken_relation_accuracy"] <= (
        metrics["adequacy_gated_accuracy"] - 0.20
    )
