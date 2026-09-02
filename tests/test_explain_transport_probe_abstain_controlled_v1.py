from __future__ import annotations

from scripts.science.run_explain_transport_probe_abstain_controlled_v1 import run


def test_controlled_phase_study_exercises_all_operational_outcomes() -> None:
    result = run()

    assert result["decision"] == ("explain-transport-probe-abstain-strict-separation")
    assert all(result["checks"].values())
    metrics = result["metrics"]
    assert metrics["registered_phase_targets"] == 3
    assert metrics["transport_without_unique_cause"] == 1
    assert metrics["target_directed_probe_targets"] == 2
    assert metrics["unique_explanation_transport_targets"] == 1
    assert metrics["none_of_the_above_targets"] == 1
    assert metrics["unresolvable_abstentions"] == 1
    assert metrics["no_detectable_error_targets"] == 1
    assert metrics["relative_mean_cost_reduction_vs_full_cause"] >= (2.0 / 3.0 - 1e-12)
