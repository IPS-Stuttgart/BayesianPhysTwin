from __future__ import annotations

import pytest

from experiments.adaptive_stochastic_policy_tree_v1.run import build_result


def test_controlled_result_has_strict_adaptive_separation() -> None:
    result = build_result()
    policies = result["policies"]
    strict = result["strict_separation"]

    assert policies["direct_only"]["output_mode"] == "fallback"
    assert policies["at_most_one_probe"]["output_mode"] == "fallback"
    assert policies["fixed_nonadaptive_two_probe"][
        "best_worst_case_regret"
    ] == pytest.approx(0.45)
    adaptive = policies["adaptive_depth_two"]
    assert adaptive["output_mode"] == "sense"
    assert adaptive["worst_case_regret"] == pytest.approx(0.129)
    assert adaptive["first_probe"] == "route"
    assert adaptive["second_probe_by_first_outcome"] == ["x", "y"]
    assert adaptive["uses_highest_information_nuisance_probe"] is False
    assert result["highest_information_probe"] == "nuisance"
    assert strict["direct_and_one_probe_fall_back"] is True
    assert strict["fixed_two_probe_cannot_beat_fallback"] is True
    assert strict["adaptive_policy_is_certified"] is True
    assert strict["complete_state_remains_unidentified"] is True
    assert strict[
        "adaptive_regret_reduction_vs_fallback_fraction"
    ] == pytest.approx(1.0 - 0.129 / 0.45)


def test_controlled_result_is_content_addressed() -> None:
    first = build_result()
    second = build_result()

    assert first == second
    assert len(first["result_id"]) == 64
