from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.deform_dlo45_support_aware_outer_v1.run import (
    CONTRACT,
    load_outer_protocol,
    policy_metrics,
    select_source_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/deform_dlo45_support_aware_outer_v1/protocol.json"
EVIDENCE = (
    ROOT
    / "results/science/deform_dlo45_support_aware_outer_v1/retrospective_prototype.json"
)


def test_protocol_keeps_retrospective_information_boundary() -> None:
    value = load_outer_protocol(PROTOCOL)
    assert value["contract"] == CONTRACT
    assert value["outer_model"]["held_outcome_features_forbidden"] is True
    assert value["held_evaluation"]["target_outcomes_used_for_model_selection"] is False
    assert value["held_evaluation"]["target_outcomes_used_for_threshold_selection"] is False
    assert value["held_evaluation"]["target_outcomes_used_for_descriptive_scoring"] is True


def test_source_threshold_is_group_bootstrap_controlled() -> None:
    probabilities = np.asarray([0.01, 0.02, 0.03, 0.20] * 6, dtype=np.float64)
    violations = np.asarray([0, 0, 0, 1] * 6, dtype=np.int64)
    groups = [f"g{index // 4}" for index in range(len(probabilities))]
    result = select_source_threshold(
        probabilities,
        violations,
        groups,
        risk_cap=0.10,
        repetitions=500,
        seed=7,
    )
    assert 0.03 <= float(result["threshold"]) < 0.20
    assert int(result["source_selected_count"]) == 18
    assert float(result["source_empirical_violation_fraction"]) == 0.0
    assert float(result["source_block_bootstrap_upper_095"]) <= 0.10


def test_policy_metrics_use_exact_fallback_on_rejection() -> None:
    rows = [
        {
            "certificate_regret_excess": 0.1,
            "certificate_realized_regret": 0.2,
            "fallback_realized_regret": 0.4,
            "certificate_harmful_vs_fallback": False,
        },
        {
            "certificate_regret_excess": -0.1,
            "certificate_realized_regret": 0.0,
            "fallback_realized_regret": 0.5,
            "certificate_harmful_vs_fallback": False,
        },
    ]
    selected = np.asarray([False, True], dtype=bool)
    result = policy_metrics(rows, selected, tolerance=0.05)
    assert result["selected_count"] == 1
    assert result["support_bound_violation_count"] == 0
    assert result["regret_tolerance_violation_count"] == 0
    assert result["mean_normalized_regret"] == 0.2
    assert result["fallback_mean_normalized_regret"] == 0.45


def test_committed_prototype_evidence_is_bounded() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "retrospective-method-development-positive"
    inner = evidence["held"]["inner"]
    outer = evidence["held"]["outer_plus_inner"]
    assert inner["support_bound_violation_count"] == 45
    assert inner["selected_count"] == 82
    assert outer["support_bound_violation_count"] == 2
    assert outer["selected_count"] == 27
    assert outer["normalized_regret_reduction_vs_fallback"] > 0.0
    assert evidence["negative_control"]["held_accepted_count"] == 0
    assert "Retrospective" in evidence["claim_boundary"]
