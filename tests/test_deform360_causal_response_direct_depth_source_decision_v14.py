from __future__ import annotations

from bayesian_phystwin.deform360_causal_response_direct_depth_source_decision_v14 import (
    evaluate_v14_outcome_blind_gates,
)


def _gate() -> dict[str, int]:
    return {
        "required_prediction_or_exact_fallback_count": 12,
        "maximum_technical_failure_count": 0,
        "minimum_event_admitted_object_count": 6,
    }


def _dispositions(event_count: int) -> list[dict[str, bool]]:
    return [
        {
            "event_admitted": index < event_count,
            "candidate_applied": False,
        }
        for index in range(12)
    ]


def test_target_free_event_gate_can_stop_before_outcome_reveal() -> None:
    result = evaluate_v14_outcome_blind_gates(
        _dispositions(event_count=1),
        _gate(),
    )

    assert result["gates"]["prediction_completeness"] is True
    assert result["gates"]["event_admitted_object_count"] is False
    assert result["source_outcome_authorized"] is False
    assert result["decision"] == "close_v14_without_source_outcome_reveal"
    assert (
        result["gates"]["outcome_dependent_accuracy_safety_and_calibration"]
        == "not_evaluated"
    )


def test_passing_target_free_gates_only_authorizes_source_evaluation() -> None:
    result = evaluate_v14_outcome_blind_gates(
        _dispositions(event_count=6),
        _gate(),
    )

    assert result["source_outcome_authorized"] is True
    assert result["source_gate_status"] == "requires_source_outcome_evaluation"
    assert result["decision"] == "proceed_to_registered_source_outcome_evaluation"


def test_missing_prediction_fails_completeness() -> None:
    result = evaluate_v14_outcome_blind_gates(
        _dispositions(event_count=6)[:-1],
        _gate(),
    )

    assert result["technical_failure_count"] == 1
    assert result["gates"]["prediction_completeness"] is False
    assert result["source_outcome_authorized"] is False
