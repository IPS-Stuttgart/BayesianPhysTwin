from __future__ import annotations

import pytest

from bayesian_phystwin.discrepancy_candidate_tournament import (
    DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
    analyze_discrepancy_candidate_tournament,
    parse_discrepancy_candidate_tournament,
)


def _candidate(candidate_id: str, digest_character: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "family": candidate_id,
        "state_dimension": 1,
        "parameter_count": 1,
        "runtime_milliseconds": 1.0,
        "covariance_bytes": 24,
        "source_revision": "1" * 40,
        "configuration_sha256": "2" * 64,
        "prediction_artifact_sha256": digest_character * 64,
    }


def _record(
    candidate_id: str,
    group_id: str,
    point_loss: float,
    proper_score: float,
    *,
    accepted: bool,
    covered: bool | None,
    width: float | None,
) -> dict[str, object]:
    fallback_point_loss = 10.0
    fallback_proper_score = 5.0
    return {
        "candidate_id": candidate_id,
        "unit_id": f"{group_id}-endpoint",
        "group_id": group_id,
        "horizon": "endpoint",
        "accepted": accepted,
        "point_loss": point_loss,
        "fallback_point_loss": fallback_point_loss,
        "deployed_point_loss": point_loss if accepted else fallback_point_loss,
        "proper_score": proper_score,
        "fallback_proper_score": fallback_proper_score,
        "deployed_proper_score": (
            proper_score if accepted else fallback_proper_score
        ),
        "interval_covered": covered,
        "interval_width": width,
    }


def _payload(*, intervals: bool = False) -> dict[str, object]:
    candidates = [
        _candidate("physical_fallback", "a"),
        _candidate("last_residual", "b"),
        _candidate("challenger", "c"),
    ]
    records: list[dict[str, object]] = []
    for index in range(4):
        group_id = f"object-{index}"
        fallback_interval = False if intervals else None
        fallback_width = 3.0 if intervals else None
        records.extend(
            [
                _record(
                    "physical_fallback",
                    group_id,
                    10.0,
                    5.0,
                    accepted=False,
                    covered=fallback_interval,
                    width=fallback_width,
                ),
                _record(
                    "last_residual",
                    group_id,
                    8.0,
                    4.0,
                    accepted=True,
                    covered=True if intervals else None,
                    width=2.0 if intervals else None,
                ),
                _record(
                    "challenger",
                    group_id,
                    7.0,
                    3.0,
                    accepted=True,
                    covered=True if intervals else None,
                    width=1.5 if intervals else None,
                ),
            ]
        )
    return {
        "contract": DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": "source-tournament-v1",
        "statistical_unit": "physical-object",
        "split": "source-only",
        "reference_candidate": "last_residual",
        "physical_fallback_candidate": "physical_fallback",
        "information_boundary": {
            "candidate_predictions_sealed_before_scoring": True,
            "candidate_generation_used_scored_targets": False,
            "future_observations_used": False,
            "confirmation_payloads_opened": False,
            "replacement_allowed": False,
        },
        "evaluation": {
            "evaluator_revision": "3" * 40,
            "scoring_policy_sha256": "4" * 64,
            "scored_unit_roster_sha256": "5" * 64,
            "physical_fallback_artifact_sha256": "6" * 64,
            "prediction_barrier_sha256": "7" * 64,
            "point_loss_id": "endpoint-rmse-v1",
            "proper_score_id": "gaussian-nll-v1",
            "interval_semantics_id": "marginal-90-v1" if intervals else "none",
        },
        "selection": {
            "minimum_group_count": 3,
            "minimum_relative_point_improvement": 0.05,
            "maximum_worst_group_relative_regression": 0.0,
            "maximum_harmful_accepted_count": 0,
            "maximum_mean_proper_score_regression": 0.0,
            "require_paired_point_upper_bound_nonpositive": False,
            "bootstrap_samples": 100,
            "bootstrap_seed": 20260810,
            "require_crossfit_stability": False,
            "nominal_interval_coverage": 0.9 if intervals else None,
            "maximum_interval_coverage_shortfall": 0.1 if intervals else 0.0,
            "numerical_tolerance": 1e-12,
        },
        "candidates": candidates,
        "records": records,
    }


def test_public_analysis_uses_hardened_validation() -> None:
    report = analyze_discrepancy_candidate_tournament(_payload())
    assert report["selected_candidate"] == "challenger"
    assert report["source_gate_passed"] is True


def test_rejects_frame_level_statistical_unit() -> None:
    payload = _payload()
    payload["statistical_unit"] = "frame"
    with pytest.raises(ValueError, match="physical object or acquisition session"):
        parse_discrepancy_candidate_tournament(payload)


def test_rejects_candidate_dependent_interval_availability() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    for record in records:
        if record["candidate_id"] == "challenger":
            record["interval_covered"] = True
            record["interval_width"] = 1.0
    with pytest.raises(ValueError, match="differ in interval availability"):
        parse_discrepancy_candidate_tournament(payload)


def test_rejected_candidate_must_retain_fallback_interval() -> None:
    payload = _payload(intervals=True)
    records = payload["records"]
    assert isinstance(records, list)
    record = next(
        row
        for row in records
        if row["candidate_id"] == "challenger"
        and row["group_id"] == "object-0"
    )
    record["accepted"] = False
    record["deployed_point_loss"] = record["fallback_point_loss"]
    record["deployed_proper_score"] = record["fallback_proper_score"]
    with pytest.raises(ValueError, match="interval coverage violates exact fallback"):
        parse_discrepancy_candidate_tournament(payload)


def test_interval_shortfall_cannot_exceed_nominal_coverage() -> None:
    payload = _payload(intervals=True)
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selection["maximum_interval_coverage_shortfall"] = 0.95
    with pytest.raises(ValueError, match="cannot exceed nominal coverage"):
        parse_discrepancy_candidate_tournament(payload)


def test_bootstrap_allocation_is_bounded_before_analysis() -> None:
    payload = _payload()
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selection["bootstrap_samples"] = 2_500_001
    with pytest.raises(ValueError, match="resource budget"):
        parse_discrepancy_candidate_tournament(payload)
