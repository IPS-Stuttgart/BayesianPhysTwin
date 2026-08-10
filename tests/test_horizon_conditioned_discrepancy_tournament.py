from __future__ import annotations

from copy import deepcopy

import pytest

from bayesian_phystwin.discrepancy_candidate_tournament import (
    DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
    analyze_discrepancy_candidate_tournament,
    parse_discrepancy_candidate_tournament,
)


def _candidate(
    candidate_id: str,
    *,
    family: str,
    state_dimension: int,
    parameter_count: int,
    runtime_milliseconds: float,
    covariance_bytes: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "family": family,
        "state_dimension": state_dimension,
        "parameter_count": parameter_count,
        "runtime_milliseconds": runtime_milliseconds,
        "covariance_bytes": covariance_bytes,
        "source_revision": "1" * 40,
        "configuration_sha256": "2" * 64,
        "prediction_artifact_sha256": candidate_id.encode().hex().ljust(64, "0")[:64],
    }


def _record(
    candidate_id: str,
    group: str,
    point_loss: float,
    proper_score: float,
    *,
    accepted: bool = True,
    fallback_point_loss: float = 10.0,
    fallback_proper_score: float = 5.0,
    covered: bool | None = True,
    width: float | None = 2.0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "unit_id": f"{group}-endpoint",
        "group_id": group,
        "horizon": "endpoint",
        "accepted": accepted,
        "point_loss": point_loss,
        "fallback_point_loss": fallback_point_loss,
        "deployed_point_loss": point_loss if accepted else fallback_point_loss,
        "proper_score": proper_score,
        "fallback_proper_score": fallback_proper_score,
        "deployed_proper_score": proper_score if accepted else fallback_proper_score,
        "interval_covered": covered,
        "interval_width": width,
    }


def _payload() -> dict[str, object]:
    candidates = [
        _candidate(
            "physical_fallback",
            family="physical",
            state_dimension=0,
            parameter_count=0,
            runtime_milliseconds=0.0,
            covariance_bytes=0,
        ),
        _candidate(
            "last_residual",
            family="persistence",
            state_dimension=0,
            parameter_count=0,
            runtime_milliseconds=0.1,
            covariance_bytes=0,
        ),
        _candidate(
            "dynamic",
            family="dynamic",
            state_dimension=6,
            parameter_count=8,
            runtime_milliseconds=1.0,
            covariance_bytes=288,
        ),
        _candidate(
            "structured",
            family="structured",
            state_dimension=4,
            parameter_count=6,
            runtime_milliseconds=0.8,
            covariance_bytes=192,
        ),
    ]
    records = []
    for index in range(6):
        group = f"group-{index}"
        records.extend(
            [
                _record(
                    "physical_fallback",
                    group,
                    10.0,
                    5.0,
                    accepted=False,
                ),
                _record("last_residual", group, 8.0, 4.0),
                _record("dynamic", group, 7.6, 3.8),
                _record("structured", group, 7.0, 3.0),
            ]
        )
    return {
        "contract": DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": "discrepancy-source-v1",
        "statistical_unit": "physical-object-or-session",
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
            "interval_semantics_id": "marginal-90-v1",
        },
        "selection": {
            "minimum_group_count": 5,
            "minimum_relative_point_improvement": 0.05,
            "maximum_worst_group_relative_regression": 0.0,
            "maximum_harmful_accepted_count": 0,
            "maximum_mean_proper_score_regression": 0.0,
            "require_paired_point_upper_bound_nonpositive": True,
            "bootstrap_samples": 500,
            "bootstrap_seed": 20260810,
            "require_crossfit_stability": True,
            "nominal_interval_coverage": 0.9,
            "maximum_interval_coverage_shortfall": 0.1,
            "numerical_tolerance": 1e-12,
        },
        "candidates": candidates,
        "records": records,
    }


def _summary(report: dict[str, object], candidate_id: str) -> dict[str, object]:
    summaries = report["candidate_summaries"]
    assert isinstance(summaries, list)
    return next(row for row in summaries if row["candidate_id"] == candidate_id)


def test_selects_stable_candidate_and_never_authorizes_claim() -> None:
    report = analyze_discrepancy_candidate_tournament(_payload())

    assert report["selected_candidate"] == "structured"
    assert report["source_gate_passed"] is True
    assert report["decision"] == "advance-selected-candidate"
    assert report["claim_authorized"] is False
    assert report["cross_fitted"]["stable_selection"] is True
    assert report["cross_fitted"]["held_group_nonregression"] is True
    assert len(report["report_id"]) == 64
    assert _summary(report, "structured")["eligible"] is True
    assert _summary(report, "physical_fallback")["eligible"] is False


def test_harmful_accepted_update_rejects_mean_improving_candidate() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    row = next(
        record
        for record in records
        if record["candidate_id"] == "structured" and record["group_id"] == "group-0"
    )
    row["point_loss"] = 11.0
    row["deployed_point_loss"] = 11.0
    row["proper_score"] = 3.0
    row["deployed_proper_score"] = 3.0

    report = analyze_discrepancy_candidate_tournament(payload)

    structured = _summary(report, "structured")
    assert structured["eligible"] is False
    assert "harmful-accepted-updates" in structured["eligibility_failures"]
    assert report["selected_candidate"] == "dynamic"


def test_exact_fallback_and_matched_roster_fail_closed() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    row = next(
        record
        for record in records
        if record["candidate_id"] == "structured" and record["group_id"] == "group-0"
    )
    row["accepted"] = False
    with pytest.raises(ValueError, match="exact fallback"):
        parse_discrepancy_candidate_tournament(payload)

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    records.pop()
    with pytest.raises(ValueError, match="every registered candidate"):
        parse_discrepancy_candidate_tournament(payload)


def test_information_boundary_and_interval_coverage_fail_closed() -> None:
    payload = _payload()
    boundary = payload["information_boundary"]
    assert isinstance(boundary, dict)
    boundary["candidate_generation_used_scored_targets"] = True
    with pytest.raises(ValueError, match="used scored targets"):
        parse_discrepancy_candidate_tournament(payload)

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    for record in records:
        if record["candidate_id"] == "structured" and record["group_id"] != "group-0":
            record["interval_covered"] = False
    report = analyze_discrepancy_candidate_tournament(payload)
    structured = _summary(report, "structured")
    assert structured["eligible"] is False
    assert "interval-undercoverage" in structured["eligibility_failures"]


def test_complexity_breaks_exact_score_tie() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    for record in records:
        if record["candidate_id"] == "dynamic":
            record["point_loss"] = 7.0
            record["deployed_point_loss"] = 7.0
            record["proper_score"] = 3.0
            record["deployed_proper_score"] = 3.0
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    dynamic = next(row for row in candidates if row["candidate_id"] == "dynamic")
    structured = next(row for row in candidates if row["candidate_id"] == "structured")
    dynamic["state_dimension"] = structured["state_dimension"] + 1

    report = analyze_discrepancy_candidate_tournament(payload)

    assert report["selected_candidate"] == "structured"


def test_retains_reference_when_no_challenger_is_eligible() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    for record in records:
        if record["candidate_id"] in {"dynamic", "structured"}:
            record["point_loss"] = 8.5
            record["deployed_point_loss"] = 8.5
            record["proper_score"] = 4.5
            record["deployed_proper_score"] = 4.5

    report = analyze_discrepancy_candidate_tournament(payload)

    assert report["selected_candidate"] == "last_residual"
    assert report["source_gate_passed"] is False
    assert report["decision"] == "retain-reference-candidate"


def test_report_identity_is_deterministic() -> None:
    first = analyze_discrepancy_candidate_tournament(_payload())
    second = analyze_discrepancy_candidate_tournament(deepcopy(_payload()))
    assert first == second


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("records"), "input fields changed"),
        (
            lambda payload: payload.__setitem__("contract", "unsupported"),
            "unsupported tournament input contract",
        ),
        (
            lambda payload: payload.__setitem__("schema_version", True),
            "schema_version must be the integer 1",
        ),
        (
            lambda payload: payload.__setitem__("split", "target"),
            "split must be source-only",
        ),
        (
            lambda payload: payload["information_boundary"].__setitem__(
                "unexpected", False
            ),
            "information_boundary fields changed",
        ),
    ],
)
def test_top_level_contracts_fail_closed(mutate, message: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        parse_discrepancy_candidate_tournament(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["candidates"][0].pop("family"),
            "candidates\\[0\\] fields changed",
        ),
        (
            lambda payload: payload["selection"].pop("bootstrap_seed"),
            "selection fields changed",
        ),
        (
            lambda payload: payload["records"][0].pop("horizon"),
            "records\\[0\\] fields changed",
        ),
        (
            lambda payload: payload["records"][0].__setitem__("interval_width", None),
            "provide both interval fields or neither",
        ),
        (
            lambda payload: payload["records"][0].__setitem__(
                "deployed_proper_score", 6.0
            ),
            "deployed_proper_score violates exact fallback",
        ),
    ],
)
def test_nested_contracts_fail_closed(mutate, message: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        parse_discrepancy_candidate_tournament(payload)


def test_interval_free_non_crossfit_tournament_is_supported() -> None:
    payload = _payload()
    selection = payload["selection"]
    records = payload["records"]
    assert isinstance(selection, dict)
    assert isinstance(records, list)
    selection["require_crossfit_stability"] = False
    selection["minimum_group_count"] = 3
    selection["nominal_interval_coverage"] = None
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["interval_semantics_id"] = "none"
    for record in records:
        record["interval_covered"] = None
        record["interval_width"] = None

    report = analyze_discrepancy_candidate_tournament(payload)

    assert report["source_gate_passed"] is True
    selected = report["selected_candidate_summary"]
    assert selected["interval_coverage"] is None
    assert selected["mean_interval_width"] is None


def test_interval_coverage_is_equal_group_weighted() -> None:
    payload = _payload()
    records = payload["records"]
    selection = payload["selection"]
    assert isinstance(records, list)
    assert isinstance(selection, dict)
    selection["maximum_interval_coverage_shortfall"] = 0.0
    extras = []
    for record in records:
        if record["group_id"] != "group-0":
            continue
        extra = deepcopy(record)
        extra["unit_id"] = "group-0-extra"
        if extra["candidate_id"] == "structured":
            extra["interval_covered"] = False
        extras.append(extra)
    records.extend(extras)

    report = analyze_discrepancy_candidate_tournament(payload)

    structured = _summary(report, "structured")
    assert structured["interval_coverage"] == pytest.approx(11.0 / 12.0)
    assert structured["eligible"] is True


def test_interval_width_precedes_complexity_as_tie_break() -> None:
    payload = _payload()
    records = payload["records"]
    candidates = payload["candidates"]
    assert isinstance(records, list)
    assert isinstance(candidates, list)
    for record in records:
        if record["candidate_id"] == "dynamic":
            record["point_loss"] = 7.0
            record["deployed_point_loss"] = 7.0
            record["proper_score"] = 3.0
            record["deployed_proper_score"] = 3.0
            record["interval_width"] = 1.0
        elif record["candidate_id"] == "structured":
            record["interval_width"] = 2.0
    dynamic = next(row for row in candidates if row["candidate_id"] == "dynamic")
    structured = next(row for row in candidates if row["candidate_id"] == "structured")
    dynamic["state_dimension"] = structured["state_dimension"] + 100

    report = analyze_discrepancy_candidate_tournament(payload)

    assert report["selected_candidate"] == "dynamic"
