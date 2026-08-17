from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

import bayesian_phystwin.query_update_admission_v1 as admission
from bayesian_phystwin.query_update_admission_v1 import (
    QueryUpdateAdmissionPolicyV1,
    QueryUpdateEvidenceV1,
    build_query_update_evidence_from_records,
    evaluate_query_update_admission,
)


def _digest(character: str) -> str:
    return character * 64


def _unbound_evidence(**updates: object) -> QueryUpdateEvidenceV1:
    values: dict[str, object] = {
        "physical_query_id": _digest("1"),
        "baseline_belief_id": _digest("2"),
        "candidate_belief_id": _digest("3"),
        "fallback_belief_id": _digest("2"),
        "provider_decision_id": _digest("4"),
        "query_calibration_id": _digest("5"),
        "identifiability_diagnostic_id": _digest("6"),
        "regret_evidence_id": _digest("7"),
        "information_gain_evidence_id": _digest("8"),
        "provider_competence_passed": True,
        "query_calibration_passed": True,
        "identifiable_fraction": 0.8,
        "regret_upper_bound": -0.1,
        "expected_information_gain": 0.2,
    }
    values.update(updates)
    return QueryUpdateEvidenceV1(**values)  # type: ignore[arg-type]


def _context() -> dict[str, object]:
    return {
        "physical_query_id": _digest("1"),
        "baseline_belief_id": _digest("2"),
        "candidate_belief_id": _digest("3"),
        "source_protocol_id": _digest("9"),
        "grouping_rule_id": _digest("a"),
        "independent_group_count": 12,
    }


def _record(**values: object) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema": "test.bound-query-evidence",
        "schema_version": 1,
        **_context(),
        **values,
    }
    return {**descriptor, "artifact_id": admission.content_id(descriptor)}


def _bound_evidence() -> QueryUpdateEvidenceV1:
    context = _context()
    return build_query_update_evidence_from_records(
        physical_query_id=str(context["physical_query_id"]),
        baseline_belief_id=str(context["baseline_belief_id"]),
        candidate_belief_id=str(context["candidate_belief_id"]),
        fallback_belief_id=str(context["baseline_belief_id"]),
        source_protocol_id=str(context["source_protocol_id"]),
        grouping_rule_id=str(context["grouping_rule_id"]),
        independent_group_count=int(context["independent_group_count"]),
        provider_decision=_record(provider_competence_passed=True),
        query_calibration=_record(query_calibration_passed=True),
        identifiability_diagnostic=_record(identifiable_fraction=0.8),
        regret_evidence=_record(regret_upper_bound=-0.1),
        information_gain_evidence=_record(expected_information_gain=0.2),
    )


def test_bound_record_builder_binds_source_context_and_artifact_values() -> None:
    evidence = _bound_evidence()

    assert evidence.source_context_bound
    assert evidence.source_protocol_id == _digest("9")
    assert evidence.grouping_rule_id == _digest("a")
    assert evidence.independent_group_count == 12
    assert evidence.provider_competence_passed
    assert evidence.query_calibration_passed
    assert evidence.identifiable_fraction == 0.8
    assert evidence.regret_upper_bound == -0.1
    assert evidence.expected_information_gain == 0.2

    decision = evaluate_query_update_admission(
        evidence,
        policy=QueryUpdateAdmissionPolicyV1(require_source_context=True),
    )
    assert decision.authorized
    assert decision.to_record()["source_protocol_id"] == _digest("9")
    assert decision.to_record()["independent_group_count"] == 12


def test_bound_record_builder_rejects_tamper_and_cross_context_drift() -> None:
    provider = _record(provider_competence_passed=True)
    tampered = {**provider, "provider_competence_passed": False}
    context = _context()
    kwargs = {
        "physical_query_id": str(context["physical_query_id"]),
        "baseline_belief_id": str(context["baseline_belief_id"]),
        "candidate_belief_id": str(context["candidate_belief_id"]),
        "fallback_belief_id": str(context["baseline_belief_id"]),
        "source_protocol_id": str(context["source_protocol_id"]),
        "grouping_rule_id": str(context["grouping_rule_id"]),
        "independent_group_count": int(context["independent_group_count"]),
        "query_calibration": _record(query_calibration_passed=True),
        "identifiability_diagnostic": _record(identifiable_fraction=0.8),
        "regret_evidence": _record(regret_upper_bound=-0.1),
        "information_gain_evidence": _record(expected_information_gain=0.2),
    }

    with pytest.raises(ValueError, match="artifact_id does not match"):
        build_query_update_evidence_from_records(
            provider_decision=tampered,
            **kwargs,
        )

    wrong_context = _record(provider_competence_passed=True)
    wrong_context["source_protocol_id"] = _digest("b")
    descriptor = {
        key: value for key, value in wrong_context.items() if key != "artifact_id"
    }
    wrong_context["artifact_id"] = admission.content_id(descriptor)
    with pytest.raises(ValueError, match="source_protocol_id does not match"):
        build_query_update_evidence_from_records(
            provider_decision=wrong_context,
            **kwargs,
        )


def test_claim_bearing_policy_fails_closed_without_source_context() -> None:
    decision = evaluate_query_update_admission(
        _unbound_evidence(),
        policy=QueryUpdateAdmissionPolicyV1(require_source_context=True),
    )

    assert not decision.authorized
    assert decision.exact_fallback
    assert decision.reasons == ("source-context-not-bound",)


def test_split_tolerances_apply_only_to_their_own_quantity() -> None:
    evidence = _unbound_evidence(
        identifiable_fraction=0.499,
        regret_upper_bound=0.001,
        expected_information_gain=0.099,
    )
    ident_only = QueryUpdateAdmissionPolicyV1(
        minimum_identifiable_fraction=0.5,
        maximum_regret_upper_bound=0.0,
        minimum_expected_information_gain=0.1,
        identifiable_fraction_tolerance=0.001,
        regret_tolerance=0.0,
        information_gain_tolerance=0.0,
    )
    decision = evaluate_query_update_admission(evidence, policy=ident_only)

    assert "identifiable-query-fraction-below-threshold" not in decision.reasons
    assert "query-regret-upper-bound-exceeds-threshold" in decision.reasons
    assert "query-information-gain-below-threshold" in decision.reasons


def test_legacy_uniform_tolerance_is_canonicalized_and_cannot_be_mixed() -> None:
    uniform = QueryUpdateAdmissionPolicyV1(numerical_tolerance=1e-6)
    assert uniform.identifiable_fraction_tolerance == 1e-6
    assert uniform.regret_tolerance == 1e-6
    assert uniform.information_gain_tolerance == 1e-6
    assert "numerical_tolerance" not in uniform.descriptor()

    with pytest.raises(ValueError, match="cannot be combined"):
        QueryUpdateAdmissionPolicyV1(
            numerical_tolerance=1e-6,
            regret_tolerance=1e-5,
        )


def test_source_context_fields_are_all_or_none_and_content_addressed() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        replace(
            _unbound_evidence(),
            source_protocol_id=_digest("9"),
        )

    evidence = replace(
        _unbound_evidence(),
        source_protocol_id=_digest("9"),
        grouping_rule_id=_digest("a"),
        independent_group_count=12,
        artifact_id=None,
    )
    assert evidence.source_context_bound
    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(evidence, independent_group_count=13)
