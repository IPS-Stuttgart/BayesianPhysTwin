from __future__ import annotations

from dataclasses import replace

import pytest

from bayesian_phystwin.query_update_admission_v1 import (
    QueryUpdateAdmissionCertificateV1,
    QueryUpdateAdmissionPolicyV1,
    QueryUpdateEvidenceV1,
    evaluate_query_update_admission,
)


def _digest(character: str) -> str:
    return character * 64


def _evidence(**updates: object) -> QueryUpdateEvidenceV1:
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


def test_authorized_query_update_selects_candidate() -> None:
    evidence = _evidence()
    certificate = evaluate_query_update_admission(
        evidence,
        policy=QueryUpdateAdmissionPolicyV1(
            minimum_identifiable_fraction=0.5,
            maximum_regret_upper_bound=0.0,
            minimum_expected_information_gain=0.1,
        ),
        metadata={"protocol_id": "query-gate-v1"},
    )

    assert certificate.authorized
    assert not certificate.exact_fallback
    assert certificate.selected_belief_id == evidence.candidate_belief_id
    assert certificate.reasons == ("query-update-authorized",)
    assert certificate.to_record()["artifact_id"] == certificate.artifact_id


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"provider_competence_passed": False},
            "provider-competence-not-passed",
        ),
        (
            {"query_calibration_passed": False},
            "query-calibration-not-passed",
        ),
        (
            {"identifiable_fraction": 0.49},
            "identifiable-query-fraction-below-threshold",
        ),
        (
            {"regret_upper_bound": 0.01},
            "query-regret-upper-bound-exceeds-threshold",
        ),
        (
            {"expected_information_gain": 0.09},
            "query-information-gain-below-threshold",
        ),
    ],
)
def test_each_failed_gate_selects_exact_baseline(
    updates: dict[str, object],
    reason: str,
) -> None:
    evidence = _evidence(**updates)
    certificate = evaluate_query_update_admission(
        evidence,
        policy=QueryUpdateAdmissionPolicyV1(
            minimum_identifiable_fraction=0.5,
            maximum_regret_upper_bound=0.0,
            minimum_expected_information_gain=0.1,
        ),
    )

    assert not certificate.authorized
    assert certificate.exact_fallback
    assert certificate.selected_belief_id == evidence.baseline_belief_id
    assert reason in certificate.reasons


def test_policy_tolerance_and_optional_upstream_gates() -> None:
    tolerance = 1e-6
    policy = QueryUpdateAdmissionPolicyV1(
        minimum_identifiable_fraction=0.5,
        maximum_regret_upper_bound=0.0,
        minimum_expected_information_gain=0.1,
        require_provider_competence=False,
        require_query_calibration=False,
        numerical_tolerance=tolerance,
    )
    evidence = _evidence(
        provider_competence_passed=False,
        query_calibration_passed=False,
        identifiable_fraction=0.5 - tolerance,
        regret_upper_bound=tolerance,
        expected_information_gain=0.1 - tolerance,
    )

    assert evaluate_query_update_admission(evidence, policy=policy).authorized


def test_evidence_requires_distinct_candidate_and_exact_fallback_identity() -> None:
    with pytest.raises(ValueError, match="must differ"):
        _evidence(candidate_belief_id=_digest("2"))
    with pytest.raises(ValueError, match="must equal"):
        _evidence(fallback_belief_id=_digest("9"))


def test_certificate_rejects_forged_decision_fields() -> None:
    evidence = _evidence(regret_upper_bound=0.2)
    policy = QueryUpdateAdmissionPolicyV1()
    certificate = evaluate_query_update_admission(evidence, policy=policy)

    with pytest.raises(ValueError, match="authorized does not match"):
        replace(certificate, authorized=True, exact_fallback=False)
    with pytest.raises(ValueError, match="selected_belief_id contradicts"):
        replace(certificate, selected_belief_id=evidence.candidate_belief_id)
    with pytest.raises(ValueError, match="reasons do not match"):
        QueryUpdateAdmissionCertificateV1(
            policy=policy,
            evidence=evidence,
            authorized=False,
            selected_belief_id=evidence.baseline_belief_id,
            exact_fallback=True,
            reasons=("invented-reason",),
        )


def test_content_identity_is_canonical_and_tamper_evident() -> None:
    first = _evidence(metadata={"b": 2, "a": 1})
    second = _evidence(metadata={"a": 1, "b": 2})
    assert first.artifact_id == second.artifact_id

    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(first, artifact_id=_digest("f"))
