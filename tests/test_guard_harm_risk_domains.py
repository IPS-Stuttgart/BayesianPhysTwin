from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from bayesian_phystwin.calibration_domain_guard import (
    CalibrationDomainGuardCertificateV1,
    fit_calibration_domain_guard,
)
from bayesian_phystwin.guard_harm_risk import certify_guard_harm_risk
from bayesian_phystwin.guard_harm_risk_domains import (
    MULTIPLICITY_METHOD,
    DomainGuardHarmRiskCertificateV1,
    certify_domain_guard_harm_risk,
    domain_harm_risk_policy_id,
    select_domain_guard_harm_risk_belief,
)

CALIBRATION_PARTITION_ID = "a" * 64
CERTIFICATION_PARTITION_ID = "b" * 64
THRESHOLD_SOURCE_ID = "c" * 64
COMMON_DOMAIN_ID = "d" * 64


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def _domain_guard(
    *,
    supported_domains: tuple[str, ...] = ("dynamic", "oscillatory"),
    frozen_before: bool = True,
    application_used: bool = False,
    independent: bool = True,
) -> CalibrationDomainGuardCertificateV1:
    all_domains = ("dynamic", "oscillatory")
    group_ids: list[str] = []
    domain_ids: list[str] = []
    candidate_losses: list[float] = []
    fallback_losses: list[float] = []
    for domain in all_domains:
        supported = domain in supported_domains
        for index in range(3):
            group_ids.append(f"calibration-{domain}-{index}")
            domain_ids.append(domain)
            fallback_losses.append(1.0)
            candidate_losses.append(0.8 if supported else 1.2)
    return fit_calibration_domain_guard(
        calibration_partition_id=CALIBRATION_PARTITION_ID,
        statistical_unit="independent-physical-session",
        metric="endpoint-rmse-m",
        group_ids=tuple(group_ids),
        domain_ids=tuple(domain_ids),
        candidate_losses=np.asarray(candidate_losses),
        fallback_losses=np.asarray(fallback_losses),
        guard_frozen_before_application_outcomes=frozen_before,
        application_outcomes_used_for_guard_selection=application_used,
        calibration_groups_independent=independent,
    )


def _evidence(
    *,
    harmful_domain: str | None = None,
    permutation: np.ndarray | None = None,
) -> dict[str, object]:
    group_ids: list[str] = []
    domain_ids: list[str] = []
    candidate_losses: list[float] = []
    for domain in ("dynamic", "oscillatory"):
        for index in range(5):
            group_ids.append(f"certification-{domain}-{index}")
            domain_ids.append(domain)
            harmful = harmful_domain == domain and index == 0
            candidate_losses.append(1.2 if harmful else 0.9)
    count = len(group_ids)
    order = np.arange(count) if permutation is None else permutation
    return {
        "group_ids": tuple(group_ids[int(index)] for index in order),
        "domain_ids": tuple(domain_ids[int(index)] for index in order),
        "risk_scores": np.zeros(count, dtype=np.float64)[order],
        "candidate_losses": np.asarray(candidate_losses, dtype=np.float64)[order],
        "fallback_losses": np.ones(count, dtype=np.float64)[order],
        "fallback_identity_verified": np.ones(count, dtype=np.bool_)[order],
    }


def _certificate(
    *,
    guard: CalibrationDomainGuardCertificateV1 | None = None,
    harmful_domain: str | None = None,
    metadata: dict[str, object] | None = None,
    permutation: np.ndarray | None = None,
) -> DomainGuardHarmRiskCertificateV1:
    selected_guard = _domain_guard() if guard is None else guard
    evidence = _evidence(
        harmful_domain=harmful_domain,
        permutation=permutation,
    )
    return certify_domain_guard_harm_risk(
        domain_guard_certificate=selected_guard,
        threshold_source_artifact_id=THRESHOLD_SOURCE_ID,
        certification_partition_id=CERTIFICATION_PARTITION_ID,
        threshold_selection_group_ids=("threshold-selection-1",),
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.6,
        family_confidence_level=0.95,
        minimum_accepted_group_count=5,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
        metadata={} if metadata is None else metadata,
        **evidence,
    )


def _recertify(
    guard: CalibrationDomainGuardCertificateV1,
    domain: str,
    certificate: object,
    *,
    family_confidence_level: float = 0.95,
    confidence_level: float | None = None,
    group_ids: tuple[str, ...] | None = None,
    threshold_selection_group_ids: tuple[str, ...] | None = None,
    threshold_source_artifact_id: str | None = None,
    certification_partition_id: str | None = None,
    threshold: float | None = None,
    harm_margin: float | None = None,
    target_harm_probability: float | None = None,
    minimum_accepted_group_count: int | None = None,
):
    from bayesian_phystwin.guard_harm_risk import GuardHarmRiskCertificateV1

    assert isinstance(certificate, GuardHarmRiskCertificateV1)
    decision = guard.decision_for_domain(domain)
    assert decision is not None
    source_id = (
        certificate.threshold_source_artifact_id
        if threshold_source_artifact_id is None
        else threshold_source_artifact_id
    )
    threshold_value = certificate.threshold if threshold is None else threshold
    margin = certificate.harm_margin if harm_margin is None else harm_margin
    target = (
        certificate.target_harm_probability
        if target_harm_probability is None
        else target_harm_probability
    )
    per_domain_confidence = (
        certificate.confidence_level
        if confidence_level is None
        else confidence_level
    )
    policy_id = domain_harm_risk_policy_id(
        domain_guard_certificate_id=str(guard.artifact_id),
        domain_decision_id=str(decision.artifact_id),
        domain_id=domain,
        threshold_source_artifact_id=source_id,
        statistical_unit=certificate.statistical_unit,
        metric=certificate.metric,
        threshold=threshold_value,
        harm_margin=margin,
        target_harm_probability=target,
        family_confidence_level=family_confidence_level,
        supported_domain_count=len(guard.supported_domains),
    )
    selected_group_ids = certificate.group_ids if group_ids is None else group_ids
    count = len(selected_group_ids)
    return certify_guard_harm_risk(
        guard_policy_id=policy_id,
        threshold_source_artifact_id=source_id,
        certification_partition_id=(
            certificate.certification_partition_id
            if certification_partition_id is None
            else certification_partition_id
        ),
        statistical_unit=certificate.statistical_unit,
        metric=certificate.metric,
        threshold_selection_group_ids=(
            certificate.threshold_selection_group_ids
            if threshold_selection_group_ids is None
            else threshold_selection_group_ids
        ),
        group_ids=selected_group_ids,
        risk_scores=np.zeros(count),
        candidate_losses=np.full(count, 0.9),
        fallback_losses=np.ones(count),
        fallback_identity_verified=np.ones(count, dtype=np.bool_),
        threshold=threshold_value,
        harm_margin=margin,
        target_harm_probability=target,
        confidence_level=per_domain_confidence,
        minimum_accepted_group_count=(
            certificate.minimum_accepted_group_count
            if minimum_accepted_group_count is None
            else minimum_accepted_group_count
        ),
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )


def test_certifies_every_supported_domain_with_simultaneous_coverage() -> None:
    certificate = _certificate()

    assert certificate.supported_domains == ("dynamic", "oscillatory")
    assert certificate.certified_domains == ("dynamic", "oscillatory")
    assert certificate.failed_domains == ()
    assert certificate.all_supported_domains_certified
    assert certificate.deployment_admissible
    assert certificate.per_domain_confidence_level == pytest.approx(0.975)
    assert certificate.descriptor()["multiplicity_method"] == MULTIPLICITY_METHOD


def test_pooled_pass_can_hide_a_harmful_minority_domain() -> None:
    evidence = _evidence(harmful_domain="oscillatory")
    pooled = certify_guard_harm_risk(
        guard_policy_id="e" * 64,
        threshold_source_artifact_id=THRESHOLD_SOURCE_ID,
        certification_partition_id=CERTIFICATION_PARTITION_ID,
        statistical_unit="independent-physical-session",
        metric="endpoint-rmse-m",
        threshold_selection_group_ids=("threshold-selection-1",),
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.6,
        confidence_level=0.95,
        minimum_accepted_group_count=10,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
        **{key: value for key, value in evidence.items() if key != "domain_ids"},
    )
    simultaneous = _certificate(harmful_domain="oscillatory")

    assert pooled.certified
    assert pooled.harmful_accepted_count == 1
    assert simultaneous.certified_domains == ("dynamic",)
    assert simultaneous.failed_domains == ("oscillatory",)
    assert not simultaneous.deployment_admissible


def test_certificate_is_permutation_invariant() -> None:
    first = _certificate()
    second = _certificate(permutation=np.asarray([9, 0, 6, 2, 8, 1, 7, 3, 5, 4]))

    assert second.artifact_id == first.artifact_id


def test_all_domains_certified_selects_candidate() -> None:
    baseline = _Belief("1" * 64)
    candidate = _Belief("2" * 64)

    selected, record = select_domain_guard_harm_risk_belief(
        baseline,
        candidate,
        _certificate(),
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )

    assert selected is candidate
    assert record.selected_candidate
    assert record.metadata["routing_reason"] == "domain-harm-risk-authorized"


def test_failure_in_other_domain_forces_global_exact_fallback() -> None:
    baseline = _Belief("1" * 64)
    candidate = _Belief("2" * 64)

    selected, record = select_domain_guard_harm_risk_belief(
        baseline,
        candidate,
        _certificate(harmful_domain="oscillatory"),
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )

    assert selected is baseline
    assert not record.selected_candidate
    assert record.metadata["routing_reason"] == "cross-domain-harm-risk-rejected"


def test_current_domain_harm_failure_forces_exact_fallback() -> None:
    baseline = _Belief("1" * 64)
    candidate = _Belief("2" * 64)

    selected, record = select_domain_guard_harm_risk_belief(
        baseline,
        candidate,
        _certificate(harmful_domain="oscillatory"),
        domain_id="oscillatory",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )

    assert selected is baseline
    assert record.metadata["routing_reason"] == "domain-harm-risk-rejected"


def test_unsupported_and_unknown_domains_force_exact_fallback() -> None:
    guard = _domain_guard(supported_domains=("dynamic",))
    evidence = _evidence()
    dynamic = np.asarray(evidence["domain_ids"]) == "dynamic"
    certificate = certify_domain_guard_harm_risk(
        domain_guard_certificate=guard,
        threshold_source_artifact_id=THRESHOLD_SOURCE_ID,
        certification_partition_id=CERTIFICATION_PARTITION_ID,
        threshold_selection_group_ids=("threshold-selection-1",),
        group_ids=tuple(np.asarray(evidence["group_ids"])[dynamic]),
        domain_ids=tuple(np.asarray(evidence["domain_ids"])[dynamic]),
        risk_scores=np.asarray(evidence["risk_scores"])[dynamic],
        candidate_losses=np.asarray(evidence["candidate_losses"])[dynamic],
        fallback_losses=np.asarray(evidence["fallback_losses"])[dynamic],
        fallback_identity_verified=np.asarray(
            evidence["fallback_identity_verified"]
        )[dynamic],
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.8,
        family_confidence_level=0.95,
        minimum_accepted_group_count=5,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )
    baseline = _Belief("1" * 64)
    candidate = _Belief("2" * 64)

    unsupported, unsupported_record = select_domain_guard_harm_risk_belief(
        baseline,
        candidate,
        certificate,
        domain_id="oscillatory",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )
    unknown, unknown_record = select_domain_guard_harm_risk_belief(
        baseline,
        candidate,
        certificate,
        domain_id="unseen",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )

    assert unsupported is baseline
    assert unknown is baseline
    assert unsupported_record.metadata["routing_reason"] == (
        "calibration-domain-rejected"
    )
    assert unknown_record.metadata["routing_reason"] == (
        "unknown-calibration-domain"
    )


def test_inference_and_information_boundaries_override_certification() -> None:
    baseline = _Belief("1" * 64)
    candidate = _Belief("2" * 64)
    ordinary = _certificate()
    retrospective = _certificate(guard=_domain_guard(frozen_before=False))

    inference_selected, inference_record = select_domain_guard_harm_risk_belief(
        baseline,
        candidate,
        ordinary,
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=False,
    )
    retrospective_selected, retrospective_record = (
        select_domain_guard_harm_risk_belief(
            baseline,
            candidate,
            retrospective,
            domain_id="dynamic",
            common_domain_id=COMMON_DOMAIN_ID,
            inference_admissible=True,
        )
    )

    assert inference_selected is baseline
    assert retrospective_selected is baseline
    assert inference_record.metadata["routing_reason"] == "inference-rejected"
    assert retrospective_record.metadata["routing_reason"] == (
        "calibration-information-boundary-rejected"
    )


def test_policy_identity_binds_domain_and_threshold_source() -> None:
    guard = _domain_guard()
    decision = guard.decision_for_domain("dynamic")
    assert decision is not None
    arguments = {
        "domain_guard_certificate_id": str(guard.artifact_id),
        "domain_decision_id": str(decision.artifact_id),
        "domain_id": "dynamic",
        "threshold_source_artifact_id": THRESHOLD_SOURCE_ID,
        "statistical_unit": guard.statistical_unit,
        "metric": guard.metric,
        "threshold": 0.5,
        "harm_margin": 0.0,
        "target_harm_probability": 0.6,
        "family_confidence_level": 0.95,
        "supported_domain_count": 2,
    }
    first = domain_harm_risk_policy_id(**arguments)
    second = domain_harm_risk_policy_id(
        **{**arguments, "domain_id": "oscillatory"}
    )
    third = domain_harm_risk_policy_id(
        **{**arguments, "threshold_source_artifact_id": "f" * 64}
    )

    assert first != second
    assert first != third


def test_composite_rejects_wrong_policy_and_confidence() -> None:
    certificate = _certificate()
    certificates = dict(certificate.domain_certificates)
    certificates["dynamic"] = replace(
        certificates["dynamic"],
        guard_policy_id="f" * 64,
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="not bound"):
        DomainGuardHarmRiskCertificateV1(
            domain_guard_certificate=certificate.domain_guard_certificate,
            domain_certificates=certificates,
            family_confidence_level=0.95,
        )

    certificates = dict(certificate.domain_certificates)
    certificates["dynamic"] = _recertify(
        certificate.domain_guard_certificate,
        "dynamic",
        certificates["dynamic"],
        confidence_level=0.95,
    )
    with pytest.raises(ValueError, match="Bonferroni"):
        DomainGuardHarmRiskCertificateV1(
            domain_guard_certificate=certificate.domain_guard_certificate,
            domain_certificates=certificates,
            family_confidence_level=0.95,
        )


def test_composite_rejects_roster_and_group_leakage() -> None:
    certificate = _certificate()
    certificates = dict(certificate.domain_certificates)
    certificates.pop("oscillatory")
    with pytest.raises(ValueError, match="roster"):
        DomainGuardHarmRiskCertificateV1(
            domain_guard_certificate=certificate.domain_guard_certificate,
            domain_certificates=certificates,
            family_confidence_level=0.95,
        )

    certificates = dict(certificate.domain_certificates)
    certificates["dynamic"] = _recertify(
        certificate.domain_guard_certificate,
        "dynamic",
        certificates["dynamic"],
        group_ids=(
            "calibration-dynamic-0",
            "replacement-1",
            "replacement-2",
            "replacement-3",
            "replacement-4",
        ),
    )
    with pytest.raises(ValueError, match="calibration and certification"):
        DomainGuardHarmRiskCertificateV1(
            domain_guard_certificate=certificate.domain_guard_certificate,
            domain_certificates=certificates,
            family_confidence_level=0.95,
        )


def test_composite_rejects_cross_domain_duplicate_and_threshold_leakage() -> None:
    certificate = _certificate()
    certificates = dict(certificate.domain_certificates)
    certificates["oscillatory"] = _recertify(
        certificate.domain_guard_certificate,
        "oscillatory",
        certificates["oscillatory"],
        group_ids=certificates["dynamic"].group_ids,
    )
    with pytest.raises(ValueError, match="exactly one domain"):
        DomainGuardHarmRiskCertificateV1(
            domain_guard_certificate=certificate.domain_guard_certificate,
            domain_certificates=certificates,
            family_confidence_level=0.95,
        )

    certificates = dict(certificate.domain_certificates)
    leak = certificates["oscillatory"].group_ids[0]
    certificates["dynamic"] = _recertify(
        certificate.domain_guard_certificate,
        "dynamic",
        certificates["dynamic"],
        threshold_selection_group_ids=(leak,),
    )
    with pytest.raises(ValueError, match="globally disjoint"):
        DomainGuardHarmRiskCertificateV1(
            domain_guard_certificate=certificate.domain_guard_certificate,
            domain_certificates=certificates,
            family_confidence_level=0.95,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("threshold_source_artifact_id", "f" * 64, "threshold_source"),
        ("certification_partition_id", "f" * 64, "certification_partition"),
        ("threshold", 0.4, "threshold"),
        ("harm_margin", 0.1, "harm_margin"),
        ("target_harm_probability", 0.7, "target_harm_probability"),
        ("minimum_accepted_group_count", 4, "minimum_accepted_group_count"),
    ],
)
def test_composite_rejects_shared_policy_drift(
    field: str,
    value: object,
    match: str,
) -> None:
    certificate = _certificate()
    certificates = dict(certificate.domain_certificates)
    certificates["oscillatory"] = _recertify(
        certificate.domain_guard_certificate,
        "oscillatory",
        certificates["oscillatory"],
        **{field: value},  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=match):
        DomainGuardHarmRiskCertificateV1(
            domain_guard_certificate=certificate.domain_guard_certificate,
            domain_certificates=certificates,
            family_confidence_level=0.95,
        )


def test_builder_rejects_malformed_or_incomplete_inputs() -> None:
    guard = _domain_guard()
    evidence = _evidence()
    base = {
        "domain_guard_certificate": guard,
        "threshold_source_artifact_id": THRESHOLD_SOURCE_ID,
        "certification_partition_id": CERTIFICATION_PARTITION_ID,
        "threshold_selection_group_ids": ("threshold-selection-1",),
        "threshold": 0.5,
        "harm_margin": 0.0,
        "target_harm_probability": 0.6,
        "family_confidence_level": 0.95,
        "minimum_accepted_group_count": 5,
        "threshold_frozen_before_certification_outcomes": True,
        "certification_outcomes_used_for_threshold_selection": False,
        "certification_groups_independent": True,
        **evidence,
    }

    with pytest.raises(ValueError, match="duplicates"):
        certify_domain_guard_harm_risk(
            **{**base, "group_ids": ("g",) * 10}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="equal lengths"):
        certify_domain_guard_harm_risk(
            **{**base, "risk_scores": np.zeros(9)}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="supported domains"):
        certify_domain_guard_harm_risk(
            **{
                **base,
                "domain_ids": ("dynamic",) * 10,
            }  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="boolean"):
        certify_domain_guard_harm_risk(
            **{
                **base,
                "fallback_identity_verified": np.ones(10),
            }  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="family_confidence_level"):
        certify_domain_guard_harm_risk(
            **{**base, "family_confidence_level": 1.0}  # type: ignore[arg-type]
        )


def test_builder_rejects_no_supported_domain_and_unverified_fallback() -> None:
    with pytest.raises(ValueError, match="authorizes no domain"):
        certify_domain_guard_harm_risk(
            domain_guard_certificate=_domain_guard(supported_domains=()),
            threshold_source_artifact_id=THRESHOLD_SOURCE_ID,
            certification_partition_id=CERTIFICATION_PARTITION_ID,
            threshold_selection_group_ids=("threshold-selection-1",),
            group_ids=("g",),
            domain_ids=("dynamic",),
            risk_scores=np.asarray([0.0]),
            candidate_losses=np.asarray([0.9]),
            fallback_losses=np.asarray([1.0]),
            fallback_identity_verified=np.asarray([True]),
            threshold=0.5,
            harm_margin=0.0,
            target_harm_probability=0.6,
            family_confidence_level=0.95,
            minimum_accepted_group_count=1,
            threshold_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_threshold_selection=False,
            certification_groups_independent=True,
        )

    evidence = _evidence()
    risk_scores = np.asarray(evidence["risk_scores"]).copy()
    fallback_verified = np.asarray(
        evidence["fallback_identity_verified"]
    ).copy()
    risk_scores[0] = 1.0
    fallback_verified[0] = False
    with pytest.raises(ValueError, match="exact fallback"):
        certify_domain_guard_harm_risk(
            domain_guard_certificate=_domain_guard(),
            threshold_source_artifact_id=THRESHOLD_SOURCE_ID,
            certification_partition_id=CERTIFICATION_PARTITION_ID,
            threshold_selection_group_ids=("threshold-selection-1",),
            group_ids=evidence["group_ids"],
            domain_ids=evidence["domain_ids"],
            risk_scores=risk_scores,
            candidate_losses=evidence["candidate_losses"],
            fallback_losses=evidence["fallback_losses"],
            fallback_identity_verified=fallback_verified,
            threshold=0.5,
            harm_margin=0.0,
            target_harm_probability=0.6,
            family_confidence_level=0.95,
            minimum_accepted_group_count=4,
            threshold_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_threshold_selection=False,
            certification_groups_independent=True,
        )


def test_metadata_and_artifact_identity_are_immutable() -> None:
    metadata = {"protocol": {"groups": ["independent"]}}
    certificate = _certificate(metadata=metadata)
    nested = metadata["protocol"]
    assert isinstance(nested, dict)
    groups = nested["groups"]
    assert isinstance(groups, list)
    groups.append("mutated")

    assert list(certificate.metadata["protocol"]["groups"]) == ["independent"]
    with pytest.raises(TypeError, match="immutable"):
        certificate.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(certificate, artifact_id="0" * 64)


def test_selection_copies_caller_metadata() -> None:
    baseline = _Belief("1" * 64)
    candidate = _Belief("2" * 64)
    metadata = {"request": {"source": "frozen-prefix"}}

    _, record = select_domain_guard_harm_risk_belief(
        baseline,
        candidate,
        _certificate(),
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
        metadata=metadata,
    )
    request = metadata["request"]
    assert isinstance(request, dict)
    request["source"] = "mutated"

    assert record.metadata["caller"]["request"]["source"] == "frozen-prefix"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain_id", " bad"),
        ("domain_guard_certificate_id", "not-a-digest"),
        ("family_confidence_level", True),
        ("family_confidence_level", 0.0),
        ("supported_domain_count", 0),
        ("target_harm_probability", np.nan),
        ("harm_margin", -0.1),
    ],
)
def test_policy_identity_rejects_invalid_contract_fields(
    field: str,
    value: object,
) -> None:
    guard = _domain_guard()
    decision = guard.decision_for_domain("dynamic")
    assert decision is not None
    arguments: dict[str, object] = {
        "domain_guard_certificate_id": str(guard.artifact_id),
        "domain_decision_id": str(decision.artifact_id),
        "domain_id": "dynamic",
        "threshold_source_artifact_id": THRESHOLD_SOURCE_ID,
        "statistical_unit": guard.statistical_unit,
        "metric": guard.metric,
        "threshold": 0.5,
        "harm_margin": 0.0,
        "target_harm_probability": 0.6,
        "family_confidence_level": 0.95,
        "supported_domain_count": 2,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        domain_harm_risk_policy_id(**arguments)  # type: ignore[arg-type]


def test_certificate_lookup_validates_domain_identifier() -> None:
    certificate = _certificate()

    assert certificate.certificate_for_domain("dynamic") is not None
    assert certificate.certificate_for_domain("unknown") is None
    with pytest.raises(ValueError, match="canonical string"):
        certificate.certificate_for_domain(" bad")
