from __future__ import annotations

import copy

import pytest

from bayesian_phystwin_experiments.discrepancy_scope_transport_v1 import (
    DIAGNOSIS_SCHEMA,
    DIAGNOSIS_SEMANTICS,
    DIAGNOSIS_VERSION,
    DiscrepancyScopeTransportV1,
    EvidenceDisposition,
    OperationalDisposition,
    PortableTargetDiagnosisV1,
    ScopeHypothesis,
    ScopeStatus,
    TransferAxis,
    TransferEvidenceV1,
    TransportTier,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
FALLBACK = "f" * 64


def diagnosis(
    disposition: str = "transport_without_cause",
) -> PortableTargetDiagnosisV1:
    transporting = disposition in {"transport_without_cause", "explain_and_transport"}
    none = disposition == "none_of_the_above"
    fallback = not transporting
    return PortableTargetDiagnosisV1(
        pipeline_artifact_id=DIGEST_A,
        target_id="held-query",
        disposition=disposition,
        adequacy_status=("unmodeled_cause" if none else "adequate_set_valued"),
        transport_permitted=transporting,
        fallback_required_now=fallback,
        none_of_the_above=none,
    )


def evidence(
    axis: TransferAxis,
    tier: TransportTier,
    supported: bool,
    *,
    improvement: float = 0.05,
    wins: int = 8,
    total: int = 8,
) -> TransferEvidenceV1:
    return TransferEvidenceV1(
        axis=axis,
        tier=tier,
        disposition=(
            EvidenceDisposition.SUPPORTED if supported else EvidenceDisposition.REJECTED
        ),
        evidence_id=DIGEST_B,
        relative_improvement=improvement if supported else -abs(improvement),
        wins=wins if supported else 0,
        total=total,
        frozen_before_outcome=True,
        target_selection_free=True,
    )


def exact_pattern(
    *,
    same_backend: bool,
    same_object: bool,
    double_shift: bool,
) -> list[TransferEvidenceV1]:
    return [
        evidence(
            TransferAxis.SAME_OBJECT_NEW_BACKEND,
            TransportTier.EXACT_COEFFICIENTS,
            same_object,
        ),
        evidence(
            TransferAxis.NEW_OBJECT_SAME_BACKEND,
            TransportTier.EXACT_COEFFICIENTS,
            same_backend,
        ),
        evidence(
            TransferAxis.NEW_OBJECT_NEW_BACKEND,
            TransportTier.EXACT_COEFFICIENTS,
            double_shift,
        ),
    ]


@pytest.mark.parametrize(
    ("same_object", "same_backend", "double_shift", "expected"),
    [
        (True, True, True, ScopeHypothesis.SHARED_PHYSICS),
        (
            True,
            False,
            False,
            ScopeHypothesis.OBJECT_SPECIFIC_BACKEND_STABLE,
        ),
        (
            False,
            True,
            False,
            ScopeHypothesis.BACKEND_SPECIFIC_OBJECT_STABLE,
        ),
        (False, False, False, ScopeHypothesis.OBJECT_BACKEND_LOCAL),
    ],
)
def test_exact_transfer_signatures_identify_registered_scope(
    same_object: bool,
    same_backend: bool,
    double_shift: bool,
    expected: ScopeHypothesis,
) -> None:
    records = exact_pattern(
        same_object=same_object,
        same_backend=same_backend,
        double_shift=double_shift,
    )
    records.append(
        evidence(
            TransferAxis.NEW_OBJECT_NEW_BACKEND,
            TransportTier.PROCEDURE_ONLY,
            True,
        )
    )
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=records,
        diagnosis=diagnosis(),
    )
    assert result.scope_status is ScopeStatus.UNIQUE
    assert result.compatible_scopes == (expected,)


def test_shared_scope_transports_exact_coefficients_on_double_shift() -> None:
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=exact_pattern(
            same_object=True,
            same_backend=True,
            double_shift=True,
        ),
        diagnosis=diagnosis(),
    )
    assert (
        result.operational_disposition
        is OperationalDisposition.TRANSPORT_EXACT_COEFFICIENTS
    )
    assert result.fallback_required_now is False


def test_object_specific_scope_uses_exact_coefficients_only_across_backend() -> None:
    records = exact_pattern(
        same_object=True,
        same_backend=False,
        double_shift=False,
    )
    records.append(
        evidence(
            TransferAxis.NEW_OBJECT_SAME_BACKEND,
            TransportTier.PROCEDURE_ONLY,
            True,
        )
    )
    backend_shift = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=records,
        diagnosis=diagnosis(),
    )
    object_shift = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_SAME_BACKEND,
        fallback_id=FALLBACK,
        evidence=records,
        diagnosis=diagnosis(),
    )
    assert (
        backend_shift.operational_disposition
        is OperationalDisposition.TRANSPORT_EXACT_COEFFICIENTS
    )
    assert object_shift.operational_disposition is (
        OperationalDisposition.PROCEDURE_ONLY_REFIT_REQUIRED
    )
    assert object_shift.fallback_required_now is True


def test_lower_direct_tier_is_used_after_exact_coefficients_are_rejected() -> None:
    records = exact_pattern(
        same_object=True,
        same_backend=False,
        double_shift=False,
    )
    records.append(
        evidence(
            TransferAxis.NEW_OBJECT_SAME_BACKEND,
            TransportTier.QUERY_EFFECT,
            True,
        )
    )
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_SAME_BACKEND,
        fallback_id=FALLBACK,
        evidence=records,
        diagnosis=diagnosis(),
    )
    assert result.strongest_directly_supported_tier is TransportTier.QUERY_EFFECT
    assert (
        result.operational_disposition is OperationalDisposition.TRANSPORT_QUERY_EFFECT
    )
    assert result.fallback_required_now is False


def test_unmodeled_cause_overrides_positive_transfer_evidence() -> None:
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=exact_pattern(
            same_object=True,
            same_backend=False,
            double_shift=False,
        ),
        diagnosis=diagnosis("none_of_the_above"),
    )
    assert result.operational_disposition is OperationalDisposition.NONE_OF_THE_ABOVE
    assert result.fallback_required_now is True


def test_probe_recommendation_never_transports_before_reassessment() -> None:
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=exact_pattern(
            same_object=True,
            same_backend=False,
            double_shift=False,
        ),
        diagnosis=diagnosis("probe_then_reassess"),
    )
    assert result.operational_disposition is OperationalDisposition.PROBE_THEN_REASSESS
    assert result.fallback_required_now is True


def test_scope_evidence_without_diagnosis_is_descriptive_only() -> None:
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=exact_pattern(
            same_object=True,
            same_backend=False,
            double_shift=False,
        ),
        diagnosis=None,
    )
    assert result.scope_status is ScopeStatus.UNIQUE
    assert result.compatible_scopes == (ScopeHypothesis.OBJECT_SPECIFIC_BACKEND_STABLE,)
    assert result.strongest_directly_supported_tier is (
        TransportTier.EXACT_COEFFICIENTS
    )
    assert result.operational_disposition is OperationalDisposition.EVIDENCE_ONLY
    assert result.fallback_required_now is True


def test_incompatible_transfer_pattern_is_none_of_the_above() -> None:
    # A double-shift success together with failures on both component axes
    # cannot be represented by any registered invariance scope.
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=exact_pattern(
            same_object=False,
            same_backend=False,
            double_shift=True,
        ),
        diagnosis=diagnosis(),
    )
    assert result.scope_status is ScopeStatus.NONE_OF_THE_ABOVE
    assert result.compatible_scopes == ()
    assert result.operational_disposition is OperationalDisposition.NONE_OF_THE_ABOVE
    assert result.fallback_required_now is True


def test_missing_cross_axis_evidence_is_not_promoted() -> None:
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=[
            evidence(
                TransferAxis.SAME_OBJECT_SAME_BACKEND,
                TransportTier.EXACT_COEFFICIENTS,
                True,
            )
        ],
        diagnosis=diagnosis(),
    )
    assert result.scope_status is ScopeStatus.INSUFFICIENT_EVIDENCE
    assert result.strongest_directly_supported_tier is None
    assert result.operational_disposition is OperationalDisposition.EXACT_FALLBACK
    assert result.fallback_required_now is True


def test_nonprospective_evidence_cannot_authorize_transport() -> None:
    record = TransferEvidenceV1(
        axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        tier=TransportTier.EXACT_COEFFICIENTS,
        disposition=EvidenceDisposition.SUPPORTED,
        evidence_id=DIGEST_B,
        relative_improvement=0.1,
        wins=8,
        total=8,
        frozen_before_outcome=False,
        target_selection_free=True,
    )
    result = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=[record],
        diagnosis=diagnosis(),
    )
    assert result.strongest_directly_supported_tier is None
    assert result.operational_disposition is OperationalDisposition.EXACT_FALLBACK


def test_portable_pipeline_record_is_validated() -> None:
    pipeline = {
        "schema": DIAGNOSIS_SCHEMA,
        "schema_version": DIAGNOSIS_VERSION,
        "semantics": DIAGNOSIS_SEMANTICS,
        "artifact_id": DIGEST_A,
        "target_decisions": [
            {
                "target_id": "held-query",
                "disposition": "transport_without_cause",
                "adequacy_status": "adequate_set_valued",
                "transport_permitted": True,
                "fallback_required_now": False,
                "none_of_the_above": False,
            }
        ],
    }
    result = PortableTargetDiagnosisV1.from_pipeline_record(
        pipeline,
        target_id="held-query",
        source_record_id=DIGEST_C,
    )
    assert result.pipeline_artifact_id == DIGEST_A
    assert result.source_record_id == DIGEST_C

    broken = copy.deepcopy(pipeline)
    broken["semantics"] = "wrong"
    with pytest.raises(ValueError, match="semantics"):
        PortableTargetDiagnosisV1.from_pipeline_record(
            broken,
            target_id="held-query",
        )


def test_content_addressed_round_trip_and_tamper_rejection() -> None:
    original = DiscrepancyScopeTransportV1(
        requested_axis=TransferAxis.SAME_OBJECT_NEW_BACKEND,
        fallback_id=FALLBACK,
        evidence=exact_pattern(
            same_object=True,
            same_backend=False,
            double_shift=False,
        ),
        diagnosis=diagnosis(),
        metadata={"study": "controlled"},
    )
    record = original.to_record()
    restored = DiscrepancyScopeTransportV1.from_record(record)
    assert restored.to_record() == record

    tampered = copy.deepcopy(record)
    tampered["reason"] = "invented"
    with pytest.raises(ValueError, match="inconsistent"):
        DiscrepancyScopeTransportV1.from_record(tampered)


def test_unavailable_evidence_cannot_embed_outcomes() -> None:
    with pytest.raises(ValueError, match="numerical outcomes"):
        TransferEvidenceV1(
            axis=TransferAxis.NEW_OBJECT_NEW_BACKEND,
            tier=TransportTier.EXACT_COEFFICIENTS,
            disposition=EvidenceDisposition.UNAVAILABLE,
            evidence_id=None,
            relative_improvement=0.1,
        )
