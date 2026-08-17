from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.guard_harm_risk import certify_guard_harm_risk
from bayesian_phystwin.guard_harm_risk_artifacts import (
    GuardFallbackArtifactBindingV1,
    GuardHarmRiskArtifactCertificateV1,
    certify_guard_harm_risk_from_artifacts,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64


def _ids(start: int, count: int) -> tuple[str, ...]:
    return tuple(f"{start + index:064x}" for index in range(count))


def _certificate() -> GuardHarmRiskArtifactCertificateV1:
    count = 30
    return certify_guard_harm_risk_from_artifacts(
        guard_policy_id=A,
        threshold_source_artifact_id=B,
        certification_partition_id=C,
        statistical_unit="independent-physical-object-v1",
        metric="endpoint-rmse-m",
        threshold_selection_group_ids=("source-object",),
        group_ids=tuple(f"object-{index:02d}" for index in range(count)),
        risk_scores=np.linspace(0.0, 0.9, count),
        candidate_losses=np.ones(count),
        fallback_losses=np.ones(count),
        selected_artifact_ids=_ids(1000, count),
        fallback_artifact_ids=_ids(0, count),
        threshold=1.0,
        harm_margin=0.0,
        target_harm_probability=0.10,
        confidence_level=0.95,
        minimum_accepted_group_count=29,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )


def test_claim_bearing_certificate_derives_identity_mask_from_ids() -> None:
    certificate = _certificate()

    assert certificate.certified
    assert not np.any(certificate.fallback_binding.exact_fallback_mask)
    np.testing.assert_array_equal(
        certificate.risk_certificate.fallback_identity_verified,
        certificate.fallback_binding.exact_fallback_mask,
    )
    assert not certificate.fallback_binding.exact_fallback_mask.flags.writeable


def test_rejected_group_requires_selected_artifact_to_equal_fallback() -> None:
    with pytest.raises(ValueError, match="exact fallback"):
        certify_guard_harm_risk_from_artifacts(
            guard_policy_id=A,
            threshold_source_artifact_id=B,
            certification_partition_id=C,
            statistical_unit="object",
            metric="loss",
            threshold_selection_group_ids=(),
            group_ids=("accepted", "rejected"),
            risk_scores=np.asarray([0.0, 1.0]),
            candidate_losses=np.ones(2),
            fallback_losses=np.ones(2),
            selected_artifact_ids=("1" * 64, "2" * 64),
            fallback_artifact_ids=("3" * 64, "4" * 64),
            threshold=0.5,
            harm_margin=0.0,
            target_harm_probability=0.50,
            confidence_level=0.80,
            minimum_accepted_group_count=1,
            threshold_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_threshold_selection=False,
            certification_groups_independent=True,
        )


def test_rejected_exact_fallback_is_recomputed_without_boolean_input() -> None:
    certificate = certify_guard_harm_risk_from_artifacts(
        guard_policy_id=A,
        threshold_source_artifact_id=B,
        certification_partition_id=C,
        statistical_unit="object",
        metric="loss",
        threshold_selection_group_ids=(),
        group_ids=("accepted", "rejected"),
        risk_scores=np.asarray([0.0, 1.0]),
        candidate_losses=np.ones(2),
        fallback_losses=np.ones(2),
        selected_artifact_ids=("1" * 64, "4" * 64),
        fallback_artifact_ids=("3" * 64, "4" * 64),
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.90,
        confidence_level=0.50,
        minimum_accepted_group_count=1,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )

    assert certificate.fallback_binding.exact_fallback_mask.tolist() == [
        False,
        True,
    ]
    assert certificate.risk_certificate.accepted_mask.tolist() == [True, False]


def test_compound_contract_rejects_forged_boolean_evidence() -> None:
    binding = GuardFallbackArtifactBindingV1(
        group_ids=("a", "b"),
        selected_artifact_ids=("1" * 64, "2" * 64),
        fallback_artifact_ids=("3" * 64, "2" * 64),
    )
    forged = certify_guard_harm_risk(
        guard_policy_id=A,
        threshold_source_artifact_id=B,
        certification_partition_id=C,
        statistical_unit="object",
        metric="loss",
        threshold_selection_group_ids=(),
        group_ids=("a", "b"),
        risk_scores=np.asarray([0.0, 1.0]),
        candidate_losses=np.ones(2),
        fallback_losses=np.ones(2),
        fallback_identity_verified=np.asarray([True, True]),
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.90,
        confidence_level=0.50,
        minimum_accepted_group_count=1,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )

    with pytest.raises(ValueError, match="differs from artifact IDs"):
        GuardHarmRiskArtifactCertificateV1(
            fallback_binding=binding,
            risk_certificate=forged,
        )


def test_binding_identity_is_group_order_invariant() -> None:
    first = GuardFallbackArtifactBindingV1(
        group_ids=("b", "a"),
        selected_artifact_ids=("2" * 64, "1" * 64),
        fallback_artifact_ids=("4" * 64, "3" * 64),
    )
    second = GuardFallbackArtifactBindingV1(
        group_ids=("a", "b"),
        selected_artifact_ids=("1" * 64, "2" * 64),
        fallback_artifact_ids=("3" * 64, "4" * 64),
    )

    assert first.artifact_id == second.artifact_id
    assert first.group_ids == second.group_ids


def test_compound_artifact_id_rejects_tampering() -> None:
    certificate = _certificate()

    with pytest.raises(ValueError, match="artifact_id"):
        replace(certificate, artifact_id="0" * 64)
