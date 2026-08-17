from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.guard_harm_risk as risk
import bayesian_phystwin.guard_harm_risk_artifacts as artifacts

A = "a" * 64
B = "b" * 64
C = "c" * 64


def _certificate() -> risk.GuardHarmRiskCertificateV1:
    return risk.certify_guard_harm_risk(
        guard_policy_id=A,
        threshold_source_artifact_id=B,
        certification_partition_id=C,
        statistical_unit="object",
        metric="loss",
        threshold_selection_group_ids=("selection",),
        group_ids=("accepted", "rejected"),
        risk_scores=np.asarray([0.0, 1.0]),
        candidate_losses=np.asarray([1.0, 1.0]),
        fallback_losses=np.asarray([1.0, 1.0]),
        fallback_identity_verified=np.asarray([True, True]),
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.9,
        confidence_level=0.5,
        minimum_accepted_group_count=1,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )


def test_low_level_validators_reject_malformed_values() -> None:
    for value in ("", " x", 1):
        with pytest.raises(ValueError, match="canonical string"):
            risk._canonical_string(value, name="value")

    for value in (True, "1"):
        with pytest.raises(ValueError, match="finite real"):
            risk._finite_real(value, name="value")
    with pytest.raises(ValueError, match="finite real"):
        risk._finite_real(np.inf, name="value")
    with pytest.raises(ValueError, match="at least"):
        risk._finite_real(-1.0, name="value", minimum=0.0)

    for value in (0.0, 1.0):
        with pytest.raises(ValueError, match="strictly inside"):
            risk._open_probability(value, name="probability")

    with pytest.raises(ValueError, match="real vector"):
        risk._float_vector(object(), name="vector")
    for value in ([[1.0]], [np.nan]):
        with pytest.raises(ValueError, match="finite real vector"):
            risk._float_vector(value, name="vector")
    with pytest.raises(ValueError, match="boolean vector"):
        risk._boolean_vector([0, 1], name="mask")

    with pytest.raises(ValueError, match="sequence"):
        risk._group_id_tuple("group", expected_count=1)
    with pytest.raises(ValueError, match="length"):
        risk._group_id_tuple(("group",), expected_count=2)
    with pytest.raises(ValueError, match="duplicates"):
        risk._group_id_tuple(("group", "group"), expected_count=2)
    with pytest.raises(ValueError, match="sequence"):
        risk._selection_group_id_tuple("group")
    with pytest.raises(ValueError, match="duplicates"):
        risk._selection_group_id_tuple(("group", "group"))


def test_binomial_boundary_and_count_validation_branches() -> None:
    assert risk._binomial_cdf(0, 1, 0.0) == 1.0
    assert risk._binomial_cdf(1, 1, 1.0) == 1.0
    assert risk._binomial_cdf(0, 1, 1.0) == 0.0
    assert risk.one_sided_binomial_upper_bound(0, 0, 0.95) == 1.0
    with pytest.raises(ValueError, match="cannot exceed"):
        risk.one_sided_binomial_upper_bound(2, 1, 0.95)


def test_certificate_constructor_rejects_malformed_and_tampered_fields() -> None:
    certificate = _certificate()
    empty_float = np.asarray([], dtype=np.float64)
    empty_bool = np.asarray([], dtype=bool)

    with pytest.raises(ValueError, match="at least one"):
        replace(
            certificate,
            group_ids=(),
            risk_scores=empty_float,
            candidate_losses=empty_float,
            fallback_losses=empty_float,
            fallback_identity_verified=empty_bool,
            accepted_mask=empty_bool,
            harmful_mask=empty_bool,
            accepted_count=0,
            harmful_accepted_count=0,
            one_sided_upper_bound=1.0,
            certified=False,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="equal length"):
        replace(
            certificate,
            candidate_losses=certificate.candidate_losses[:-1],
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        replace(
            certificate,
            candidate_losses=-np.ones(certificate.group_count),
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="accepted_mask"):
        replace(
            certificate,
            accepted_mask=~certificate.accepted_mask,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="harmful_mask"):
        replace(
            certificate,
            harmful_mask=~certificate.harmful_mask,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="harmful_accepted_count"):
        replace(
            certificate,
            harmful_accepted_count=certificate.harmful_accepted_count + 1,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="upper_bound"):
        replace(certificate, one_sided_upper_bound=1.5, artifact_id=None)
    with pytest.raises(ValueError, match="certified"):
        replace(certificate, certified=not certificate.certified, artifact_id=None)


def test_mapping_loader_rejects_schema_and_derived_field_drift() -> None:
    certificate = _certificate()
    record = certificate.to_record()

    with pytest.raises(ValueError, match="mapping"):
        risk.GuardHarmRiskCertificateV1.from_mapping([])

    mutations = (
        ("schema", "wrong", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("bound_method", "wrong", "bound method changed"),
        ("risk_score_semantics", "wrong", "risk-score semantics changed"),
        ("group_count", certificate.group_count + 1, "group_count changed"),
        (
            "acceptance_rate",
            certificate.acceptance_rate + 0.1,
            "acceptance_rate changed",
        ),
        ("observed_harm_rate", None, "observed_harm_rate changed"),
        (
            "minimum_zero_harm_accepted_groups",
            certificate.minimum_zero_harm_accepted_groups + 1,
            "minimum zero-harm support changed",
        ),
    )
    for key, value, match in mutations:
        payload = dict(record)
        payload[key] = value
        with pytest.raises(ValueError, match=match):
            risk.GuardHarmRiskCertificateV1.from_mapping(payload)

    with pytest.raises(TypeError, match="certificate"):
        risk.save_guard_harm_risk_certificate(object(), "unused.json")


def test_artifact_binding_validators_and_identity_fail_closed() -> None:
    with pytest.raises(ValueError, match="canonical string"):
        artifacts.GuardFallbackArtifactBindingV1(
            group_ids=(" bad",),
            selected_artifact_ids=("1" * 64,),
            fallback_artifact_ids=("1" * 64,),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        artifacts.GuardFallbackArtifactBindingV1(
            group_ids=(),
            selected_artifact_ids=(),
            fallback_artifact_ids=(),
        )
    with pytest.raises(ValueError, match="duplicates"):
        artifacts.GuardFallbackArtifactBindingV1(
            group_ids=("a", "a"),
            selected_artifact_ids=("1" * 64, "2" * 64),
            fallback_artifact_ids=("1" * 64, "2" * 64),
        )
    with pytest.raises(ValueError, match="length"):
        artifacts.GuardFallbackArtifactBindingV1(
            group_ids=("a", "b"),
            selected_artifact_ids=("1" * 64,),
            fallback_artifact_ids=("1" * 64, "2" * 64),
        )

    binding = artifacts.GuardFallbackArtifactBindingV1(
        group_ids=("a",),
        selected_artifact_ids=("1" * 64,),
        fallback_artifact_ids=("1" * 64,),
    )
    with pytest.raises(ValueError, match="artifact_id"):
        replace(binding, artifact_id="0" * 64)


def test_compound_artifact_contract_rejects_type_group_and_rejection_drift() -> None:
    certificate = _certificate()
    binding = artifacts.GuardFallbackArtifactBindingV1(
        group_ids=certificate.group_ids,
        selected_artifact_ids=("1" * 64, "2" * 64),
        fallback_artifact_ids=("3" * 64, "2" * 64),
    )

    with pytest.raises(TypeError, match="fallback_binding"):
        artifacts.GuardHarmRiskArtifactCertificateV1(
            fallback_binding=object(),
            risk_certificate=certificate,
        )
    with pytest.raises(TypeError, match="risk_certificate"):
        artifacts.GuardHarmRiskArtifactCertificateV1(
            fallback_binding=binding,
            risk_certificate=object(),
        )

    different_groups = artifacts.GuardFallbackArtifactBindingV1(
        group_ids=("x", "y"),
        selected_artifact_ids=("1" * 64, "2" * 64),
        fallback_artifact_ids=("1" * 64, "2" * 64),
    )
    with pytest.raises(ValueError, match="groups differ"):
        artifacts.GuardHarmRiskArtifactCertificateV1(
            fallback_binding=different_groups,
            risk_certificate=certificate,
        )

    all_accepted = risk.certify_guard_harm_risk(
        guard_policy_id=A,
        threshold_source_artifact_id=B,
        certification_partition_id=C,
        statistical_unit="object",
        metric="loss",
        threshold_selection_group_ids=(),
        group_ids=certificate.group_ids,
        risk_scores=np.asarray([0.0, 0.0]),
        candidate_losses=np.ones(2),
        fallback_losses=np.ones(2),
        fallback_identity_verified=binding.exact_fallback_mask,
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.9,
        confidence_level=0.5,
        minimum_accepted_group_count=1,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )
    object.__setattr__(
        all_accepted,
        "accepted_mask",
        np.asarray([False, True], dtype=bool),
    )
    with pytest.raises(ValueError, match="every rejected group"):
        artifacts.GuardHarmRiskArtifactCertificateV1(
            fallback_binding=binding,
            risk_certificate=all_accepted,
        )


def test_artifact_builder_exercises_explicit_metadata_paths() -> None:
    certificate = artifacts.certify_guard_harm_risk_from_artifacts(
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
        selected_artifact_ids=("1" * 64, "2" * 64),
        fallback_artifact_ids=("3" * 64, "2" * 64),
        threshold=0.5,
        harm_margin=0.0,
        target_harm_probability=0.9,
        confidence_level=0.5,
        minimum_accepted_group_count=1,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
        binding_metadata={"source": "explicit"},
        certificate_metadata={"source": "explicit"},
    )

    assert certificate.fallback_binding.metadata["source"] == "explicit"
    assert certificate.risk_certificate.metadata["source"] == "explicit"
