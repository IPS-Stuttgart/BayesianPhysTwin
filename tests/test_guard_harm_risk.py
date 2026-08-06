from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.guard_harm_risk import (
    GuardHarmRiskCertificateV1,
    certify_guard_harm_risk,
    load_guard_harm_risk_certificate,
    minimum_zero_harm_groups_for_certificate,
    one_sided_binomial_upper_bound,
    save_guard_harm_risk_certificate,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64


def _certificate(
    count: int = 30,
    *,
    threshold: float = 1.0,
    minimum_accepted_group_count: int = 10,
) -> GuardHarmRiskCertificateV1:
    return certify_guard_harm_risk(
        guard_policy_id=A,
        threshold_source_artifact_id=B,
        certification_partition_id=C,
        statistical_unit="independent-physical-object-v1",
        metric="endpoint-rmse-m",
        threshold_selection_group_ids=("source-object-00",),
        group_ids=tuple(f"object-{index:02d}" for index in range(count)),
        risk_scores=np.linspace(0.0, 0.9, count),
        candidate_losses=np.ones(count),
        fallback_losses=np.ones(count),
        fallback_identity_verified=np.ones(count, dtype=bool),
        threshold=threshold,
        harm_margin=0.0,
        target_harm_probability=0.10,
        confidence_level=0.95,
        minimum_accepted_group_count=minimum_accepted_group_count,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
        metadata={"partition": "certification-only"},
    )


def test_zero_harm_support_planning_is_exact() -> None:
    assert minimum_zero_harm_groups_for_certificate(0.10, 0.95) == 29
    assert minimum_zero_harm_groups_for_certificate(0.20, 0.95) == 14
    assert one_sided_binomial_upper_bound(0, 10, 0.95) == pytest.approx(
        1.0 - 0.05 ** (1.0 / 10.0)
    )
    assert one_sided_binomial_upper_bound(10, 10, 0.95) == 1.0


def test_general_clopper_pearson_bound_inverts_binomial_cdf() -> None:
    upper = one_sided_binomial_upper_bound(1, 10, 0.95)
    probability = sum(
        float(math.comb(10, index))
        * upper**index
        * (1.0 - upper) ** (10 - index)
        for index in range(2)
    )

    assert probability == pytest.approx(0.05, abs=2e-14)


def test_certificate_passes_only_with_sufficient_independent_support() -> None:
    certificate = _certificate()

    assert certificate.certified
    assert certificate.accepted_count == 30
    assert certificate.harmful_accepted_count == 0
    assert certificate.one_sided_upper_bound < 0.10
    assert certificate.minimum_zero_harm_accepted_groups == 29


def test_insufficient_support_is_a_valid_negative_certificate() -> None:
    certificate = _certificate(
        count=10,
        minimum_accepted_group_count=29,
    )

    assert not certificate.certified
    assert certificate.accepted_count == 10
    assert certificate.harmful_accepted_count == 0
    assert certificate.one_sided_upper_bound > 0.10
    assert certificate.minimum_accepted_group_count > certificate.group_count


def test_group_order_does_not_change_the_certificate_identity() -> None:
    certificate = _certificate()
    permutation = np.arange(certificate.group_count)[::-1]
    permuted = certify_guard_harm_risk(
        guard_policy_id=certificate.guard_policy_id,
        threshold_source_artifact_id=(
            certificate.threshold_source_artifact_id
        ),
        certification_partition_id=certificate.certification_partition_id,
        statistical_unit=certificate.statistical_unit,
        metric=certificate.metric,
        threshold_selection_group_ids=(
            certificate.threshold_selection_group_ids
        ),
        group_ids=tuple(certificate.group_ids[index] for index in permutation),
        risk_scores=certificate.risk_scores[permutation],
        candidate_losses=certificate.candidate_losses[permutation],
        fallback_losses=certificate.fallback_losses[permutation],
        fallback_identity_verified=(
            certificate.fallback_identity_verified[permutation]
        ),
        threshold=certificate.threshold,
        harm_margin=certificate.harm_margin,
        target_harm_probability=certificate.target_harm_probability,
        confidence_level=certificate.confidence_level,
        minimum_accepted_group_count=(
            certificate.minimum_accepted_group_count
        ),
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
        metadata=certificate.metadata,
    )

    assert permuted.artifact_id == certificate.artifact_id
    assert permuted.group_ids == certificate.group_ids


def test_certificate_round_trip_and_arrays_are_immutable(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "certificate.json"

    save_guard_harm_risk_certificate(certificate, path)
    restored = load_guard_harm_risk_certificate(path)

    assert restored.artifact_id == certificate.artifact_id
    assert not restored.risk_scores.flags.writeable
    assert not restored.accepted_mask.flags.writeable
    with pytest.raises(ValueError):
        restored.risk_scores[0] = 100.0
    with pytest.raises(FileExistsError):
        save_guard_harm_risk_certificate(certificate, path)


def test_rejected_groups_must_verify_exact_fallback() -> None:
    verified = np.ones(3, dtype=bool)
    verified[-1] = False

    with pytest.raises(ValueError, match="exact fallback"):
        certify_guard_harm_risk(
            guard_policy_id=A,
            threshold_source_artifact_id=B,
            certification_partition_id=C,
            statistical_unit="object",
            metric="loss",
            threshold_selection_group_ids=(),
            group_ids=("a", "b", "c"),
            risk_scores=np.asarray([0.0, 0.5, 1.0]),
            candidate_losses=np.ones(3),
            fallback_losses=np.ones(3),
            fallback_identity_verified=verified,
            threshold=0.5,
            harm_margin=0.0,
            target_harm_probability=0.20,
            confidence_level=0.95,
            minimum_accepted_group_count=1,
            threshold_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_threshold_selection=False,
            certification_groups_independent=True,
        )


def test_information_order_and_independence_fail_closed() -> None:
    kwargs = {
        "guard_policy_id": A,
        "threshold_source_artifact_id": B,
        "certification_partition_id": C,
        "statistical_unit": "object",
        "metric": "loss",
        "threshold_selection_group_ids": (),
        "group_ids": ("a",),
        "risk_scores": np.asarray([0.0]),
        "candidate_losses": np.asarray([1.0]),
        "fallback_losses": np.asarray([1.0]),
        "fallback_identity_verified": np.asarray([True]),
        "threshold": 0.0,
        "harm_margin": 0.0,
        "target_harm_probability": 0.20,
        "confidence_level": 0.95,
        "minimum_accepted_group_count": 1,
    }
    with pytest.raises(ValueError, match="frozen"):
        certify_guard_harm_risk(
            **kwargs,
            threshold_frozen_before_certification_outcomes=False,
            certification_outcomes_used_for_threshold_selection=False,
            certification_groups_independent=True,
        )
    with pytest.raises(ValueError, match="cannot select"):
        certify_guard_harm_risk(
            **kwargs,
            threshold_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_threshold_selection=True,
            certification_groups_independent=True,
        )
    with pytest.raises(ValueError, match="independent physical units"):
        certify_guard_harm_risk(
            **kwargs,
            threshold_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_threshold_selection=False,
            certification_groups_independent=False,
        )


def test_threshold_selection_and_certification_groups_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="groups overlap"):
        certify_guard_harm_risk(
            guard_policy_id=A,
            threshold_source_artifact_id=B,
            certification_partition_id=C,
            statistical_unit="object",
            metric="loss",
            threshold_selection_group_ids=("shared",),
            group_ids=("shared",),
            risk_scores=np.asarray([0.0]),
            candidate_losses=np.asarray([1.0]),
            fallback_losses=np.asarray([1.0]),
            fallback_identity_verified=np.asarray([True]),
            threshold=0.0,
            harm_margin=0.0,
            target_harm_probability=0.20,
            confidence_level=0.95,
            minimum_accepted_group_count=1,
            threshold_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_threshold_selection=False,
            certification_groups_independent=True,
        )


def test_direct_construction_rejects_derived_field_tampering() -> None:
    certificate = _certificate()

    with pytest.raises(ValueError, match="accepted_count"):
        replace(
            certificate,
            accepted_count=certificate.accepted_count - 1,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="upper_bound"):
        replace(
            certificate,
            one_sided_upper_bound=0.0,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="artifact_id"):
        replace(certificate, artifact_id="0" * 64)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "duplicate.json"
    record = json.dumps(certificate.to_record(), sort_keys=True)
    path.write_text(
        "{\"schema\":\"duplicate\"," + record[1:],
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_guard_harm_risk_certificate(path)
