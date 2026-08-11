from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.covariance_only_value import (
    CovarianceOnlyValueCertificateV1,
    bonferroni_gate_confidence_level,
    certify_covariance_only_value,
    hoeffding_mean_upper_bound,
    load_covariance_only_value_certificate,
    save_covariance_only_value_certificate,
)


def _digest(index: int, *, offset: int = 0) -> str:
    return f"{index + offset + 1:064x}"


def _certificate(
    *,
    count: int = 60,
    mismatch_index: int | None = None,
    candidate_width: float = 0.020,
    maximum_width: float = 0.026,
    score_difference: float = -1.0,
    target_harm_probability: float = 0.07,
) -> CovarianceOnlyValueCertificateV1:
    group_ids = [f"object-{index:03d}" for index in range(count)]
    reference_digests = [_digest(index) for index in range(count)]
    candidate_digests = list(reference_digests)
    if mismatch_index is not None:
        candidate_digests[mismatch_index] = _digest(mismatch_index, offset=count)
    reference_scores = np.zeros(count, dtype=np.float64)
    candidate_scores = np.full(count, score_difference, dtype=np.float64)
    return certify_covariance_only_value(
        candidate_policy_id="1" * 64,
        reference_policy_id="2" * 64,
        query_set_id="3" * 64,
        policy_freeze_artifact_id="4" * 64,
        certification_partition_id="5" * 64,
        statistical_unit="independent-physical-object-v1",
        score_metric="group-gaussian-nll-v1",
        width_metric="group-mean-full-width-m-v1",
        selection_group_ids=("source-a", "source-b"),
        group_ids=group_ids,
        candidate_mean_sha256=candidate_digests,
        reference_mean_sha256=reference_digests,
        candidate_scores=candidate_scores,
        reference_scores=reference_scores,
        candidate_full_widths=np.full(count, candidate_width),
        reference_full_widths=np.full(count, 0.010),
        score_difference_lower_bound=-1.1,
        score_difference_upper_bound=1.1,
        full_width_upper_bound=0.030,
        maximum_expected_score_regret=0.0,
        maximum_expected_full_width=maximum_width,
        harm_margin=0.0,
        target_harm_probability=target_harm_probability,
        familywise_confidence_level=0.95,
        minimum_group_count=50,
        thresholds_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_policy_selection=False,
        certification_groups_independent=True,
        metadata={"protocol": "fresh-covariance-only-v1"},
    )


def test_exact_mean_certificate_passes_all_three_gates() -> None:
    certificate = _certificate()

    assert certificate.certified is True
    assert certificate.mean_identity_count == certificate.group_count == 60
    assert certificate.score_upper_confidence_bound < 0.0
    assert certificate.width_upper_confidence_bound < 0.026
    assert certificate.harmful_group_count == 0
    assert certificate.harm_probability_upper_bound < 0.07
    assert certificate.mean_full_width_difference == pytest.approx(0.010)


def test_mean_digest_mismatch_is_retained_as_negative_evidence() -> None:
    certificate = _certificate(mismatch_index=13)

    assert certificate.certified is False
    assert certificate.mean_identity_count == 59
    assert certificate.mean_identity_mask[13] == np.bool_(False)


def test_width_budget_failure_is_retained_as_negative_evidence() -> None:
    certificate = _certificate(candidate_width=0.025, maximum_width=0.026)

    assert certificate.certified is False
    assert certificate.width_upper_confidence_bound > 0.026
    assert certificate.score_upper_confidence_bound < 0.0


def test_harm_probability_failure_is_retained_as_negative_evidence() -> None:
    certificate = _certificate(target_harm_probability=0.05)

    assert certificate.certified is False
    assert certificate.harm_probability_upper_bound > 0.05


def test_group_order_does_not_change_artifact_identity() -> None:
    first = _certificate()
    order = np.arange(first.group_count)[::-1]
    second = certify_covariance_only_value(
        candidate_policy_id=first.candidate_policy_id,
        reference_policy_id=first.reference_policy_id,
        query_set_id=first.query_set_id,
        policy_freeze_artifact_id=first.policy_freeze_artifact_id,
        certification_partition_id=first.certification_partition_id,
        statistical_unit=first.statistical_unit,
        score_metric=first.score_metric,
        width_metric=first.width_metric,
        selection_group_ids=first.selection_group_ids,
        group_ids=[first.group_ids[index] for index in order],
        candidate_mean_sha256=[first.candidate_mean_sha256[index] for index in order],
        reference_mean_sha256=[first.reference_mean_sha256[index] for index in order],
        candidate_scores=first.candidate_scores[order],
        reference_scores=first.reference_scores[order],
        candidate_full_widths=first.candidate_full_widths[order],
        reference_full_widths=first.reference_full_widths[order],
        score_difference_lower_bound=first.score_difference_lower_bound,
        score_difference_upper_bound=first.score_difference_upper_bound,
        full_width_upper_bound=first.full_width_upper_bound,
        maximum_expected_score_regret=first.maximum_expected_score_regret,
        maximum_expected_full_width=first.maximum_expected_full_width,
        harm_margin=first.harm_margin,
        target_harm_probability=first.target_harm_probability,
        familywise_confidence_level=first.familywise_confidence_level,
        minimum_group_count=first.minimum_group_count,
        thresholds_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_policy_selection=False,
        certification_groups_independent=True,
        metadata=first.metadata,
    )

    assert second.artifact_id == first.artifact_id
    assert second.to_record() == first.to_record()


def test_save_load_and_derived_field_tamper_detection(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "certificate.json"
    save_covariance_only_value_certificate(certificate, path)

    restored = load_covariance_only_value_certificate(path)
    assert restored.artifact_id == certificate.artifact_id
    assert restored.to_record() == certificate.to_record()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["score_upper_confidence_bound"] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_covariance_only_value_certificate(path)


def test_certificate_arrays_are_bytes_backed_and_immutable() -> None:
    certificate = _certificate()

    for array in (
        certificate.candidate_scores,
        certificate.reference_scores,
        certificate.candidate_full_widths,
        certificate.reference_full_widths,
        certificate.mean_identity_mask,
        certificate.score_differences,
        certificate.harmful_mask,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_selection_and_certification_groups_must_be_disjoint() -> None:
    certificate = _certificate()
    with pytest.raises(ValueError, match="selection and certification groups overlap"):
        certify_covariance_only_value(
            candidate_policy_id=certificate.candidate_policy_id,
            reference_policy_id=certificate.reference_policy_id,
            query_set_id=certificate.query_set_id,
            policy_freeze_artifact_id=certificate.policy_freeze_artifact_id,
            certification_partition_id=certificate.certification_partition_id,
            statistical_unit=certificate.statistical_unit,
            score_metric=certificate.score_metric,
            width_metric=certificate.width_metric,
            selection_group_ids=(certificate.group_ids[0],),
            group_ids=certificate.group_ids,
            candidate_mean_sha256=certificate.candidate_mean_sha256,
            reference_mean_sha256=certificate.reference_mean_sha256,
            candidate_scores=certificate.candidate_scores,
            reference_scores=certificate.reference_scores,
            candidate_full_widths=certificate.candidate_full_widths,
            reference_full_widths=certificate.reference_full_widths,
            score_difference_lower_bound=certificate.score_difference_lower_bound,
            score_difference_upper_bound=certificate.score_difference_upper_bound,
            full_width_upper_bound=certificate.full_width_upper_bound,
            maximum_expected_score_regret=certificate.maximum_expected_score_regret,
            maximum_expected_full_width=certificate.maximum_expected_full_width,
            harm_margin=certificate.harm_margin,
            target_harm_probability=certificate.target_harm_probability,
            familywise_confidence_level=certificate.familywise_confidence_level,
            minimum_group_count=certificate.minimum_group_count,
            thresholds_frozen_before_certification_outcomes=True,
            certification_outcomes_used_for_policy_selection=False,
            certification_groups_independent=True,
        )


@pytest.mark.parametrize(
    "invalid_values",
    (
        np.asarray([1.0 + 1.0j]),
        np.asarray([True]),
        np.asarray(["1.0"], dtype=object),
    ),
)
def test_hoeffding_bound_rejects_lossy_numeric_coercions(
    invalid_values: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="values must be a real vector"):
        hoeffding_mean_upper_bound(
            invalid_values,
            lower_bound=-1.0,
            upper_bound=1.0,
            confidence_level=0.95,
        )


def test_hoeffding_bound_rejects_values_outside_registered_range() -> None:
    with pytest.raises(ValueError, match="frozen bounded interval"):
        hoeffding_mean_upper_bound(
            np.asarray([0.0, 1.1]),
            lower_bound=-1.0,
            upper_bound=1.0,
            confidence_level=0.95,
        )


def test_bonferroni_level_uses_three_one_sided_gates() -> None:
    assert bonferroni_gate_confidence_level(0.95) == pytest.approx(1.0 - 0.05 / 3.0)


def _replace_certificate(**changes: object) -> CovarianceOnlyValueCertificateV1:
    return replace(_certificate(), artifact_id=None, **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"statistical_unit": ""}, "nonempty canonical string"),
        ({"full_width_upper_bound": True}, "finite real number"),
        ({"full_width_upper_bound": np.inf}, "finite real number"),
        ({"harm_margin": -1.0}, "must be at least"),
        ({"target_harm_probability": 1.0}, "strictly inside"),
        ({"candidate_scores": np.full(60, np.nan)}, "finite real vector"),
        ({"group_ids": "not-a-sequence"}, "group_ids must be a sequence"),
        ({"group_ids": ("only-one",)}, "group_ids length"),
        ({"group_ids": ("same",) * 60}, "group_ids must not contain duplicates"),
        ({"selection_group_ids": "not-a-sequence"}, "selection_group_ids must be"),
        ({"selection_group_ids": ("same", "same")}, "must not contain duplicates"),
        ({"candidate_mean_sha256": "not-a-sequence"}, "must be a sequence"),
        ({"candidate_mean_sha256": ("1" * 64,)}, "length must match"),
        (
            {
                "score_difference_lower_bound": 1.0,
                "score_difference_upper_bound": 1.0,
            },
            "lower bound must be smaller",
        ),
        ({"full_width_upper_bound": 0.0}, "must be positive"),
        ({"maximum_expected_score_regret": 2.0}, "must lie inside score bounds"),
        ({"maximum_expected_full_width": 0.031}, "cannot exceed"),
        (
            {"thresholds_frozen_before_certification_outcomes": False},
            "thresholds must be frozen",
        ),
        (
            {"certification_outcomes_used_for_policy_selection": True},
            "cannot select",
        ),
        ({"certification_groups_independent": False}, "must be independent"),
        ({"candidate_scores": np.full(60, 1.2)}, "score differences must lie"),
        ({"candidate_full_widths": np.full(60, 0.031)}, "widths must lie"),
        ({"candidate_full_widths": np.full(60, -0.001)}, "must be nonnegative"),
        ({"reference_scores": np.zeros(59)}, "arrays must have equal length"),
    ),
)
def test_certificate_rejects_structural_and_information_order_violations(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _replace_certificate(**changes)


def test_certificate_rejects_an_empty_certification_partition() -> None:
    with pytest.raises(ValueError, match="at least one independent"):
        _replace_certificate(candidate_scores=np.asarray([], dtype=np.float64))


def test_hoeffding_rejects_empty_groups() -> None:
    with pytest.raises(ValueError, match="at least one independent group"):
        hoeffding_mean_upper_bound(
            np.asarray([], dtype=np.float64),
            lower_bound=-1.0,
            upper_bound=1.0,
            confidence_level=0.95,
        )


def test_artifact_identity_mismatch_is_rejected() -> None:
    certificate = _certificate()
    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(certificate, artifact_id="0" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema", "changed.schema", "schema changed"),
        ("schema_version", 2, "version changed"),
        (
            "score_difference_semantics",
            "changed-semantics",
            "score_difference_semantics changed",
        ),
    ),
)
def test_from_mapping_rejects_schema_and_semantics_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _certificate().to_record()
    record[field] = value
    with pytest.raises(ValueError, match=message):
        CovarianceOnlyValueCertificateV1.from_mapping(record)


def test_from_mapping_rejects_nonmapping_and_derived_type_drift() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        CovarianceOnlyValueCertificateV1.from_mapping([])

    record = _certificate().to_record()
    record["mean_identity_count"] = float(record["mean_identity_count"])
    with pytest.raises(ValueError, match="derived field changed"):
        CovarianceOnlyValueCertificateV1.from_mapping(record)


def test_save_rejects_the_wrong_contract_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="CovarianceOnlyValueCertificateV1"):
        save_covariance_only_value_certificate(  # type: ignore[arg-type]
            object(),
            tmp_path / "invalid.json",
        )
