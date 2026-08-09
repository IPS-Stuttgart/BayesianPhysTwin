import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_group_sandwich_covariance import (
    test_arrays_are_immutable,
    test_duplicate_rows_stay_inside_one_group,
    test_matches_manual_group_cr1_covariance,
    test_rejects_invalid_bread_and_scores,
    test_rejects_too_few_or_malformed_groups,
    test_result_rejects_covariance_not_generated_by_declared_scores,
    test_row_and_group_permutation_is_content_invariant,
    test_splitting_a_group_changes_the_claim_identity,
)
from test_posterior_uncertainty import (
    test_already_calibrated_source_semantics_are_rejected,
    test_artifact_identity_round_trips_and_detects_substitution,
    test_calibration_is_bound_to_the_predictor_and_query_set,
    test_calibration_with_another_predictor_is_rejected,
    test_calibration_with_another_query_set_is_rejected,
    test_invalid_query_covariance_fails_closed,
    test_maximum_ten_group_coverage_round_trips_as_finite,
    test_nonworking_covariance_requires_estimator_identity,
    test_semantics_and_covariance_dimension_must_agree,
    test_ten_groups_report_ninety_five_percent_as_unavailable,
    test_uncalibrated_artifact_keeps_raw_semantics_explicit,
    test_uncertainty_record_binds_artifact_identity,
    test_wrong_contract_types_are_rejected,
)
from test_uncertainty_public_surface import (
    test_uncertainty_namespace_is_narrow_and_explicit,
)

from bayesian_phystwin import BinaryCalibrationMetrics, binary_calibration_metrics
from bayesian_phystwin.calibration import (
    FiniteGroupCalibrationDesign,
    finite_group_conformal_rank,
    maximum_finite_group_coverage,
    minimum_groups_for_finite_conformal,
    plan_finite_group_calibration,
)

_GROUP_SANDWICH_COVARIANCE_STABLE_TESTS = (
    test_arrays_are_immutable,
    test_duplicate_rows_stay_inside_one_group,
    test_matches_manual_group_cr1_covariance,
    test_rejects_invalid_bread_and_scores,
    test_rejects_too_few_or_malformed_groups,
    test_result_rejects_covariance_not_generated_by_declared_scores,
    test_row_and_group_permutation_is_content_invariant,
    test_splitting_a_group_changes_the_claim_identity,
)

_POSTERIOR_UNCERTAINTY_STABLE_TESTS = (
    test_already_calibrated_source_semantics_are_rejected,
    test_artifact_identity_round_trips_and_detects_substitution,
    test_calibration_is_bound_to_the_predictor_and_query_set,
    test_calibration_with_another_predictor_is_rejected,
    test_calibration_with_another_query_set_is_rejected,
    test_invalid_query_covariance_fails_closed,
    test_maximum_ten_group_coverage_round_trips_as_finite,
    test_nonworking_covariance_requires_estimator_identity,
    test_semantics_and_covariance_dimension_must_agree,
    test_ten_groups_report_ninety_five_percent_as_unavailable,
    test_uncalibrated_artifact_keeps_raw_semantics_explicit,
    test_uncertainty_namespace_is_narrow_and_explicit,
    test_uncertainty_record_binds_artifact_identity,
    test_wrong_contract_types_are_rejected,
)


def test_finite_group_rank_and_capacity_are_object_level() -> None:
    assert finite_group_conformal_rank(10, 0.9) == 10
    assert finite_group_conformal_rank(10, 0.95) == 11
    assert maximum_finite_group_coverage(10) == pytest.approx(10.0 / 11.0)
    assert minimum_groups_for_finite_conformal(0.9) == 9
    assert minimum_groups_for_finite_conformal(0.95) == 19


def test_boundary_coverage_uses_decimal_exact_rank() -> None:
    assert finite_group_conformal_rank(9, 0.9) == 9
    assert minimum_groups_for_finite_conformal(0.9) == 9


@pytest.mark.parametrize("count", [5, 10, 12])
def test_maximum_finite_coverage_round_trips_through_rank_and_minimum(
    count: int,
) -> None:
    coverage = maximum_finite_group_coverage(count)

    assert finite_group_conformal_rank(count, coverage) == count
    assert minimum_groups_for_finite_conformal(coverage) == count


def test_float_immediately_above_rank_boundary_is_not_snapped_down() -> None:
    count = 5
    coverage = float(np.nextafter(maximum_finite_group_coverage(count), 1.0))

    assert finite_group_conformal_rank(count, coverage) == count + 1


def test_valid_pooled_design_records_the_frozen_information_order() -> None:
    design = plan_finite_group_calibration(
        10,
        0.9,
        pooling="pooled",
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
    )

    assert design.finite_sample_rank == 10
    assert design.maximum_finite_coverage == pytest.approx(10.0 / 11.0)
    assert design.as_dict()["pooling"] == "pooled"


def test_impossible_finite_group_coverage_fails_before_target_access() -> None:
    with pytest.raises(ValueError, match="infinite quantile"):
        plan_finite_group_calibration(
            10,
            0.95,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
        )

    with pytest.raises(ValueError, match="coverage <="):
        plan_finite_group_calibration(
            5,
            0.9,
            pooling="stratum",
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
        )


def test_split_conformal_design_rejects_adaptive_policy_selection() -> None:
    with pytest.raises(ValueError, match="frozen"):
        plan_finite_group_calibration(
            10,
            0.9,
            predictor_frozen_before_scores=False,
            calibration_outcomes_used_for_selection=False,
        )

    with pytest.raises(ValueError, match="cannot also select"):
        plan_finite_group_calibration(
            10,
            0.9,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=True,
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_finite_group_design_rejects_invalid_counts(count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        finite_group_conformal_rank(
            count,  # type: ignore[arg-type]
            0.9,
        )


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, np.nan, True, "0.9"])
def test_finite_group_design_rejects_invalid_coverage(coverage: object) -> None:
    with pytest.raises(ValueError, match="coverage"):
        minimum_groups_for_finite_conformal(
            coverage,  # type: ignore[arg-type]
        )


def test_design_contract_rejects_inconsistent_derived_fields() -> None:
    with pytest.raises(ValueError, match="finite_sample_rank"):
        FiniteGroupCalibrationDesign(
            calibration_group_count=10,
            nominal_coverage=0.9,
            finite_sample_rank=9,
            maximum_finite_coverage=10.0 / 11.0,
            pooling="pooled",
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
        )

    with pytest.raises(ValueError, match="maximum_finite_group_coverage|maximum"):
        FiniteGroupCalibrationDesign(
            calibration_group_count=10,
            nominal_coverage=0.9,
            finite_sample_rank=10,
            maximum_finite_coverage=0.9,
            pooling="pooled",
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
        )


def test_design_contract_rejects_nonliteral_controls() -> None:
    with pytest.raises(ValueError, match="pooling"):
        plan_finite_group_calibration(
            10,
            0.9,
            pooling="frame",  # type: ignore[arg-type]
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
        )

    with pytest.raises(ValueError, match="boolean"):
        plan_finite_group_calibration(
            10,
            0.9,
            predictor_frozen_before_scores=np.bool_(True),  # type: ignore[arg-type]
            calibration_outcomes_used_for_selection=False,
        )

    with pytest.raises(ValueError, match="boolean"):
        plan_finite_group_calibration(
            10,
            0.9,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=np.bool_(False),  # type: ignore[arg-type]
        )


def test_deform360_calibration_amendment_is_finite_and_target_blind() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_calibration_separation.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    descriptor = dict(record)
    declared_id = descriptor.pop("artifact_id")
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert declared_id == hashlib.sha256(canonical).hexdigest()

    primary = descriptor["primary_interval"]
    design = plan_finite_group_calibration(
        primary["calibration_group_count"],
        primary["nominal_coverage"],
        pooling=primary["pooling"],
        predictor_frozen_before_scores=descriptor["information_order"][
            "predictor_score_guard_grouping_and_endpoints_frozen_before_scores"
        ],
        calibration_outcomes_used_for_selection=descriptor["information_order"][
            "calibration_outcomes_used_for_policy_selection"
        ],
    )
    assert design.finite_sample_rank == primary["finite_sample_rank"]
    assert (
        descriptor["stratum_reporting"]["nominal_90_percent_interval_claim"]
        == "forbidden"
    )
    assert descriptor["access_boundary"] == {
        "calibration_payloads_opened": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
    }


def test_binary_calibration_metrics_for_ranked_predictions() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([True, True, False, False]),
        n_bins=5,
    )

    assert metrics.count == 4
    assert metrics.positive_rate == pytest.approx(0.5)
    assert metrics.brier_score == pytest.approx(0.025)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert 0.0 <= metrics.expected_calibration_error <= 1.0


def test_auc_is_undefined_for_one_class() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.8, 0.9]),
        np.array([True, True]),
    )

    assert metrics.roc_auc is None


def test_exact_numeric_binary_labels_are_supported() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.9, 0.1]),
        np.array([1, 0]),
    )

    assert metrics.roc_auc == pytest.approx(1.0)


@pytest.mark.parametrize("probability", [[-0.01], [1.01], [np.nan], [np.inf]])
def test_invalid_probabilities_fail_closed(probability: list[float]) -> None:
    with pytest.raises(ValueError, match="probability"):
        binary_calibration_metrics(np.asarray(probability), np.array([True]))


@pytest.mark.parametrize(
    "target",
    [
        np.array([2]),
        np.array([-1]),
        np.array([0.5]),
        np.array([np.nan]),
        np.array(["True"]),
    ],
)
def test_nonbinary_targets_fail_closed(target: np.ndarray) -> None:
    with pytest.raises(ValueError, match="target"):
        binary_calibration_metrics(np.array([0.5]), target)


@pytest.mark.parametrize("n_bins", [True, 0, -1, 1.5])
def test_invalid_bin_counts_fail_closed(n_bins: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        binary_calibration_metrics(
            np.array([0.5]),
            np.array([True]),
            n_bins=n_bins,  # type: ignore[arg-type]
        )


def test_string_encoded_probability_fails_closed() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        binary_calibration_metrics(
            np.array(["0.5"]),
            np.array([True]),
        )


def test_metrics_contract_rejects_inconsistent_scalars() -> None:
    with pytest.raises(ValueError, match="positive_rate"):
        BinaryCalibrationMetrics(
            count=1,
            positive_rate=1.1,
            brier_score=0.0,
            log_loss=0.0,
            expected_calibration_error=0.0,
            roc_auc=None,
        )


def test_tied_probabilities_use_average_ranks() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.5, 0.5, 0.1, 0.9]),
        np.array([True, False, False, True]),
    )

    assert metrics.roc_auc == pytest.approx(0.875)


def test_metrics_as_dict_preserves_validated_values() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.9, 0.1]),
        np.array([True, False]),
    )

    assert metrics.as_dict()["count"] == 2
    assert metrics.as_dict()["roc_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize("count", [0, True, 1.5])
def test_metrics_contract_rejects_invalid_counts(count: object) -> None:
    with pytest.raises(ValueError, match="count"):
        BinaryCalibrationMetrics(
            count=count,  # type: ignore[arg-type]
            positive_rate=0.5,
            brier_score=0.25,
            log_loss=0.5,
            expected_calibration_error=0.1,
            roc_auc=0.5,
        )


@pytest.mark.parametrize("log_loss", [-0.1, np.nan, "0.5"])
def test_metrics_contract_rejects_invalid_log_loss(log_loss: object) -> None:
    with pytest.raises(ValueError, match="log_loss"):
        BinaryCalibrationMetrics(
            count=1,
            positive_rate=0.5,
            brier_score=0.25,
            log_loss=log_loss,  # type: ignore[arg-type]
            expected_calibration_error=0.1,
            roc_auc=0.5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("positive_rate", True),
        ("positive_rate", "0.5"),
        ("brier_score", 1.1),
        ("expected_calibration_error", np.nan),
        ("roc_auc", -0.1),
    ],
)
def test_metrics_contract_rejects_invalid_unit_interval_values(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "count": 1,
        "positive_rate": 0.5,
        "brier_score": 0.25,
        "log_loss": 0.5,
        "expected_calibration_error": 0.1,
        "roc_auc": 0.5,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        BinaryCalibrationMetrics(**kwargs)  # type: ignore[arg-type]


def test_target_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        binary_calibration_metrics(
            np.array([0.5, 0.5]),
            np.array([True]),
        )


def test_nonvector_probabilities_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        binary_calibration_metrics(
            np.array([[0.5]]),
            np.array([[True]]),
        )


def test_empty_probabilities_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one"):
        binary_calibration_metrics(
            np.array([], dtype=float),
            np.array([], dtype=bool),
        )
