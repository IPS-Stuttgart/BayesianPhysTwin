from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    working_irls_covariance_semantics,
)
from bayesian_phystwin.posterior_uncertainty import (
    PosteriorQueryUncertaintyV1,
    finite_group_coverage_status,
)
from bayesian_phystwin.query_calibration import (
    calibrate_query_covariance,
    fit_query_calibration,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_covariance() -> np.ndarray:
    return np.asarray(
        [
            [[4.0, 0.4], [0.4, 1.0]],
            [[2.0, 0.1], [0.1, 3.0]],
        ],
        dtype=np.float64,
    )


def _uncalibrated() -> PosteriorQueryUncertaintyV1:
    covariance = _source_covariance()
    return PosteriorQueryUncertaintyV1(
        inference_result_id=_digest("inference"),
        query_set_id=_digest("query-set"),
        source_query_covariance_m2=covariance,
        source_covariance_semantics=working_irls_covariance_semantics(
            covariance[0],
            metadata={"query_projection": "registered-linearization-v1"},
        ),
        metadata={"protocol": "prospective-test-v1"},
    )


def _calibration(
    source: PosteriorQueryUncertaintyV1,
    *,
    predictor_id: str | None = None,
    query_set_id: str | None = None,
):
    covariance = _source_covariance()
    residual_groups = [
        np.asarray(
            [
                [0.10 + 0.01 * index, 0.01],
                [0.02, 0.12 + 0.01 * index],
            ]
        )
        for index in range(9)
    ]
    covariance_groups = [covariance for _ in range(9)]
    return fit_query_calibration(
        tuple(f"object-{index:02d}" for index in range(9)),
        residual_groups,
        covariance_groups,
        nominal_coverage=0.9,
        predictor_id=source.predictor_id if predictor_id is None else predictor_id,
        query_set_id=source.query_set_id if query_set_id is None else query_set_id,
        grouping_rule_id=_digest("independent-object-v1"),
        guard_id=_digest("guard-v1"),
        calibration_evidence_id=_digest("calibration-evidence-v1"),
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
    )


def test_uncalibrated_artifact_keeps_raw_semantics_explicit() -> None:
    uncertainty = _uncalibrated()

    assert uncertainty.calibrated is False
    assert uncertainty.nominal_coverage is None
    assert uncertainty.calibration_group_count is None
    assert uncertainty.calibrated_query_covariance_m2 is None
    assert uncertainty.reported_covariance_semantics.calibrated is False
    np.testing.assert_array_equal(
        uncertainty.reported_query_covariance_m2,
        uncertainty.source_query_covariance_m2,
    )
    with pytest.raises(ValueError):
        uncertainty.source_query_covariance_m2.setflags(write=True)


def test_calibration_is_bound_to_the_predictor_and_query_set() -> None:
    source = _uncalibrated()
    calibration = _calibration(source)
    calibrated = PosteriorQueryUncertaintyV1(
        inference_result_id=source.inference_result_id,
        query_set_id=source.query_set_id,
        source_query_covariance_m2=source.source_query_covariance_m2,
        source_covariance_semantics=source.source_covariance_semantics,
        query_calibration=calibration,
        metadata=source.metadata,
    )

    assert calibrated.predictor_id == source.predictor_id
    assert calibrated.artifact_id != source.artifact_id
    assert calibrated.calibrated is True
    assert calibrated.nominal_coverage == 0.9
    assert calibrated.calibration_group_count == 9
    assert calibrated.reported_covariance_semantics.calibrated is True
    assert (
        calibrated.reported_covariance_semantics.calibration_artifact_id
        == calibration.artifact_id
    )
    np.testing.assert_allclose(
        calibrated.reported_query_covariance_m2,
        calibrate_query_covariance(source.source_query_covariance_m2, calibration),
    )
    assert calibrated.calibrated_query_covariance_m2 is not None
    with pytest.raises(ValueError):
        calibrated.calibrated_query_covariance_m2.setflags(write=True)


def test_calibration_with_another_predictor_is_rejected() -> None:
    source = _uncalibrated()
    calibration = _calibration(source, predictor_id=_digest("other-predictor"))

    with pytest.raises(ValueError, match="predictor_id"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=source.inference_result_id,
            query_set_id=source.query_set_id,
            source_query_covariance_m2=source.source_query_covariance_m2,
            source_covariance_semantics=source.source_covariance_semantics,
            query_calibration=calibration,
        )


def test_calibration_with_another_query_set_is_rejected() -> None:
    source = _uncalibrated()
    calibration = _calibration(source, query_set_id=_digest("other-query-set"))

    with pytest.raises(ValueError, match="query_set_id"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=source.inference_result_id,
            query_set_id=source.query_set_id,
            source_query_covariance_m2=source.source_query_covariance_m2,
            source_covariance_semantics=source.source_covariance_semantics,
            query_calibration=calibration,
        )


def test_nonworking_covariance_requires_estimator_identity() -> None:
    covariance = _source_covariance()
    semantics = PosteriorCovarianceSemanticsV1(
        method="group_sandwich",
        dimension=2,
        likelihood_power_semantics="grouped-student-t-generalized-bayes-power-v1",
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=False,
        group_score_correction=True,
        calibrated=False,
        metadata={"grouping_semantics": "independent-object-v1"},
    )

    with pytest.raises(ValueError, match="covariance_estimator_artifact_id"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=_digest("inference"),
            query_set_id=_digest("query-set"),
            source_query_covariance_m2=covariance,
            source_covariance_semantics=semantics,
        )

    uncertainty = PosteriorQueryUncertaintyV1(
        inference_result_id=_digest("inference"),
        query_set_id=_digest("query-set"),
        source_query_covariance_m2=covariance,
        source_covariance_semantics=semantics,
        covariance_estimator_artifact_id=_digest("group-sandwich-result"),
    )
    assert uncertainty.source_covariance_semantics.method == "group_sandwich"


def test_semantics_and_covariance_dimension_must_agree() -> None:
    semantics = working_irls_covariance_semantics(np.eye(3))
    with pytest.raises(ValueError, match="dimension"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=_digest("inference"),
            query_set_id=_digest("query-set"),
            source_query_covariance_m2=_source_covariance(),
            source_covariance_semantics=semantics,
        )


def test_already_calibrated_source_semantics_are_rejected() -> None:
    covariance = _source_covariance()
    semantics = PosteriorCovarianceSemanticsV1(
        method="irls_working",
        dimension=2,
        likelihood_power_semantics="grouped-student-t-generalized-bayes-power-v1",
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=False,
        group_score_correction=False,
        calibrated=True,
        calibration_artifact_id=_digest("old-calibration"),
    )
    with pytest.raises(ValueError, match="uncalibrated"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=_digest("inference"),
            query_set_id=_digest("query-set"),
            source_query_covariance_m2=covariance,
            source_covariance_semantics=semantics,
        )


def test_artifact_identity_round_trips_and_detects_substitution() -> None:
    uncertainty = _uncalibrated()
    reconstructed = PosteriorQueryUncertaintyV1(
        inference_result_id=uncertainty.inference_result_id,
        query_set_id=uncertainty.query_set_id,
        source_query_covariance_m2=uncertainty.source_query_covariance_m2,
        source_covariance_semantics=uncertainty.source_covariance_semantics,
        metadata=uncertainty.metadata,
        artifact_id=uncertainty.artifact_id,
    )
    assert reconstructed.artifact_id == uncertainty.artifact_id

    with pytest.raises(ValueError, match="artifact_id"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=uncertainty.inference_result_id,
            query_set_id=uncertainty.query_set_id,
            source_query_covariance_m2=2.0 * uncertainty.source_query_covariance_m2,
            source_covariance_semantics=uncertainty.source_covariance_semantics,
            metadata=uncertainty.metadata,
            artifact_id=uncertainty.artifact_id,
        )


def test_ten_groups_report_ninety_five_percent_as_unavailable() -> None:
    status = finite_group_coverage_status(10, 0.95)

    assert status.finite is False
    assert status.status == "unavailable"
    assert status.finite_sample_rank == 11
    assert status.maximum_finite_coverage == pytest.approx(10.0 / 11.0)
    assert status.minimum_required_group_count == 19


def test_maximum_ten_group_coverage_round_trips_as_finite() -> None:
    status = finite_group_coverage_status(10, 10.0 / 11.0)

    assert status.finite is True
    assert status.status == "finite"
    assert status.finite_sample_rank == 10
    assert status.minimum_required_group_count == 10
    assert status.to_record()["artifact_id"] == status.artifact_id


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.asarray([["x"]]), "real numeric"),
        (np.ones(3), "square matrices"),
        (np.empty((0, 0)), "nonempty"),
        (np.asarray([[np.nan, 0.0], [0.0, 1.0]]), "finite"),
        (np.asarray([[1.0, 1.0], [0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, -1.0]), "positive semidefinite"),
    ],
)
def test_invalid_query_covariance_fails_closed(
    covariance: np.ndarray,
    message: str,
) -> None:
    semantics = working_irls_covariance_semantics(np.eye(1))
    with pytest.raises(ValueError, match=message):
        PosteriorQueryUncertaintyV1(
            inference_result_id=_digest("inference"),
            query_set_id=_digest("query-set"),
            source_query_covariance_m2=covariance,
            source_covariance_semantics=semantics,
        )


def test_wrong_contract_types_are_rejected() -> None:
    covariance = _source_covariance()
    with pytest.raises(ValueError, match="source_covariance_semantics"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=_digest("inference"),
            query_set_id=_digest("query-set"),
            source_query_covariance_m2=covariance,
            source_covariance_semantics=object(),  # type: ignore[arg-type]
        )

    source = _uncalibrated()
    with pytest.raises(ValueError, match="query_calibration"):
        PosteriorQueryUncertaintyV1(
            inference_result_id=source.inference_result_id,
            query_set_id=source.query_set_id,
            source_query_covariance_m2=source.source_query_covariance_m2,
            source_covariance_semantics=source.source_covariance_semantics,
            query_calibration=object(),  # type: ignore[arg-type]
        )


def test_uncertainty_record_binds_artifact_identity() -> None:
    uncertainty = _uncalibrated()
    assert uncertainty.to_record()["artifact_id"] == uncertainty.artifact_id
