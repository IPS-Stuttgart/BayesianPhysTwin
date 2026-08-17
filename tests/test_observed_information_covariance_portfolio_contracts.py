from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.posterior_covariance_portfolio import (
    PosteriorCovarianceSourceV1,
    PosteriorQueryCovariancePortfolioV1,
    build_posterior_query_covariance_portfolio,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceMethod,
    PosteriorCovarianceSemanticsV1,
    working_irls_covariance_semantics,
)

RESULT_ID = "a" * 64
QUERY_ID = "b" * 64
LIKELIHOOD = "grouped-student-t-generalized-bayes-power-v1"


def _semantics(
    method: PosteriorCovarianceMethod,
    covariance: np.ndarray,
) -> PosteriorCovarianceSemanticsV1:
    if method == "irls_working":
        return working_irls_covariance_semantics(covariance)
    return PosteriorCovarianceSemanticsV1(
        method=method,
        dimension=len(covariance),
        likelihood_power_semantics=LIKELIHOOD,
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=(method == "laplace_observed_information"),
        group_score_correction=method == "group_sandwich",
        calibrated=False,
        metadata={"fixture": method},
    )


def _source(
    method: PosteriorCovarianceMethod,
    covariance: np.ndarray,
    *,
    result_id: str = RESULT_ID,
    source_character: str = "c",
) -> PosteriorCovarianceSourceV1:
    return PosteriorCovarianceSourceV1(
        inference_result_id=result_id,
        source_artifact_id=source_character * 64,
        covariance=covariance,
        covariance_semantics=_semantics(method, covariance),
        metadata={"fixture": method},
    )


def _one_method_portfolio() -> PosteriorQueryCovariancePortfolioV1:
    return build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        [_source("irls_working", np.eye(2))],
        inference_admissible=True,
        reason="inference-admissible",
        unavailable_methods={
            "laplace_observed_information": "not-positive-definite",
            "group_sandwich": "too-few-independent-groups",
        },
    )


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.diag([1.0, -1.0]), "positive semidefinite"),
        (np.ones((2, 3)), "square matrix"),
        (np.asarray([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.asarray([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (np.asarray([[True]]), "real numeric"),
    ],
)
def test_source_rejects_invalid_covariance(
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _source("irls_working", covariance)


def test_source_rejects_high_dynamic_range_indefinite_covariance() -> None:
    covariance = np.diag([1.0e12, -1.0e-2])
    with pytest.raises(ValueError, match="positive semidefinite"):
        _source("irls_working", covariance)


def test_source_rejects_calibration_dimension_and_forged_identity() -> None:
    valid = _source("irls_working", np.eye(2))
    assert not valid.covariance.flags.writeable
    assert valid.to_record()["artifact_id"] == valid.artifact_id

    with pytest.raises(ValueError, match="dimension"):
        PosteriorCovarianceSourceV1(
            inference_result_id=RESULT_ID,
            source_artifact_id="c" * 64,
            covariance=np.eye(2),
            covariance_semantics=working_irls_covariance_semantics(np.eye(3)),
        )
    calibrated = PosteriorCovarianceSemanticsV1(
        method="irls_working",
        dimension=2,
        likelihood_power_semantics=LIKELIHOOD,
        calibrated=True,
        calibration_artifact_id="9" * 64,
    )
    with pytest.raises(ValueError, match="uncalibrated"):
        replace(
            valid,
            covariance_semantics=calibrated,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="artifact_id"):
        replace(valid, artifact_id="0" * 64)


def test_builder_rejects_mixed_results_and_duplicate_methods() -> None:
    working = _source("irls_working", np.eye(2))
    different_result = _source(
        "laplace_observed_information",
        np.eye(2),
        result_id="1" * 64,
        source_character="d",
    )
    with pytest.raises(ValueError, match="inference_result_id"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [working, different_result],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={"group_sandwich": "not-available"},
        )
    with pytest.raises(ValueError, match="methods must be unique"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [
                working,
                _source(
                    "irls_working",
                    np.eye(2),
                    source_character="d",
                ),
            ],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={
                "laplace_observed_information": "not-available",
                "group_sandwich": "not-available",
            },
        )


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([[True, False]]),
        np.asarray([["1", "0"]]),
        np.asarray([[np.nan, 0.0]]),
        np.ones((1, 3)),
    ],
)
def test_builder_rejects_invalid_query_matrix(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="query_matrix"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            invalid,
            [_source("irls_working", np.eye(2))],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={
                "laplace_observed_information": "not-available",
                "group_sandwich": "not-available",
            },
        )


def test_portfolio_rejects_forged_identity_and_owns_arrays() -> None:
    portfolio = _one_method_portfolio()
    assert not portfolio.query_matrix.flags.writeable
    assert not portfolio.entry(
        "irls_working"
    ).source_query_covariance_m2.flags.writeable
    with pytest.raises(ValueError, match="artifact_id"):
        replace(portfolio, artifact_id="0" * 64)
