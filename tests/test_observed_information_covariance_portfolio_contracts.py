from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.posterior_covariance_portfolio import (
    PosteriorCovarianceSourceV1,
    build_posterior_query_covariance_portfolio,
    exact_prior_fallback_covariance_source,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    working_irls_covariance_semantics,
)

from _posterior_covariance_portfolio_support import (
    LIKELIHOOD,
    QUERY_ID,
    RESULT_ID,
    one_method_portfolio,
    source,
)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.diag([1.0, -1.0]), "positive semidefinite"),
        (np.diag([1.0e12, -1.0]), "positive semidefinite"),
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
        source("irls_working", covariance)


def test_source_rejects_calibration_dimension_and_forged_identity() -> None:
    valid = source("irls_working", np.eye(2))
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
    working = source("irls_working", np.eye(2))
    different_result = source(
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
                source(
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
            [source("irls_working", np.eye(2))],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={
                "laplace_observed_information": "not-available",
                "group_sandwich": "not-available",
            },
        )


def test_rejected_portfolio_reason_matches_fallback_semantics() -> None:
    fallback = exact_prior_fallback_covariance_source(
        RESULT_ID,
        np.eye(2),
        source_artifact_id="f" * 64,
        reason="no-identifiable-query-state",
    )
    with pytest.raises(ValueError, match="reason does not match exact fallback"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [fallback],
            inference_admissible=False,
            reason="different-rejection-reason",
        )


def test_portfolio_rejects_forged_identity_and_owns_arrays() -> None:
    portfolio = one_method_portfolio()
    assert not portfolio.query_matrix.flags.writeable
    assert not portfolio.entry(
        "irls_working"
    ).source_query_covariance_m2.flags.writeable
    with pytest.raises(ValueError, match="artifact_id"):
        replace(portfolio, artifact_id="0" * 64)
