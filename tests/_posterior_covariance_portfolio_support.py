from __future__ import annotations

import numpy as np

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


def semantics(
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
        mixture_curvature_exact=(
            method == "laplace_observed_information"
        ),
        group_score_correction=method == "group_sandwich",
        calibrated=False,
        metadata={"fixture": method},
    )


def source(
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
        covariance_semantics=semantics(method, covariance),
        metadata={"fixture": method},
    )


def accepted_sources() -> tuple[PosteriorCovarianceSourceV1, ...]:
    return (
        source(
            "group_sandwich",
            np.diag([4.0, 1.0]),
            source_character="e",
        ),
        source(
            "irls_working",
            np.diag([2.0, 3.0]),
            source_character="c",
        ),
        source(
            "laplace_observed_information",
            np.asarray([[1.0, 0.2], [0.2, 2.0]]),
            source_character="d",
        ),
    )


def one_method_portfolio() -> PosteriorQueryCovariancePortfolioV1:
    return build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        [source("irls_working", np.eye(2))],
        inference_admissible=True,
        reason="inference-admissible",
        unavailable_methods={
            "laplace_observed_information": "not-positive-definite",
            "group_sandwich": "too-few-independent-groups",
        },
    )
