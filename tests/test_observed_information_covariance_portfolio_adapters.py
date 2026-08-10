from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from bayesian_phystwin.group_sandwich_covariance import (
    GroupSandwichCovarianceResultV1,
)
from bayesian_phystwin.observed_information_covariance import (
    ObservedInformationCovarianceResultV1,
)
from bayesian_phystwin.posterior_covariance_portfolio import (
    exact_prior_fallback_covariance_source,
    group_sandwich_covariance_source,
    observed_information_covariance_source,
    working_covariance_source,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceMethod,
    PosteriorCovarianceSemanticsV1,
)

RESULT_ID = "a" * 64
LIKELIHOOD = "grouped-student-t-generalized-bayes-power-v1"


def _semantics(
    method: PosteriorCovarianceMethod,
    covariance: np.ndarray,
) -> PosteriorCovarianceSemanticsV1:
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


def test_working_and_fallback_adapters_bind_required_semantics() -> None:
    working = working_covariance_source(
        RESULT_ID,
        np.eye(2),
        source_artifact_id="c" * 64,
    )
    assert working.method == "irls_working"
    assert working.covariance_semantics.metadata["portfolio_source"] == ("working-irls")
    fallback = exact_prior_fallback_covariance_source(
        RESULT_ID,
        np.eye(2),
        source_artifact_id="f" * 64,
        reason="no-identifiable-query-state",
    )
    assert fallback.method == "exact_prior_fallback"
    assert fallback.covariance_semantics.metadata["fallback_reason"] == (
        "no-identifiable-query-state"
    )
    with pytest.raises(ValueError, match="contradicts portfolio_source"):
        working_covariance_source(
            RESULT_ID,
            np.eye(2),
            source_artifact_id="c" * 64,
            metadata={"portfolio_source": "changed"},
        )


def test_observed_information_adapter_preserves_estimator_identity() -> None:
    result = object.__new__(ObservedInformationCovarianceResultV1)
    object.__setattr__(result, "artifact_id", "d" * 64)
    object.__setattr__(result, "full_covariance", np.eye(2) * 2.0)
    object.__setattr__(
        result,
        "covariance_semantics",
        _semantics("laplace_observed_information", np.eye(2)),
    )

    source = observed_information_covariance_source(RESULT_ID, result)
    assert source.source_artifact_id == "d" * 64
    assert source.method == "laplace_observed_information"
    np.testing.assert_allclose(source.covariance, np.eye(2) * 2.0)
    with pytest.raises(TypeError, match="ObservedInformation"):
        observed_information_covariance_source(
            RESULT_ID,
            cast(ObservedInformationCovarianceResultV1, object()),
        )


def test_group_sandwich_adapter_preserves_estimator_identity() -> None:
    result = object.__new__(GroupSandwichCovarianceResultV1)
    object.__setattr__(result, "artifact_id", "e" * 64)
    object.__setattr__(result, "covariance", np.eye(2) * 3.0)
    object.__setattr__(
        result,
        "covariance_semantics",
        _semantics("group_sandwich", np.eye(2)),
    )

    source = group_sandwich_covariance_source(RESULT_ID, result)
    assert source.source_artifact_id == "e" * 64
    assert source.method == "group_sandwich"
    np.testing.assert_allclose(source.covariance, np.eye(2) * 3.0)
    with pytest.raises(TypeError, match="GroupSandwich"):
        group_sandwich_covariance_source(
            RESULT_ID,
            cast(GroupSandwichCovarianceResultV1, object()),
        )
