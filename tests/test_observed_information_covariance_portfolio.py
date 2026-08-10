from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin._posterior_covariance_portfolio_common import (
    canonical_string,
    covariance_method,
    projected_semantics,
    validated_covariance,
    validated_query_matrix,
)
from bayesian_phystwin.posterior_covariance_portfolio import (
    PosteriorCovarianceSourceV1,
    PosteriorQueryCovariancePortfolioV1,
    build_posterior_query_covariance_portfolio,
    exact_prior_fallback_covariance_source,
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


def _accepted_sources() -> tuple[PosteriorCovarianceSourceV1, ...]:
    return (
        _source(
            "group_sandwich",
            np.diag([4.0, 1.0]),
            source_character="e",
        ),
        _source(
            "irls_working",
            np.diag([2.0, 3.0]),
            source_character="c",
        ),
        _source(
            "laplace_observed_information",
            np.asarray([[1.0, 0.2], [0.2, 2.0]]),
            source_character="d",
        ),
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


def test_complete_portfolio_orders_and_projects_one_common_query() -> None:
    query = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        query,
        _accepted_sources(),
        inference_admissible=True,
        reason="inference-admissible",
    )

    assert portfolio.methods == (
        "irls_working",
        "laplace_observed_information",
        "group_sandwich",
    )
    assert portfolio.reference_method == "irls_working"
    assert portfolio.descriptor()["selection_semantics"] == (
        "no-implicit-covariance-winner-v1"
    )
    expected = query @ np.diag([2.0, 3.0]) @ query.T
    working = portfolio.entry("irls_working")
    np.testing.assert_allclose(
        working.source_query_covariance_m2,
        expected,
    )
    assert working.covariance_estimator_artifact_id is None
    assert (
        portfolio.entry("laplace_observed_information").covariance_estimator_artifact_id
        == "d" * 64
    )
    assert (
        portfolio.entry("group_sandwich").covariance_estimator_artifact_id == "e" * 64
    )
    assert not portfolio.query_matrix.flags.writeable
    assert not working.source_query_covariance_m2.flags.writeable
    assert portfolio.to_record()["artifact_id"] == portfolio.artifact_id


def test_accepted_portfolio_requires_complete_method_accounting() -> None:
    portfolio = _one_method_portfolio()
    assert portfolio.methods == ("irls_working",)
    assert dict(portfolio.unavailable_methods) == {
        "laplace_observed_information": "not-positive-definite",
        "group_sandwich": "too-few-independent-groups",
    }

    with pytest.raises(ValueError, match="contain or explain"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [_source("irls_working", np.eye(2))],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={
                "laplace_observed_information": "not-positive-definite",
            },
        )
    with pytest.raises(ValueError, match="present and unavailable"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [_source("irls_working", np.eye(2))],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={
                "irls_working": "contradiction",
                "laplace_observed_information": "not-positive-definite",
                "group_sandwich": "too-few-groups",
            },
        )


def test_rejected_portfolio_retains_only_exact_prior_fallback() -> None:
    fallback = exact_prior_fallback_covariance_source(
        RESULT_ID,
        np.diag([2.0, 3.0]),
        source_artifact_id="f" * 64,
        reason="no-identifiable-query-state",
    )
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.asarray([[1.0, -1.0]]),
        [fallback],
        inference_admissible=False,
        reason="no-identifiable-query-state",
    )

    assert portfolio.methods == ("exact_prior_fallback",)
    assert portfolio.reference_method == "exact_prior_fallback"
    np.testing.assert_allclose(
        portfolio.entry("exact_prior_fallback").source_query_covariance_m2,
        np.asarray([[5.0]]),
    )
    with pytest.raises(ValueError, match="only exact fallback"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [_source("irls_working", np.eye(2))],
            inference_admissible=False,
            reason="rejected",
        )
    with pytest.raises(ValueError, match="rejection reason"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [fallback],
            inference_admissible=False,
            reason="inference-admissible",
        )
    with pytest.raises(ValueError, match="fallback covariance reason"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [fallback],
            inference_admissible=False,
            reason="different-rejection",
        )


def test_portfolio_identity_binds_query_and_unavailability_reason() -> None:
    first = _one_method_portfolio()
    second = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.asarray([[1.0, 1.0]]),
        [_source("irls_working", np.eye(2))],
        inference_admissible=True,
        reason="inference-admissible",
        unavailable_methods={
            "laplace_observed_information": "not-positive-definite",
            "group_sandwich": "too-few-independent-groups",
        },
    )
    third = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        [_source("irls_working", np.eye(2))],
        inference_admissible=True,
        reason="inference-admissible",
        unavailable_methods={
            "laplace_observed_information": "not-positive-definite",
            "group_sandwich": "four-groups-required",
        },
    )

    assert len({first.artifact_id, second.artifact_id, third.artifact_id}) == 3
    assert first.source("irls_working").method == "irls_working"
    with pytest.raises(KeyError):
        first.entry("group_sandwich")
    with pytest.raises(KeyError):
        first.source("group_sandwich")


@pytest.mark.parametrize("value", [None, "", " padded "])
def test_common_validator_rejects_noncanonical_strings(value: object) -> None:
    with pytest.raises(ValueError, match="canonical string"):
        canonical_string(value, name="value")


@pytest.mark.parametrize("value", [None, "unknown"])
def test_common_validator_rejects_unknown_covariance_methods(value: object) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        covariance_method(value, name="method")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.asarray([[True]]), "real numeric"),
        (np.ones((2, 3)), "square matrix"),
        (np.empty((0, 0)), "nonempty and finite"),
        (np.diag([1.0, -1.0]), "positive semidefinite"),
    ],
)
def test_common_validator_rejects_invalid_covariance_directly(
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validated_covariance(value, name="covariance")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.asarray([[True]]), "real numeric"),
        (np.asarray([[np.nan]]), "finite"),
    ],
)
def test_common_validator_rejects_invalid_query_directly(
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validated_query_matrix(value)


def test_common_validator_rejects_projected_semantics_metadata_drift() -> None:
    semantics = working_irls_covariance_semantics(
        np.eye(1),
        metadata={"query_matrix_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="metadata contradicts"):
        projected_semantics(
            semantics,
            dimension=1,
            source_id="1" * 64,
            query_id="2" * 64,
        )
