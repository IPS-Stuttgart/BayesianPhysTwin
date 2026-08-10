from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.posterior_covariance_portfolio import (
    build_posterior_query_covariance_portfolio,
    exact_prior_fallback_covariance_source,
)

from _posterior_covariance_portfolio_support import (
    QUERY_ID,
    RESULT_ID,
    accepted_sources,
    one_method_portfolio,
    source,
)


def test_complete_portfolio_orders_and_projects_one_common_query() -> None:
    query = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        query,
        accepted_sources(),
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
    assert portfolio.entry(
        "laplace_observed_information"
    ).covariance_estimator_artifact_id == "d" * 64
    assert portfolio.entry(
        "group_sandwich"
    ).covariance_estimator_artifact_id == "e" * 64
    assert not portfolio.query_matrix.flags.writeable
    assert not working.source_query_covariance_m2.flags.writeable
    assert portfolio.to_record()["artifact_id"] == portfolio.artifact_id


def test_accepted_portfolio_requires_complete_method_accounting() -> None:
    portfolio = one_method_portfolio()
    assert portfolio.methods == ("irls_working",)
    assert tuple(portfolio.unavailable_methods) == (
        "laplace_observed_information",
        "group_sandwich",
    )

    with pytest.raises(ValueError, match="contain or explain"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [source("irls_working", np.eye(2))],
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
            [source("irls_working", np.eye(2))],
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
        portfolio.entry(
            "exact_prior_fallback"
        ).source_query_covariance_m2,
        np.asarray([[5.0]]),
    )
    with pytest.raises(ValueError, match="only exact fallback"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [source("irls_working", np.eye(2))],
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


def test_portfolio_identity_binds_query_and_unavailability_reason() -> None:
    first = one_method_portfolio()
    second = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.asarray([[1.0, 1.0]]),
        [source("irls_working", np.eye(2))],
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
        [source("irls_working", np.eye(2))],
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
