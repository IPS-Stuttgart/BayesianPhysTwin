from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.posterior_covariance_portfolio import (
    PosteriorCovarianceSourceV1,
    PosteriorQueryCovariancePortfolioV1,
    build_posterior_query_covariance_portfolio,
    exact_prior_fallback_covariance_source,
    group_sandwich_covariance_source,
    observed_information_covariance_source,
    working_covariance_source,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    exact_prior_fallback_covariance_semantics,
    working_irls_covariance_semantics,
)
from bayesian_phystwin.posterior_uncertainty import PosteriorQueryUncertaintyV1

RESULT_ID = "a" * 64
QUERY_ID = "b" * 64
LIKELIHOOD = "grouped-student-t-generalized-bayes-power-v1"


def _semantics(method: str, covariance: np.ndarray) -> PosteriorCovarianceSemanticsV1:
    if method == "irls_working":
        return working_irls_covariance_semantics(covariance)
    if method == "exact_prior_fallback":
        return exact_prior_fallback_covariance_semantics(
            covariance,
            reason="no-identifiable-query-state",
        )
    return PosteriorCovarianceSemanticsV1(
        method=method,  # type: ignore[arg-type]
        dimension=len(covariance),
        likelihood_power_semantics=LIKELIHOOD,
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=method == "laplace_observed_information",
        group_score_correction=method == "group_sandwich",
        calibrated=False,
        metadata={"test_method": method},
    )


def _source(
    method: str,
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


def test_builds_canonical_complete_raw_covariance_portfolio() -> None:
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
    assert tuple(source.method for source in portfolio.sources) == portfolio.methods
    assert portfolio.reference_method == "irls_working"
    assert portfolio.query_dimension == 2
    assert portfolio.parameter_dimension == 2
    assert portfolio.unavailable_methods == {}
    assert portfolio.descriptor()["selection_semantics"] == (
        "no-implicit-covariance-winner-v1"
    )
    assert not portfolio.query_matrix.flags.writeable

    working = portfolio.entry("irls_working")
    expected = query @ np.diag([2.0, 3.0]) @ query.T
    np.testing.assert_allclose(working.source_query_covariance_m2, expected)
    assert working.covariance_estimator_artifact_id is None
    assert not working.source_query_covariance_m2.flags.writeable

    observed = portfolio.entry("laplace_observed_information")
    assert observed.covariance_estimator_artifact_id == "d" * 64
    assert observed.source_covariance_semantics.mixture_curvature_exact

    sandwich = portfolio.entry("group_sandwich")
    assert sandwich.covariance_estimator_artifact_id == "e" * 64
    assert sandwich.source_covariance_semantics.group_score_correction
    assert portfolio.to_record()["artifact_id"] == portfolio.artifact_id


def test_requires_every_accepted_method_or_explicit_unavailability() -> None:
    working = _source("irls_working", np.eye(2))
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        [working],
        inference_admissible=True,
        reason="inference-admissible",
        unavailable_methods={
            "laplace_observed_information": (
                "observed-information-not-positive-definite"
            ),
            "group_sandwich": "fewer-than-three-independent-groups",
        },
    )
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
            [working],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={
                "laplace_observed_information": "not-positive-definite",
            },
        )


def test_rejected_portfolio_contains_only_exact_prior_fallback() -> None:
    covariance = np.diag([2.0, 3.0])
    fallback = exact_prior_fallback_covariance_source(
        RESULT_ID,
        covariance,
        source_artifact_id="f" * 64,
        reason="no-identifiable-query-state",
    )
    query = np.asarray([[1.0, -1.0]])
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        query,
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
        PosteriorQueryCovariancePortfolioV1(
            inference_result_id=RESULT_ID,
            query_set_id=QUERY_ID,
            query_matrix=query,
            sources=(fallback, _source("irls_working", np.eye(2))),
            entries=(
                portfolio.entry("exact_prior_fallback"),
                _working_entry(query),
            ),
            reference_method="exact_prior_fallback",
            inference_admissible=False,
            reason="rejected",
        )


def _working_entry(query: np.ndarray) -> PosteriorQueryUncertaintyV1:
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        query,
        [_source("irls_working", np.eye(2))],
        inference_admissible=True,
        reason="inference-admissible",
        unavailable_methods={
            "laplace_observed_information": "not-requested",
            "group_sandwich": "not-requested",
        },
    )
    return portfolio.entry("irls_working")


def test_source_contract_rejects_invalid_covariance_and_semantics() -> None:
    covariance = np.eye(2)
    source = _source("irls_working", covariance)
    assert not source.covariance.flags.writeable
    assert source.to_record()["artifact_id"] == source.artifact_id

    with pytest.raises(ValueError, match="positive semidefinite"):
        PosteriorCovarianceSourceV1(
            inference_result_id=RESULT_ID,
            source_artifact_id="c" * 64,
            covariance=np.diag([1.0, -1.0]),
            covariance_semantics=_semantics("irls_working", covariance),
        )
    with pytest.raises(ValueError, match="dimension"):
        PosteriorCovarianceSourceV1(
            inference_result_id=RESULT_ID,
            source_artifact_id="c" * 64,
            covariance=covariance,
            covariance_semantics=_semantics("irls_working", np.eye(3)),
        )
    calibrated = PosteriorCovarianceSemanticsV1(
        method="irls_working",
        dimension=2,
        likelihood_power_semantics=LIKELIHOOD,
        calibrated=True,
        calibration_artifact_id="9" * 64,
    )
    with pytest.raises(ValueError, match="uncalibrated"):
        PosteriorCovarianceSourceV1(
            inference_result_id=RESULT_ID,
            source_artifact_id="c" * 64,
            covariance=covariance,
            covariance_semantics=calibrated,
        )
    with pytest.raises(ValueError, match="artifact_id"):
        replace(source, artifact_id="0" * 64)


def test_convenience_sources_bind_required_semantics() -> None:
    working = working_covariance_source(
        RESULT_ID,
        np.eye(2),
        source_artifact_id="c" * 64,
    )
    assert working.method == "irls_working"
    assert working.covariance_semantics.metadata["portfolio_source"] == ("working-irls")
    with pytest.raises(ValueError, match="contradicts portfolio_source"):
        working_covariance_source(
            RESULT_ID,
            np.eye(2),
            source_artifact_id="c" * 64,
            metadata={"portfolio_source": "changed"},
        )
    with pytest.raises(TypeError, match="ObservedInformation"):
        observed_information_covariance_source(
            RESULT_ID,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="GroupSandwich"):
        group_sandwich_covariance_source(RESULT_ID, object())  # type: ignore[arg-type]


def test_builder_rejects_mixed_results_duplicate_methods_and_bad_queries() -> None:
    working = _source("irls_working", np.eye(2))
    different = _source(
        "laplace_observed_information",
        np.eye(2),
        result_id="1" * 64,
        source_character="d",
    )
    common = {
        "inference_admissible": True,
        "reason": "inference-admissible",
        "unavailable_methods": {
            "group_sandwich": "not-available",
        },
    }
    with pytest.raises(ValueError, match="inference_result_id"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [working, different],
            **common,
        )
    with pytest.raises(ValueError, match="methods must be unique"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.eye(2),
            [working, _source("irls_working", np.eye(2), source_character="d")],
            **common,
        )
    for invalid in (
        np.asarray([[True, False]]),
        np.asarray([["1", "0"]]),
        np.asarray([[np.nan, 0.0]]),
    ):
        with pytest.raises(ValueError, match="query_matrix"):
            build_posterior_query_covariance_portfolio(
                RESULT_ID,
                QUERY_ID,
                invalid,
                [working],
                inference_admissible=True,
                reason="inference-admissible",
                unavailable_methods={
                    "laplace_observed_information": "not-available",
                    "group_sandwich": "not-available",
                },
            )
    with pytest.raises(ValueError, match="width"):
        build_posterior_query_covariance_portfolio(
            RESULT_ID,
            QUERY_ID,
            np.ones((1, 3)),
            [working],
            inference_admissible=True,
            reason="inference-admissible",
            unavailable_methods={
                "laplace_observed_information": "not-available",
                "group_sandwich": "not-available",
            },
        )


def test_portfolio_recomputes_projection_and_revalidates_identities() -> None:
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        _accepted_sources(),
        inference_admissible=True,
        reason="inference-admissible",
    )
    working = portfolio.entry("irls_working")
    tampered_covariance = replace(
        working,
        source_query_covariance_m2=np.eye(2) * 99.0,
        artifact_id=None,
    )
    entries = tuple(
        tampered_covariance if entry is working else entry
        for entry in portfolio.entries
    )
    with pytest.raises(ValueError, match="does not match source"):
        replace(portfolio, entries=entries)

    observed = portfolio.entry("laplace_observed_information")
    tampered_metadata = replace(
        observed,
        metadata={**observed.metadata, "query_matrix_sha256": "0" * 64},
        artifact_id=None,
    )
    entries = tuple(
        tampered_metadata if entry is observed else entry for entry in portfolio.entries
    )
    with pytest.raises(ValueError, match="query matrix identity"):
        replace(portfolio, entries=entries)

    tampered_source = replace(
        portfolio.source("group_sandwich"),
        source_artifact_id="8" * 64,
        artifact_id=None,
    )
    sources = tuple(
        tampered_source if source.method == "group_sandwich" else source
        for source in portfolio.sources
    )
    with pytest.raises(ValueError, match="source identity"):
        replace(portfolio, sources=sources)


def test_portfolio_identity_changes_with_query_or_unavailable_reason() -> None:
    working = _source("irls_working", np.eye(2))
    common = {
        "inference_admissible": True,
        "reason": "inference-admissible",
        "unavailable_methods": {
            "laplace_observed_information": "not-positive-definite",
            "group_sandwich": "too-few-groups",
        },
    }
    first = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        [working],
        **common,
    )
    second = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.asarray([[1.0, 1.0]]),
        [working],
        **common,
    )
    third = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        [working],
        inference_admissible=True,
        reason="inference-admissible",
        unavailable_methods={
            "laplace_observed_information": "not-positive-definite",
            "group_sandwich": "four-groups-required",
        },
    )
    assert len({first.artifact_id, second.artifact_id, third.artifact_id}) == 3
    with pytest.raises(KeyError):
        first.entry("group_sandwich")
    with pytest.raises(KeyError):
        first.source("group_sandwich")


def test_portfolio_hardening_rejects_semantic_and_reason_drift() -> None:
    portfolio = build_posterior_query_covariance_portfolio(
        RESULT_ID,
        QUERY_ID,
        np.eye(2),
        _accepted_sources(),
        inference_admissible=True,
        reason="inference-admissible",
    )
    working = portfolio.entry("irls_working")
    semantics = working.source_covariance_semantics
    tampered_semantics = PosteriorCovarianceSemanticsV1(
        method=semantics.method,
        dimension=semantics.dimension,
        likelihood_power_semantics="different-generalized-bayes-power-v1",
        prior_included=semantics.prior_included,
        generalized_bayes=semantics.generalized_bayes,
        mixture_curvature_exact=semantics.mixture_curvature_exact,
        group_score_correction=semantics.group_score_correction,
        calibrated=False,
        metadata=semantics.metadata,
    )
    tampered_entry = replace(
        working,
        source_covariance_semantics=tampered_semantics,
        artifact_id=None,
    )
    entries = tuple(
        tampered_entry if entry is working else entry for entry in portfolio.entries
    )
    with pytest.raises(ValueError, match="projected covariance semantics changed"):
        replace(portfolio, entries=entries, artifact_id=None)

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


def test_portfolio_source_rejects_large_scale_material_indefiniteness() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        PosteriorCovarianceSourceV1(
            inference_result_id=RESULT_ID,
            source_artifact_id="c" * 64,
            covariance=np.diag([1.0e12, -1.0]),
            covariance_semantics=_semantics("irls_working", np.eye(2)),
        )
