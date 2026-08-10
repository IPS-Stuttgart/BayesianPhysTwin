from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin._posterior_covariance_portfolio_common import (
    canonical_string,
    covariance_method,
    portfolio_metadata,
    projected_semantics,
    sha256_id,
    validated_covariance,
    validated_query_matrix,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    working_irls_covariance_semantics,
)

LIKELIHOOD = "grouped-student-t-generalized-bayes-power-v1"


def test_common_identifier_string_and_method_validation() -> None:
    assert sha256_id("a" * 64, name="identifier") == "a" * 64
    with pytest.raises((TypeError, ValueError)):
        sha256_id("not-a-digest", name="identifier")

    assert canonical_string("accepted", name="reason") == "accepted"
    for invalid in ("", " padded ", True):
        with pytest.raises(ValueError, match="canonical string"):
            canonical_string(invalid, name="reason")

    assert covariance_method("irls_working", name="method") == "irls_working"
    for invalid in ("unknown", True):
        with pytest.raises(ValueError, match="must be one of"):
            covariance_method(invalid, name="method")


def test_common_covariance_validation_rejection_paths() -> None:
    accepted = validated_covariance(np.eye(2), name="covariance")
    assert not accepted.flags.writeable

    invalid_cases = (
        (np.asarray([[True]]), "real numeric"),
        (np.asarray([1.0, 2.0]), "square matrix"),
        (np.empty((0, 0)), "nonempty and finite"),
        (np.asarray([[1.0, np.nan], [np.nan, 1.0]]), "nonempty and finite"),
        (np.asarray([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, -1.0]), "positive semidefinite"),
    )
    for covariance, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            validated_covariance(covariance, name="covariance")


def test_common_query_matrix_validation_rejection_paths() -> None:
    accepted = validated_query_matrix(np.eye(2))
    assert not accepted.flags.writeable

    invalid_cases = (
        (np.asarray([[True]]), "real numeric"),
        (np.asarray([1.0, 2.0]), "nonempty matrix"),
        (np.empty((0, 2)), "nonempty matrix"),
        (np.asarray([[np.nan, 0.0]]), "finite"),
    )
    for query, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            validated_query_matrix(query)


def test_required_metadata_conflicts_fail_closed() -> None:
    metadata = portfolio_metadata(
        None,
        name="metadata",
        required={"adapter": "expected"},
    )
    assert metadata["adapter"] == "expected"

    with pytest.raises(ValueError, match="contradicts adapter"):
        portfolio_metadata(
            {"adapter": "changed"},
            name="metadata",
            required={"adapter": "expected"},
        )


def test_projected_semantics_rejects_conflicting_lineage() -> None:
    source = PosteriorCovarianceSemanticsV1(
        method="irls_working",
        dimension=2,
        likelihood_power_semantics=LIKELIHOOD,
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=False,
        group_score_correction=False,
        calibrated=False,
        metadata={"query_matrix_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="query_matrix_sha256"):
        projected_semantics(
            source,
            dimension=1,
            source_id="c" * 64,
            query_id="d" * 64,
        )

    valid = projected_semantics(
        working_irls_covariance_semantics(np.eye(2)),
        dimension=1,
        source_id="c" * 64,
        query_id="d" * 64,
    )
    assert valid.dimension == 1
    assert valid.metadata["source_covariance_source_id"] == "c" * 64
    assert valid.metadata["query_matrix_sha256"] == "d" * 64
