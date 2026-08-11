from __future__ import annotations

import hashlib

import pytest

from bayesian_phystwin.physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    MARGINAL_GAUGE_COVARIANCE,
)
from bayesian_phystwin.query_covariance_decision_v1 import (
    PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY,
    PROB4D_QUERY_COVARIANCE_SCHEMA,
    PROB4D_QUERY_COVARIANCE_VERSION,
    QueryCovarianceTreatmentDecisionV1,
    _validated_summary,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _decision(**overrides: object) -> QueryCovarianceTreatmentDecisionV1:
    values: dict[str, object] = {
        "physical_query_id": _sha256("query"),
        "source_observation_artifact_id": _sha256("observation"),
        "projection_summary_id": _sha256("projection"),
        "value_certificate_id": _sha256("certificate"),
        "candidate_policy_id": _sha256("candidate"),
        "reference_policy_id": _sha256("reference"),
        "exact_fallback_id": _sha256("fallback"),
        "shared_covariance_relevance": 0.2,
        "relevance_threshold": 0.05,
        "selected_covariance_treatment": (
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
        ),
        "principal_covariance_treatment": (
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
        ),
        "principal_treatment_matches": True,
        "value_certificate_certified": True,
        "authorized": True,
        "reasons": ("covariance-treatment-authorized",),
    }
    values.update(overrides)
    return QueryCovarianceTreatmentDecisionV1(**values)  # type: ignore[arg-type]


def _summary() -> dict[str, object]:
    return {
        "schema": PROB4D_QUERY_COVARIANCE_SCHEMA,
        "version": PROB4D_QUERY_COVARIANCE_VERSION,
        "observation_count": 20,
        "query_dimension": 2,
        "shared_rank_column_count": 1,
        "total_effective_rank": 2,
        "shared_effective_rank": 1,
        "active_query_dimension": 2,
        "conditional_trace": 0.8,
        "shared_trace": 0.2,
        "total_trace": 1.0,
        "shared_trace_fraction": 0.2,
        "shared_frobenius_fraction": 0.2,
        "coordinate_shared_fractions": [0.2, 0.2],
        "minimum_directional_shared_fraction": 0.1,
        "mean_directional_shared_fraction": 0.2,
        "maximum_directional_shared_fraction": 0.3,
        "relative_rank_tolerance": 1e-10,
        "claim_boundary": PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY,
    }


def test_decision_rejects_treatment_inconsistent_with_relevance() -> None:
    with pytest.raises(
        ValueError,
        match="selected covariance treatment contradicts relevance threshold",
    ):
        _decision(shared_covariance_relevance=0.01)


def test_decision_rejects_treatment_when_relevance_is_undefined() -> None:
    with pytest.raises(
        ValueError,
        match="selected covariance treatment contradicts relevance threshold",
    ):
        _decision(
            shared_covariance_relevance=None,
            selected_covariance_treatment=MARGINAL_GAUGE_COVARIANCE,
            principal_covariance_treatment=MARGINAL_GAUGE_COVARIANCE,
            authorized=False,
            reasons=("shared-covariance-relevance-undefined",),
        )


def test_summary_requires_matching_active_and_total_ranks() -> None:
    summary = _summary()
    summary["active_query_dimension"] = 1

    with pytest.raises(
        ValueError,
        match="active_query_dimension must equal total_effective_rank",
    ):
        _validated_summary(summary, expected_query_dimension=2)


def test_summary_rejects_positive_rank_with_zero_trace() -> None:
    summary = _summary()
    summary.update(
        {
            "total_effective_rank": 1,
            "active_query_dimension": 1,
            "shared_effective_rank": 0,
            "conditional_trace": 0.0,
            "shared_trace": 0.0,
            "total_trace": 0.0,
            "shared_trace_fraction": None,
            "shared_frobenius_fraction": None,
            "coordinate_shared_fractions": [None, None],
            "minimum_directional_shared_fraction": 0.0,
            "mean_directional_shared_fraction": 0.0,
            "maximum_directional_shared_fraction": 0.0,
        }
    )

    with pytest.raises(ValueError, match="zero total trace requires zero"):
        _validated_summary(summary, expected_query_dimension=2)


def test_summary_rejects_shared_rank_above_total_rank() -> None:
    summary = _summary()
    summary.update(
        {
            "shared_rank_column_count": 2,
            "total_effective_rank": 1,
            "active_query_dimension": 1,
            "shared_effective_rank": 2,
        }
    )

    with pytest.raises(ValueError, match="shared rank exceeds total effective rank"):
        _validated_summary(summary, expected_query_dimension=2)


def test_summary_rejects_impossible_zero_shared_fractions() -> None:
    summary = _summary()
    summary.update(
        {
            "shared_effective_rank": 0,
            "shared_trace": 0.0,
            "conditional_trace": 1.0,
            "shared_trace_fraction": 0.0,
            "shared_frobenius_fraction": 0.2,
            "coordinate_shared_fractions": [0.0, 0.0],
            "minimum_directional_shared_fraction": 0.0,
            "mean_directional_shared_fraction": 0.0,
            "maximum_directional_shared_fraction": 0.0,
        }
    )

    with pytest.raises(ValueError, match="zero Frobenius fraction"):
        _validated_summary(summary, expected_query_dimension=2)


def test_summary_rejects_defined_coordinate_for_zero_covariance() -> None:
    summary = _summary()
    summary.update(
        {
            "shared_rank_column_count": 0,
            "total_effective_rank": 0,
            "shared_effective_rank": 0,
            "active_query_dimension": 0,
            "conditional_trace": 0.0,
            "shared_trace": 0.0,
            "total_trace": 0.0,
            "shared_trace_fraction": None,
            "shared_frobenius_fraction": None,
            "coordinate_shared_fractions": [0.0, None],
            "minimum_directional_shared_fraction": None,
            "mean_directional_shared_fraction": None,
            "maximum_directional_shared_fraction": None,
        }
    )

    with pytest.raises(ValueError, match="null coordinate fractions"):
        _validated_summary(summary, expected_query_dimension=2)
