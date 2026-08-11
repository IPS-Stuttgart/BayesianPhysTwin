from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import pytest

import bayesian_phystwin.query_covariance_decision_v1 as decision
from bayesian_phystwin.physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _summary() -> dict[str, object]:
    return {
        "schema": decision.PROB4D_QUERY_COVARIANCE_SCHEMA,
        "version": decision.PROB4D_QUERY_COVARIANCE_VERSION,
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
        "claim_boundary": decision.PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY,
    }


def _decision_kwargs() -> dict[str, Any]:
    return {
        "physical_query_id": _sha256("query"),
        "source_observation_artifact_id": _sha256("observation"),
        "projection_summary_id": _sha256("projection"),
        "value_certificate_id": _sha256("certificate"),
        "candidate_policy_id": _sha256("candidate"),
        "reference_policy_id": _sha256("reference"),
        "exact_fallback_id": _sha256("fallback"),
        "shared_covariance_relevance": 0.2,
        "relevance_threshold": 0.05,
        "selected_covariance_treatment": (COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE),
        "principal_covariance_treatment": (COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE),
        "principal_treatment_matches": True,
        "value_certificate_certified": True,
        "authorized": True,
        "reasons": ("covariance-treatment-authorized",),
        "metadata": {},
    }


def _valid_decision() -> decision.QueryCovarianceTreatmentDecisionV1:
    return decision.QueryCovarianceTreatmentDecisionV1(**_decision_kwargs())


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be a JSON object"),
        ({1: "value"}, "must use literal string keys"),
    ],
)
def test_mapping_rejects_noncanonical_objects(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decision._mapping(value, name="value")


@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "message"),
    [
        (True, None, None, "finite real number"),
        (float("inf"), None, None, "finite real number"),
        (-1.0, 0.0, None, "at least"),
        (2.0, None, 1.0, "at most"),
    ],
)
def test_finite_real_rejects_invalid_values(
    value: object,
    minimum: float | None,
    maximum: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decision._finite_real(
            value,
            name="value",
            minimum=minimum,
            maximum=maximum,
        )


def _set_values(**updates: object) -> Callable[[dict[str, object]], None]:
    def mutate(summary: dict[str, object]) -> None:
        summary.update(updates)

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set_values(schema="changed"), "schema changed"),
        (_set_values(version=2), "version changed"),
        (_set_values(claim_boundary="changed"), "claim boundary changed"),
        (_set_values(query_dimension=3), "dimension differs"),
        (_set_values(total_effective_rank=3), "rank exceeds"),
        (
            _set_values(shared_effective_rank=2),
            "shared rank exceeds declared dimensions",
        ),
        (_set_values(total_trace=2.0), "traces are inconsistent"),
        (
            _set_values(
                conditional_trace=0.0,
                shared_trace=0.0,
                total_trace=0.0,
                shared_trace_fraction=0.1,
            ),
            "must be null for zero trace",
        ),
        (
            _set_values(coordinate_shared_fractions="invalid"),
            "must be a JSON array",
        ),
        (
            _set_values(coordinate_shared_fractions=[0.2]),
            "length must equal query_dimension",
        ),
        (
            _set_values(active_query_dimension=0),
            "directional fractions require an active query",
        ),
        (
            _set_values(minimum_directional_shared_fraction=None),
            "active queries require directional fractions",
        ),
        (
            _set_values(
                minimum_directional_shared_fraction=0.4,
                mean_directional_shared_fraction=0.2,
                maximum_directional_shared_fraction=0.3,
            ),
            "directional shared fractions are not ordered",
        ),
        (
            _set_values(relative_rank_tolerance=1.0),
            "must be smaller than one",
        ),
    ],
)
def test_projection_summary_rejects_inconsistent_contracts(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    summary = _summary()
    mutate(summary)

    with pytest.raises(ValueError, match=message):
        decision._validated_summary(summary, expected_query_dimension=2)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"selected_covariance_treatment": "not-registered"},
            "selected covariance treatment is not registered",
        ),
        (
            {"principal_covariance_treatment": "not-registered"},
            "principal covariance treatment is not registered",
        ),
        (
            {"principal_treatment_matches": False},
            "principal_treatment_matches contradicts treatments",
        ),
        (
            {
                "reasons": (
                    "covariance-treatment-authorized",
                    "covariance-treatment-authorized",
                )
            },
            "reasons must not contain duplicates",
        ),
        (
            {"reasons": ("",)},
            "reasons must contain nonempty strings",
        ),
        (
            {"reasons": ("principal-covariance-treatment-mismatch",)},
            "reasons do not match covariance treatment gates",
        ),
        (
            {"artifact_id": _sha256("wrong-artifact")},
            "artifact_id does not match decision content",
        ),
    ],
)
def test_decision_rejects_internally_inconsistent_records(
    updates: dict[str, object],
    message: str,
) -> None:
    kwargs = _decision_kwargs()
    kwargs.update(updates)

    with pytest.raises(ValueError, match=message):
        decision.QueryCovarianceTreatmentDecisionV1(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("claim_boundary", "changed", "claim boundary changed"),
        ("reasons", "not-an-array", "reasons must be a JSON array"),
    ],
)
def test_from_mapping_rejects_changed_wire_contracts(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _valid_decision().to_record()
    record[field] = value

    with pytest.raises(ValueError, match=message):
        decision.QueryCovarianceTreatmentDecisionV1.from_mapping(record)


def test_write_rejects_nondecision_objects(tmp_path: Any) -> None:
    with pytest.raises(TypeError, match="must be a QueryCovarianceTreatmentDecisionV1"):
        decision.write_query_covariance_treatment_decision(
            object(),  # type: ignore[arg-type]
            tmp_path / "decision.json",
        )
