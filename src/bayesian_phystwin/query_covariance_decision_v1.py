"""Compose frozen query, covariance-value, and Prob4D relevance artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .covariance_only_value import CovarianceOnlyValueCertificateV1
from .physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    MARGINAL_GAUGE_COVARIANCE,
    PhysicalQueryV1,
)

QUERY_COVARIANCE_DECISION_SCHEMA: Final = (
    "bayesian_phystwin.query_covariance_treatment_decision"
)
QUERY_COVARIANCE_DECISION_VERSION: Final = 1
PROB4D_QUERY_COVARIANCE_SCHEMA: Final = "prob4d.query-covariance-relevance"
PROB4D_QUERY_COVARIANCE_VERSION: Final = 1
PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY: Final = (
    "This diagnostic projects a supplied Prob4D conditional-plus-low-rank "
    "covariance through a caller-supplied query Jacobian. It does not define "
    "the physical query, select a covariance treatment, authorize an update, "
    "or establish BayesianPhysTwin or Causal4D benefit."
)
TRACE_RELEVANCE_DIAGNOSTIC: Final = (
    "query trace fraction attributable to shared gauge covariance"
)
TRACE_RELEVANCE_SELECTION_RULE: Final = (
    "use complete explicit joint gauge unless the frozen relevance diagnostic "
    "is below its source-only threshold"
)
QUERY_COVARIANCE_DECISION_CLAIM_BOUNDARY: Final = (
    "Software composition evidence only. A passing decision does not establish "
    "provider competence, fresh-object benefit, deployment calibration, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)

_SUMMARY_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "observation_count",
        "query_dimension",
        "shared_rank_column_count",
        "total_effective_rank",
        "shared_effective_rank",
        "active_query_dimension",
        "conditional_trace",
        "shared_trace",
        "total_trace",
        "shared_trace_fraction",
        "shared_frobenius_fraction",
        "coordinate_shared_fractions",
        "minimum_directional_shared_fraction",
        "mean_directional_shared_fraction",
        "maximum_directional_shared_fraction",
        "relative_rank_tolerance",
        "claim_boundary",
    }
)
_DECISION_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "schema",
        "schema_version",
        "physical_query_id",
        "source_observation_artifact_id",
        "projection_summary_id",
        "value_certificate_id",
        "candidate_policy_id",
        "reference_policy_id",
        "exact_fallback_id",
        "shared_covariance_relevance",
        "relevance_threshold",
        "selected_covariance_treatment",
        "principal_covariance_treatment",
        "principal_treatment_matches",
        "value_certificate_certified",
        "authorized",
        "reasons",
        "claim_boundary",
        "metadata",
    }
)
_REGISTERED_TREATMENTS: Final = frozenset(
    {
        MARGINAL_GAUGE_COVARIANCE,
        COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _optional_fraction(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_real(value, name=name, minimum=0.0, maximum=1.0)


def _close(left: float, right: float, *, scale: float = 1.0) -> bool:
    return math.isclose(left, right, abs_tol=1e-12 * scale, rel_tol=1e-10)


def _all_close(
    values: Sequence[float | None],
    expected: float,
) -> bool:
    return all(value is None or _close(value, expected) for value in values)


def _selected_treatment(
    relevance: float | None,
    threshold: float,
) -> str | None:
    if relevance is None:
        return None
    if relevance < threshold:
        return MARGINAL_GAUGE_COVARIANCE
    return COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE


def _validated_summary(
    value: object,
    *,
    expected_query_dimension: int,
) -> Mapping[str, Any]:
    summary = _mapping(value, name="Prob4D query covariance summary")
    require_exact_fields(
        summary,
        expected=_SUMMARY_FIELDS,
        name="Prob4D query covariance summary",
    )
    if summary["schema"] != PROB4D_QUERY_COVARIANCE_SCHEMA:
        raise ValueError("Prob4D query covariance schema changed")
    version = genuine_integer(
        summary["version"],
        name="Prob4D query covariance version",
        minimum=1,
    )
    if version != PROB4D_QUERY_COVARIANCE_VERSION:
        raise ValueError("Prob4D query covariance version changed")
    if summary["claim_boundary"] != PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY:
        raise ValueError("Prob4D query covariance claim boundary changed")

    genuine_integer(
        summary["observation_count"],
        name="observation_count",
        minimum=1,
    )
    query_dimension = genuine_integer(
        summary["query_dimension"],
        name="query_dimension",
        minimum=1,
    )
    if query_dimension != expected_query_dimension:
        raise ValueError("Prob4D query dimension differs from PhysicalQueryV1")
    shared_columns = genuine_integer(
        summary["shared_rank_column_count"],
        name="shared_rank_column_count",
        minimum=0,
    )
    total_rank = genuine_integer(
        summary["total_effective_rank"],
        name="total_effective_rank",
        minimum=0,
    )
    shared_rank = genuine_integer(
        summary["shared_effective_rank"],
        name="shared_effective_rank",
        minimum=0,
    )
    active_dimension = genuine_integer(
        summary["active_query_dimension"],
        name="active_query_dimension",
        minimum=0,
    )
    if total_rank > query_dimension or active_dimension > query_dimension:
        raise ValueError("Prob4D query rank exceeds query_dimension")
    if shared_rank > min(query_dimension, shared_columns):
        raise ValueError("Prob4D shared rank exceeds declared dimensions")
    if active_dimension != total_rank:
        raise ValueError("active_query_dimension must equal total_effective_rank")
    if shared_rank > total_rank:
        raise ValueError("Prob4D shared rank exceeds total effective rank")

    conditional = _finite_real(
        summary["conditional_trace"],
        name="conditional_trace",
        minimum=0.0,
    )
    shared = _finite_real(
        summary["shared_trace"],
        name="shared_trace",
        minimum=0.0,
    )
    total = _finite_real(
        summary["total_trace"],
        name="total_trace",
        minimum=0.0,
    )
    scale = max(conditional, shared, total, 1.0)
    if not _close(conditional + shared, total, scale=scale):
        raise ValueError("Prob4D query covariance traces are inconsistent")
    if total == 0.0 and total_rank != 0:
        raise ValueError("zero total trace requires zero total effective rank")
    if total > 0.0 and total_rank == 0:
        raise ValueError("positive total trace requires positive total effective rank")
    if shared == 0.0 and shared_rank != 0:
        raise ValueError("zero shared trace requires zero shared effective rank")
    if shared > 0.0 and shared_rank == 0:
        raise ValueError(
            "positive shared trace requires positive shared effective rank"
        )
    if shared_columns == 0 and shared != 0.0:
        raise ValueError("zero shared-factor columns require zero shared trace")

    relevance = _optional_fraction(
        summary["shared_trace_fraction"],
        name="shared_trace_fraction",
    )
    expected_relevance = None if total == 0.0 else shared / total
    if expected_relevance is None:
        if relevance is not None:
            raise ValueError("shared_trace_fraction must be null for zero trace")
    elif relevance is None or not _close(relevance, expected_relevance):
        raise ValueError("shared_trace_fraction disagrees with query traces")

    frobenius = _optional_fraction(
        summary["shared_frobenius_fraction"],
        name="shared_frobenius_fraction",
    )
    if total == 0.0:
        if frobenius is not None:
            raise ValueError("zero total trace requires null Frobenius fraction")
    elif frobenius is None:
        raise ValueError("positive total trace requires a Frobenius fraction")
    elif shared == 0.0 and not _close(frobenius, 0.0):
        raise ValueError("zero shared covariance requires zero Frobenius fraction")
    elif conditional == 0.0 and not _close(frobenius, 1.0):
        raise ValueError("all-shared covariance requires unit Frobenius fraction")

    coordinates_raw = summary["coordinate_shared_fractions"]
    if isinstance(coordinates_raw, (str, bytes)) or not isinstance(
        coordinates_raw,
        Sequence,
    ):
        raise ValueError("coordinate_shared_fractions must be a JSON array")
    if len(coordinates_raw) != query_dimension:
        raise ValueError("coordinate fractions length must equal query_dimension")
    coordinates = tuple(
        _optional_fraction(
            coordinate,
            name=f"coordinate_shared_fractions[{index}]",
        )
        for index, coordinate in enumerate(coordinates_raw)
    )
    defined_coordinate_count = sum(value is not None for value in coordinates)
    if total == 0.0 and defined_coordinate_count != 0:
        raise ValueError("zero total trace requires null coordinate fractions")
    if total > 0.0 and defined_coordinate_count < total_rank:
        raise ValueError("coordinate fractions are inconsistent with total rank")
    if shared == 0.0 and not _all_close(coordinates, 0.0):
        raise ValueError("zero shared covariance requires zero coordinate fractions")
    if conditional == 0.0 and not _all_close(coordinates, 1.0):
        raise ValueError("all-shared covariance requires unit coordinate fractions")

    directional = tuple(
        _optional_fraction(summary[name], name=name)
        for name in (
            "minimum_directional_shared_fraction",
            "mean_directional_shared_fraction",
            "maximum_directional_shared_fraction",
        )
    )
    if active_dimension == 0:
        if any(item is not None for item in directional):
            raise ValueError("directional fractions require an active query")
    else:
        if any(item is None for item in directional):
            raise ValueError("active queries require directional fractions")
        defined_directional = cast(tuple[float, float, float], directional)
        minimum, mean, maximum = defined_directional
        if not minimum <= mean <= maximum:
            raise ValueError("directional shared fractions are not ordered")
        if shared == 0.0 and not all(_close(item, 0.0) for item in defined_directional):
            raise ValueError(
                "zero shared covariance requires zero directional fractions"
            )
        if conditional == 0.0 and not all(
            _close(item, 1.0) for item in defined_directional
        ):
            raise ValueError(
                "all-shared covariance requires unit directional fractions"
            )

    tolerance = _finite_real(
        summary["relative_rank_tolerance"],
        name="relative_rank_tolerance",
        minimum=0.0,
        maximum=1.0,
    )
    if tolerance >= 1.0:
        raise ValueError("relative_rank_tolerance must be smaller than one")
    return frozen_finite_json_mapping(
        summary,
        name="Prob4D query covariance summary",
    )


@dataclass(frozen=True, slots=True)
class QueryCovarianceTreatmentDecisionV1:
    """Content-addressed composition of existing frozen evidence artifacts."""

    physical_query_id: str
    source_observation_artifact_id: str
    projection_summary_id: str
    value_certificate_id: str
    candidate_policy_id: str
    reference_policy_id: str
    exact_fallback_id: str
    shared_covariance_relevance: float | None
    relevance_threshold: float
    selected_covariance_treatment: str | None
    principal_covariance_treatment: str
    principal_treatment_matches: bool
    value_certificate_certified: bool
    authorized: bool
    reasons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "physical_query_id",
            "source_observation_artifact_id",
            "projection_summary_id",
            "value_certificate_id",
            "candidate_policy_id",
            "reference_policy_id",
            "exact_fallback_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        relevance = _optional_fraction(
            self.shared_covariance_relevance,
            name="shared_covariance_relevance",
        )
        threshold = _finite_real(
            self.relevance_threshold,
            name="relevance_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        selected = self.selected_covariance_treatment
        if selected is not None and selected not in _REGISTERED_TREATMENTS:
            raise ValueError("selected covariance treatment is not registered")
        principal = self.principal_covariance_treatment
        if principal not in _REGISTERED_TREATMENTS:
            raise ValueError("principal covariance treatment is not registered")
        expected_selected = _selected_treatment(relevance, threshold)
        if selected != expected_selected:
            raise ValueError(
                "selected covariance treatment contradicts relevance threshold"
            )

        matches = genuine_boolean(
            self.principal_treatment_matches,
            name="principal_treatment_matches",
        )
        certified = genuine_boolean(
            self.value_certificate_certified,
            name="value_certificate_certified",
        )
        authorized = genuine_boolean(self.authorized, name="authorized")
        expected_matches = selected is not None and selected == principal
        if matches != expected_matches:
            raise ValueError("principal_treatment_matches contradicts treatments")

        reasons = tuple(sorted(self.reasons))
        if len(reasons) != len(set(reasons)):
            raise ValueError("reasons must not contain duplicates")
        if any(type(reason) is not str or not reason for reason in reasons):
            raise ValueError("reasons must contain nonempty strings")
        expected_reasons: list[str] = []
        if relevance is None:
            expected_reasons.append("shared-covariance-relevance-undefined")
        if not matches:
            expected_reasons.append("principal-covariance-treatment-mismatch")
        if not certified:
            expected_reasons.append("covariance-value-certificate-rejected")
        if not expected_reasons:
            expected_reasons.append("covariance-treatment-authorized")
        if reasons != tuple(sorted(expected_reasons)):
            raise ValueError("reasons do not match covariance treatment gates")
        expected_authorized = reasons == ("covariance-treatment-authorized",)
        if authorized != expected_authorized:
            raise ValueError("authorized does not match covariance treatment gates")

        object.__setattr__(self, "shared_covariance_relevance", relevance)
        object.__setattr__(self, "relevance_threshold", threshold)
        object.__setattr__(self, "principal_covariance_treatment", principal)
        object.__setattr__(self, "principal_treatment_matches", matches)
        object.__setattr__(self, "value_certificate_certified", certified)
        object.__setattr__(self, "authorized", authorized)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query covariance decision metadata",
            ),
        )
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match decision content")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_COVARIANCE_DECISION_SCHEMA,
            "schema_version": QUERY_COVARIANCE_DECISION_VERSION,
            "physical_query_id": self.physical_query_id,
            "source_observation_artifact_id": self.source_observation_artifact_id,
            "projection_summary_id": self.projection_summary_id,
            "value_certificate_id": self.value_certificate_id,
            "candidate_policy_id": self.candidate_policy_id,
            "reference_policy_id": self.reference_policy_id,
            "exact_fallback_id": self.exact_fallback_id,
            "shared_covariance_relevance": self.shared_covariance_relevance,
            "relevance_threshold": self.relevance_threshold,
            "selected_covariance_treatment": self.selected_covariance_treatment,
            "principal_covariance_treatment": self.principal_covariance_treatment,
            "principal_treatment_matches": self.principal_treatment_matches,
            "value_certificate_certified": self.value_certificate_certified,
            "authorized": self.authorized,
            "reasons": list(self.reasons),
            "claim_boundary": QUERY_COVARIANCE_DECISION_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> QueryCovarianceTreatmentDecisionV1:
        source = _mapping(value, name="query covariance treatment decision")
        require_exact_fields(
            source,
            expected=_DECISION_FIELDS,
            name="query covariance treatment decision",
        )
        if source["schema"] != QUERY_COVARIANCE_DECISION_SCHEMA:
            raise ValueError("query covariance treatment schema changed")
        version = genuine_integer(
            source["schema_version"],
            name="query covariance treatment schema_version",
            minimum=1,
        )
        if version != QUERY_COVARIANCE_DECISION_VERSION:
            raise ValueError("query covariance treatment version changed")
        if source["claim_boundary"] != QUERY_COVARIANCE_DECISION_CLAIM_BOUNDARY:
            raise ValueError("query covariance treatment claim boundary changed")
        raw_reasons = source["reasons"]
        if isinstance(raw_reasons, (str, bytes)) or not isinstance(
            raw_reasons,
            Sequence,
        ):
            raise ValueError("reasons must be a JSON array")
        return cls(
            physical_query_id=cast(str, source["physical_query_id"]),
            source_observation_artifact_id=cast(
                str,
                source["source_observation_artifact_id"],
            ),
            projection_summary_id=cast(str, source["projection_summary_id"]),
            value_certificate_id=cast(str, source["value_certificate_id"]),
            candidate_policy_id=cast(str, source["candidate_policy_id"]),
            reference_policy_id=cast(str, source["reference_policy_id"]),
            exact_fallback_id=cast(str, source["exact_fallback_id"]),
            shared_covariance_relevance=cast(
                float | None,
                source["shared_covariance_relevance"],
            ),
            relevance_threshold=cast(float, source["relevance_threshold"]),
            selected_covariance_treatment=cast(
                str | None,
                source["selected_covariance_treatment"],
            ),
            principal_covariance_treatment=cast(
                str,
                source["principal_covariance_treatment"],
            ),
            principal_treatment_matches=cast(
                bool,
                source["principal_treatment_matches"],
            ),
            value_certificate_certified=cast(
                bool,
                source["value_certificate_certified"],
            ),
            authorized=cast(bool, source["authorized"]),
            reasons=tuple(cast(Sequence[str], raw_reasons)),
            metadata=_mapping(source["metadata"], name="metadata"),
            artifact_id=cast(str, source["artifact_id"]),
        )


def compose_query_covariance_treatment(
    physical_query: PhysicalQueryV1,
    projection_summary: object,
    value_certificate: CovarianceOnlyValueCertificateV1,
    *,
    source_observation_artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> QueryCovarianceTreatmentDecisionV1:
    """Compose existing artifacts without recomputing statistical evidence."""

    if not isinstance(physical_query, PhysicalQueryV1):
        raise TypeError("physical_query must be a PhysicalQueryV1")
    if not isinstance(value_certificate, CovarianceOnlyValueCertificateV1):
        raise TypeError("value_certificate must be a CovarianceOnlyValueCertificateV1")
    query_id = sha256_digest(physical_query.query_id, name="physical_query.query_id")
    observation_id = sha256_digest(
        source_observation_artifact_id,
        name="source_observation_artifact_id",
    )
    summary = _validated_summary(
        projection_summary,
        expected_query_dimension=physical_query.dimension,
    )
    if value_certificate.query_set_id != query_id:
        raise ValueError("value certificate binds a different physical query")
    if value_certificate.policy_freeze_artifact_id != query_id:
        raise ValueError("value certificate was not frozen by PhysicalQueryV1")
    if value_certificate.statistical_unit != (
        physical_query.bootstrap.independent_group_definition
    ):
        raise ValueError("value certificate statistical unit differs from query")
    if value_certificate.score_metric != physical_query.primary_proper_score:
        raise ValueError("value certificate proper score differs from query")

    margins = physical_query.decision_margins
    alignments = (
        (
            value_certificate.maximum_expected_score_regret,
            margins.practical_equivalence_score,
            "score",
        ),
        (
            value_certificate.harm_margin,
            margins.maximum_harmful_score_increase,
            "harm",
        ),
        (
            value_certificate.maximum_expected_full_width,
            margins.maximum_mean_width,
            "width",
        ),
        (
            value_certificate.familywise_confidence_level,
            physical_query.bootstrap.confidence_level,
            "confidence",
        ),
    )
    for observed, expected, label in alignments:
        if not _close(observed, expected):
            raise ValueError(f"value certificate {label} threshold differs from query")
    if physical_query.shared_covariance_diagnostic != TRACE_RELEVANCE_DIAGNOSTIC:
        raise ValueError("PhysicalQueryV1 uses an unsupported relevance diagnostic")
    if physical_query.computational_selection_rule != TRACE_RELEVANCE_SELECTION_RULE:
        raise ValueError("PhysicalQueryV1 uses an unsupported selection rule")

    relevance = cast(float | None, summary["shared_trace_fraction"])
    selected = _selected_treatment(
        relevance,
        margins.minimum_shared_covariance_relevance,
    )
    matches = bool(
        selected is not None
        and selected == physical_query.principal_covariance_treatment
    )
    reasons: list[str] = []
    if relevance is None:
        reasons.append("shared-covariance-relevance-undefined")
    if not matches:
        reasons.append("principal-covariance-treatment-mismatch")
    if not value_certificate.certified:
        reasons.append("covariance-value-certificate-rejected")
    if not reasons:
        reasons.append("covariance-treatment-authorized")
    projection_id = content_id(
        {
            "physical_query_id": query_id,
            "source_observation_artifact_id": observation_id,
            "jacobian_provider_id": physical_query.jacobian_provider_id,
            "provider_manifest_id": physical_query.provider_manifest_id,
            "prob4d_summary": plain_json(summary),
        }
    )
    return QueryCovarianceTreatmentDecisionV1(
        physical_query_id=query_id,
        source_observation_artifact_id=observation_id,
        projection_summary_id=projection_id,
        value_certificate_id=sha256_digest(
            value_certificate.artifact_id,
            name="value_certificate.artifact_id",
        ),
        candidate_policy_id=value_certificate.candidate_policy_id,
        reference_policy_id=value_certificate.reference_policy_id,
        exact_fallback_id=physical_query.exact_fallback_id,
        shared_covariance_relevance=relevance,
        relevance_threshold=margins.minimum_shared_covariance_relevance,
        selected_covariance_treatment=selected,
        principal_covariance_treatment=(physical_query.principal_covariance_treatment),
        principal_treatment_matches=matches,
        value_certificate_certified=value_certificate.certified,
        authorized=reasons == ["covariance-treatment-authorized"],
        reasons=tuple(reasons),
        metadata={} if metadata is None else metadata,
    )


def load_query_covariance_treatment_decision(
    path: str | Path,
) -> QueryCovarianceTreatmentDecisionV1:
    return QueryCovarianceTreatmentDecisionV1.from_mapping(
        load_strict_json_object(path, label="query covariance treatment decision")
    )


def write_query_covariance_treatment_decision(
    decision: QueryCovarianceTreatmentDecisionV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(decision, QueryCovarianceTreatmentDecisionV1):
        raise TypeError("decision must be a QueryCovarianceTreatmentDecisionV1")
    write_atomic_json(decision.to_record(), path, overwrite=overwrite)


__all__ = [
    "PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY",
    "PROB4D_QUERY_COVARIANCE_SCHEMA",
    "PROB4D_QUERY_COVARIANCE_VERSION",
    "QUERY_COVARIANCE_DECISION_CLAIM_BOUNDARY",
    "QUERY_COVARIANCE_DECISION_SCHEMA",
    "QUERY_COVARIANCE_DECISION_VERSION",
    "QueryCovarianceTreatmentDecisionV1",
    "compose_query_covariance_treatment",
    "load_query_covariance_treatment_decision",
    "write_query_covariance_treatment_decision",
]
