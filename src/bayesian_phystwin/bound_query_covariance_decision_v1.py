"""Compose a physical-query decision from a lineage-bound Prob4D projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import content_id, require_exact_fields, sha256_digest
from .covariance_only_value import CovarianceOnlyValueCertificateV1
from .physical_query_v1 import PhysicalQueryV1
from .query_covariance_decision_v1 import (
    PROB4D_QUERY_COVARIANCE_SCHEMA,
    QueryCovarianceTreatmentDecisionV1,
    compose_query_covariance_treatment,
)
from .query_jacobian_binding_v1 import QueryJacobianBindingV1

BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA: Final = (
    "prob4d.bound-query-covariance-projection"
)
BOUND_QUERY_COVARIANCE_PROJECTION_VERSION: Final = 1
BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY: Final = (
    "This receipt binds a neutral Prob4D query-covariance projection to the exact "
    "caller-owned Jacobian bytes, ordered observation rows, source observation, and "
    "covariance inputs used for the calculation. It does not define the physical "
    "query, select a covariance treatment, authorize an update, or establish "
    "BayesianPhysTwin or Causal4D benefit."
)

_BOUND_PROJECTION_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "schema",
        "schema_version",
        "query_jacobian_binding_id",
        "source_observation_artifact_id",
        "provider_manifest_id",
        "query_jacobian_sha256",
        "row_ids_sha256",
        "local_covariance_m2",
        "low_rank_factor_m",
        "projection_summary",
        "claim_boundary",
    }
)
_ARRAY_DESCRIPTOR_FIELDS: Final = frozenset({"dtype", "shape", "sha256"})


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with literal string keys")
    return cast(Mapping[str, Any], value)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _array_descriptor(
    value: object,
    *,
    name: str,
    observation_count: int,
    covariance: bool,
) -> Mapping[str, Any]:
    source = _mapping(value, name=name)
    require_exact_fields(source, expected=_ARRAY_DESCRIPTOR_FIELDS, name=name)
    if source["dtype"] != "<f8":
        raise ValueError(f"{name} dtype must be <f8")
    shape_raw = source["shape"]
    if isinstance(shape_raw, (str, bytes)) or not isinstance(shape_raw, Sequence):
        raise ValueError(f"{name} shape must be a JSON array")
    shape = tuple(
        _integer(
            item,
            name=f"{name} shape[{index}]",
            minimum=(0 if not covariance and index == 2 else 1),
        )
        for index, item in enumerate(shape_raw)
    )
    if len(shape) != 3 or shape[:2] != (observation_count, 3):
        raise ValueError(f"{name} has an incompatible observation shape")
    if covariance and shape[2] != 3:
        raise ValueError(f"{name} must have shape [N, 3, 3]")
    digest = sha256_digest(source["sha256"], name=f"{name} sha256")
    return frozen_finite_json_mapping(
        {"dtype": "<f8", "shape": list(shape), "sha256": digest},
        name=name,
    )


@dataclass(frozen=True, slots=True)
class ValidatedBoundQueryCovarianceProjectionV1:
    """Strictly validated Prob4D projection receipt and its compact summary."""

    artifact_id: str
    query_jacobian_binding_id: str
    source_observation_artifact_id: str
    provider_manifest_id: str
    query_jacobian_sha256: str
    row_ids_sha256: str
    local_covariance_descriptor: Mapping[str, Any]
    low_rank_factor_descriptor: Mapping[str, Any]
    projection_summary: Mapping[str, Any]
    record: Mapping[str, Any] = field(repr=False)


def validate_bound_query_covariance_projection(
    value: object,
    *,
    binding: QueryJacobianBindingV1,
) -> ValidatedBoundQueryCovarianceProjectionV1:
    """Validate one Prob4D receipt against its exact BPT Jacobian binding."""

    if not isinstance(binding, QueryJacobianBindingV1):
        raise TypeError("binding must be a QueryJacobianBindingV1")
    source = _mapping(value, name="bound query covariance projection")
    require_exact_fields(
        source,
        expected=_BOUND_PROJECTION_FIELDS,
        name="bound query covariance projection",
    )
    if source["schema"] != BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA:
        raise ValueError("bound query covariance projection schema changed")
    if (
        _integer(
            source["schema_version"],
            name="bound query covariance projection schema_version",
            minimum=1,
        )
        != BOUND_QUERY_COVARIANCE_PROJECTION_VERSION
    ):
        raise ValueError("bound query covariance projection version changed")
    if source["claim_boundary"] != BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY:
        raise ValueError("bound query covariance projection claim boundary changed")

    binding_id = sha256_digest(
        source["query_jacobian_binding_id"],
        name="query_jacobian_binding_id",
    )
    if binding_id != binding.artifact_id:
        raise ValueError("projection binds a different query Jacobian artifact")
    source_observation_id = sha256_digest(
        source["source_observation_artifact_id"],
        name="source_observation_artifact_id",
    )
    if source_observation_id != binding.source_observation_artifact_id:
        raise ValueError("projection binds a different source observation")
    provider_manifest_id = sha256_digest(
        source["provider_manifest_id"],
        name="provider_manifest_id",
    )
    if provider_manifest_id != binding.provider_manifest_id:
        raise ValueError("projection binds a different provider manifest")
    jacobian_sha = sha256_digest(
        source["query_jacobian_sha256"],
        name="query_jacobian_sha256",
    )
    if jacobian_sha != binding.query_jacobian_sha256:
        raise ValueError("projection binds different query Jacobian bytes")
    row_sha = sha256_digest(source["row_ids_sha256"], name="row_ids_sha256")
    if row_sha != binding.row_ids_sha256:
        raise ValueError("projection binds a different observation-row order")

    local = _array_descriptor(
        source["local_covariance_m2"],
        name="local_covariance_m2",
        observation_count=binding.observation_count,
        covariance=True,
    )
    factor = _array_descriptor(
        source["low_rank_factor_m"],
        name="low_rank_factor_m",
        observation_count=binding.observation_count,
        covariance=False,
    )
    summary = _mapping(source["projection_summary"], name="projection_summary")
    if summary.get("schema") != PROB4D_QUERY_COVARIANCE_SCHEMA:
        raise ValueError("projection summary is not a Prob4D query-covariance summary")
    if (
        _integer(summary.get("observation_count"), name="observation_count", minimum=1)
        != binding.observation_count
    ):
        raise ValueError("projection summary observation count differs from binding")
    if (
        _integer(summary.get("query_dimension"), name="query_dimension", minimum=1)
        != binding.query_dimension
    ):
        raise ValueError("projection summary query dimension differs from binding")

    unsigned = dict(source)
    supplied_id = sha256_digest(unsigned.pop("artifact_id"), name="artifact_id")
    if supplied_id != content_id(unsigned):
        raise ValueError("artifact_id does not match bound query projection")
    record = frozen_finite_json_mapping(
        source,
        name="bound query covariance projection",
    )
    return ValidatedBoundQueryCovarianceProjectionV1(
        artifact_id=supplied_id,
        query_jacobian_binding_id=binding_id,
        source_observation_artifact_id=source_observation_id,
        provider_manifest_id=provider_manifest_id,
        query_jacobian_sha256=jacobian_sha,
        row_ids_sha256=row_sha,
        local_covariance_descriptor=local,
        low_rank_factor_descriptor=factor,
        projection_summary=frozen_finite_json_mapping(
            summary,
            name="projection_summary",
        ),
        record=record,
    )


def compose_bound_query_covariance_treatment(
    physical_query: PhysicalQueryV1,
    binding: QueryJacobianBindingV1,
    bound_projection: object,
    value_certificate: CovarianceOnlyValueCertificateV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> QueryCovarianceTreatmentDecisionV1:
    """Compose the existing treatment decision with exact Jacobian lineage."""

    if not isinstance(physical_query, PhysicalQueryV1):
        raise TypeError("physical_query must be a PhysicalQueryV1")
    if not isinstance(binding, QueryJacobianBindingV1):
        raise TypeError("binding must be a QueryJacobianBindingV1")
    if physical_query.jacobian_provider_id != binding.artifact_id:
        raise ValueError("PhysicalQueryV1 does not bind this query Jacobian artifact")
    if physical_query.query_name != binding.query_name:
        raise ValueError("physical query name differs from query Jacobian binding")
    if physical_query.component_order != binding.component_order:
        raise ValueError("physical query components differ from query Jacobian binding")
    if physical_query.physical_unit != binding.physical_unit:
        raise ValueError("physical query unit differs from query Jacobian binding")
    if physical_query.coordinate_frame != binding.coordinate_frame:
        raise ValueError("physical query frame differs from query Jacobian binding")
    if physical_query.provider_manifest_id != binding.provider_manifest_id:
        raise ValueError("physical query provider manifest differs from binding")

    validated = validate_bound_query_covariance_projection(
        bound_projection,
        binding=binding,
    )
    lineage_metadata = {
        **({} if metadata is None else plain_json(metadata)),
        "query_jacobian_binding_id": binding.artifact_id,
        "bound_query_covariance_projection_id": validated.artifact_id,
        "query_jacobian_sha256": binding.query_jacobian_sha256,
        "row_ids_sha256": binding.row_ids_sha256,
    }
    decision = compose_query_covariance_treatment(
        physical_query,
        validated.projection_summary,
        value_certificate,
        source_observation_artifact_id=validated.source_observation_artifact_id,
        metadata=lineage_metadata,
    )
    return replace(
        decision,
        projection_summary_id=validated.artifact_id,
        artifact_id=None,
    )


__all__ = [
    "BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY",
    "BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA",
    "BOUND_QUERY_COVARIANCE_PROJECTION_VERSION",
    "ValidatedBoundQueryCovarianceProjectionV1",
    "compose_bound_query_covariance_treatment",
    "validate_bound_query_covariance_projection",
]
