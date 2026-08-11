"""Contracts for query-space relevance of shared covariance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from .._portable_contracts import content_id, sha256_digest
from ._common import canonical_string, finite_float

QUERY_COVARIANCE_RELEVANCE_SCHEMA: Final = (
    "bayesian_phystwin.query_covariance_relevance"
)
QUERY_COVARIANCE_RELEVANCE_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class QueryCovarianceRelevancePolicyV1:
    """Frozen rule for deciding whether shared covariance matters in a query."""

    minimum_shared_trace_fraction: float
    minimum_maximum_generalized_eigenvalue: float
    minimum_effective_rank: int = 1
    maximum_null_mode_fraction: float = 1.0
    rank_relative_tolerance: float = 1e-10
    mode_response_relative_tolerance: float = 1e-10
    covariance_jitter: float = 1e-12
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "minimum_shared_trace_fraction",
            "maximum_null_mode_fraction",
        ):
            object.__setattr__(
                self,
                name,
                finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        for name in (
            "minimum_maximum_generalized_eigenvalue",
            "rank_relative_tolerance",
            "mode_response_relative_tolerance",
            "covariance_jitter",
            "numerical_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                finite_float(getattr(self, name), name=name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "minimum_effective_rank",
            genuine_integer(
                self.minimum_effective_rank,
                name="minimum_effective_rank",
                minimum=0,
            ),
        )
        if self.covariance_jitter <= 0.0:
            raise ValueError("covariance_jitter must be positive")

    def descriptor(self) -> dict[str, object]:
        return {
            "minimum_shared_trace_fraction": self.minimum_shared_trace_fraction,
            "minimum_maximum_generalized_eigenvalue": (
                self.minimum_maximum_generalized_eigenvalue
            ),
            "minimum_effective_rank": self.minimum_effective_rank,
            "maximum_null_mode_fraction": self.maximum_null_mode_fraction,
            "rank_relative_tolerance": self.rank_relative_tolerance,
            "mode_response_relative_tolerance": (
                self.mode_response_relative_tolerance
            ),
            "covariance_jitter": self.covariance_jitter,
            "numerical_tolerance": self.numerical_tolerance,
        }


def shared_covariance_material(
    *,
    shared_trace_fraction: float,
    effective_query_rank: int,
    null_mode_fraction: float,
    maximum_generalized_eigenvalue: float,
    policy: QueryCovarianceRelevancePolicyV1,
) -> bool:
    trace_supported = shared_trace_fraction + policy.numerical_tolerance >= (
        policy.minimum_shared_trace_fraction
    )
    eigen_supported = maximum_generalized_eigenvalue + policy.numerical_tolerance >= (
        policy.minimum_maximum_generalized_eigenvalue
    )
    return bool(
        effective_query_rank >= policy.minimum_effective_rank
        and null_mode_fraction
        <= policy.maximum_null_mode_fraction + policy.numerical_tolerance
        and (trace_supported or eigen_supported)
    )


def relevance_reasons(
    *,
    shared_trace_fraction: float,
    effective_query_rank: int,
    null_mode_fraction: float,
    maximum_generalized_eigenvalue: float,
    policy: QueryCovarianceRelevancePolicyV1,
    frozen_before_target_outcomes: bool,
    target_outcomes_used_for_selection: bool,
    calibration_groups_independent: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not frozen_before_target_outcomes:
        reasons.append("query-rule-not-frozen-before-target")
    if target_outcomes_used_for_selection:
        reasons.append("target-outcomes-used-for-query-rule")
    if not calibration_groups_independent:
        reasons.append("calibration-groups-not-independent")
    if effective_query_rank < policy.minimum_effective_rank:
        reasons.append("effective-query-rank-below-threshold")
    if null_mode_fraction > (
        policy.maximum_null_mode_fraction + policy.numerical_tolerance
    ):
        reasons.append("query-null-mode-fraction-exceeds-limit")
    trace_supported = shared_trace_fraction + policy.numerical_tolerance >= (
        policy.minimum_shared_trace_fraction
    )
    eigen_supported = maximum_generalized_eigenvalue + policy.numerical_tolerance >= (
        policy.minimum_maximum_generalized_eigenvalue
    )
    if not (trace_supported or eigen_supported):
        reasons.append("shared-query-covariance-below-materiality-threshold")
    return tuple(reasons or ["shared-query-covariance-material"])


@dataclass(frozen=True, slots=True)
class QueryCovarianceRelevanceCertificateV1:
    """Content-addressed source/calibration-only query-space diagnostic."""

    query_id: str
    covariance_artifact_id: str
    jacobian_artifact_id: str
    calibration_partition_id: str
    statistical_unit: str
    state_dimension: int
    query_dimension: int
    shared_rank: int
    local_covariance_sha256: str
    shared_factor_sha256: str
    query_jacobian_sha256: str
    query_noise_covariance_sha256: str
    shared_trace_fraction: float
    effective_query_rank: int
    null_mode_fraction: float
    maximum_generalized_eigenvalue: float
    shared_covariance_material: bool
    reasons: Sequence[str]
    policy: QueryCovarianceRelevancePolicyV1
    frozen_before_target_outcomes: bool
    target_outcomes_used_for_selection: bool
    calibration_groups_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "query_id",
            "covariance_artifact_id",
            "jacobian_artifact_id",
            "calibration_partition_id",
            "local_covariance_sha256",
            "shared_factor_sha256",
            "query_jacobian_sha256",
            "query_noise_covariance_sha256",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "statistical_unit",
            canonical_string(self.statistical_unit, name="statistical_unit"),
        )
        for name in (
            "state_dimension",
            "query_dimension",
            "shared_rank",
            "effective_query_rank",
        ):
            object.__setattr__(
                self,
                name,
                genuine_integer(getattr(self, name), name=name, minimum=0),
            )
        if self.state_dimension < 1 or self.query_dimension < 1:
            raise ValueError("state and query dimensions must be positive")
        if self.effective_query_rank > min(self.query_dimension, self.shared_rank):
            raise ValueError("effective_query_rank exceeds projected rank")
        for name in ("shared_trace_fraction", "null_mode_fraction"):
            object.__setattr__(
                self,
                name,
                finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "maximum_generalized_eigenvalue",
            finite_float(
                self.maximum_generalized_eigenvalue,
                name="maximum_generalized_eigenvalue",
                minimum=0.0,
            ),
        )
        for name in (
            "shared_covariance_material",
            "frozen_before_target_outcomes",
            "target_outcomes_used_for_selection",
            "calibration_groups_independent",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        if not isinstance(self.policy, QueryCovarianceRelevancePolicyV1):
            raise TypeError("policy must be QueryCovarianceRelevancePolicyV1")
        expected_reasons = relevance_reasons(
            shared_trace_fraction=self.shared_trace_fraction,
            effective_query_rank=self.effective_query_rank,
            null_mode_fraction=self.null_mode_fraction,
            maximum_generalized_eigenvalue=(
                self.maximum_generalized_eigenvalue
            ),
            policy=self.policy,
            frozen_before_target_outcomes=self.frozen_before_target_outcomes,
            target_outcomes_used_for_selection=(
                self.target_outcomes_used_for_selection
            ),
            calibration_groups_independent=self.calibration_groups_independent,
        )
        supplied_reasons = tuple(
            sorted(
                {
                    canonical_string(item, name=f"reasons[{index}]")
                    for index, item in enumerate(tuple(self.reasons))
                }
            )
        )
        if supplied_reasons != tuple(sorted(expected_reasons)):
            raise ValueError("reasons do not match query covariance gates")
        expected_material = shared_covariance_material(
            shared_trace_fraction=self.shared_trace_fraction,
            effective_query_rank=self.effective_query_rank,
            null_mode_fraction=self.null_mode_fraction,
            maximum_generalized_eigenvalue=(
                self.maximum_generalized_eigenvalue
            ),
            policy=self.policy,
        )
        if self.shared_covariance_material != expected_material:
            raise ValueError(
                "shared_covariance_material does not match relevance gates"
            )
        object.__setattr__(self, "reasons", tuple(sorted(expected_reasons)))
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query covariance relevance metadata",
            ),
        )
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match relevance certificate")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def deployment_admissible(self) -> bool:
        return (
            self.frozen_before_target_outcomes
            and not self.target_outcomes_used_for_selection
            and self.calibration_groups_independent
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_COVARIANCE_RELEVANCE_SCHEMA,
            "schema_version": QUERY_COVARIANCE_RELEVANCE_VERSION,
            "query_id": self.query_id,
            "covariance_artifact_id": self.covariance_artifact_id,
            "jacobian_artifact_id": self.jacobian_artifact_id,
            "calibration_partition_id": self.calibration_partition_id,
            "statistical_unit": self.statistical_unit,
            "state_dimension": self.state_dimension,
            "query_dimension": self.query_dimension,
            "shared_rank": self.shared_rank,
            "local_covariance_sha256": self.local_covariance_sha256,
            "shared_factor_sha256": self.shared_factor_sha256,
            "query_jacobian_sha256": self.query_jacobian_sha256,
            "query_noise_covariance_sha256": (
                self.query_noise_covariance_sha256
            ),
            "shared_trace_fraction": self.shared_trace_fraction,
            "effective_query_rank": self.effective_query_rank,
            "null_mode_fraction": self.null_mode_fraction,
            "maximum_generalized_eigenvalue": (
                self.maximum_generalized_eigenvalue
            ),
            "shared_covariance_material": self.shared_covariance_material,
            "reasons": list(self.reasons),
            "policy": self.policy.descriptor(),
            "frozen_before_target_outcomes": self.frozen_before_target_outcomes,
            "target_outcomes_used_for_selection": (
                self.target_outcomes_used_for_selection
            ),
            "calibration_groups_independent": self.calibration_groups_independent,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
