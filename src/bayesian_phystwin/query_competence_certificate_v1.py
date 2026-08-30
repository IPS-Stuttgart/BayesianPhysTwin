"""Query-scoped simulator competence certificates with exact fallback.

Backend-wide competence is too coarse for deformable manipulation: one frozen
policy can transfer on one task and fail on another while using the same public
simulator.  This module binds evidence to an exact query scope and admits a
candidate complete belief only for a matching, certified scope and policy.
Unknown, failed, or mismatched queries reuse the registered baseline object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast

import numpy as np

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
from .complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    select_complete_belief,
)
from .guard_harm_risk import one_sided_binomial_upper_bound

QUERY_SCOPE_SCHEMA = "bayesian_phystwin.simulator_query_scope"
QUERY_SCOPE_VERSION = 1
QUERY_COMPETENCE_GATE_SCHEMA = "bayesian_phystwin.query_competence_gate"
QUERY_COMPETENCE_GATE_VERSION = 1
QUERY_COMPETENCE_CERTIFICATE_SCHEMA = (
    "bayesian_phystwin.query_competence_certificate"
)
QUERY_COMPETENCE_CERTIFICATE_VERSION = 1
QUERY_COMPETENCE_REGISTRY_SCHEMA = "bayesian_phystwin.query_competence_registry"
QUERY_COMPETENCE_REGISTRY_VERSION = 1

BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _open_probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def _fraction(value: object, *, name: str) -> float:
    return _finite_real(value, name=name, minimum=0.0, maximum=1.0)


def _interval(value: object, *, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain exactly two finite values")
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two finite values")
    lower = _finite_real(values[0], name=f"{name}[0]")
    upper = _finite_real(values[1], name=f"{name}[1]")
    if lower > upper:
        raise ValueError(f"{name} lower endpoint cannot exceed upper endpoint")
    return lower, upper


def _canonical_reasons(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of canonical strings")
    reasons = tuple(
        _canonical_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(set(reasons)) != len(reasons):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class SimulatorQueryScopeV1:
    """Content-addressed scope of one simulator decision query."""

    simulator_id: str
    task_id: str
    observation_policy_id: str
    action_bank_id: str
    metric_id: str
    world_distribution_id: str
    statistical_unit: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    query_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "simulator_id",
            "task_id",
            "observation_policy_id",
            "action_bank_id",
            "metric_id",
            "world_distribution_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "statistical_unit",
            _canonical_string(self.statistical_unit, name="statistical_unit"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="query scope metadata"),
        )
        expected_id = content_id(self.descriptor())
        supplied_id = self.query_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="query_id")
            if supplied_id != expected_id:
                raise ValueError("query_id does not match query scope content")
        object.__setattr__(self, "query_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_SCOPE_SCHEMA,
            "schema_version": QUERY_SCOPE_VERSION,
            "simulator_id": self.simulator_id,
            "task_id": self.task_id,
            "observation_policy_id": self.observation_policy_id,
            "action_bank_id": self.action_bank_id,
            "metric_id": self.metric_id,
            "world_distribution_id": self.world_distribution_id,
            "statistical_unit": self.statistical_unit,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "query_id": self.query_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query scope",
    ) -> SimulatorQueryScopeV1:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "simulator_id",
                    "task_id",
                    "observation_policy_id",
                    "action_bank_id",
                    "metric_id",
                    "world_distribution_id",
                    "statistical_unit",
                    "metadata",
                    "query_id",
                }
            ),
            name=name,
        )
        if value["schema"] != QUERY_SCOPE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_SCOPE_VERSION:
            raise ValueError(f"{name} schema version changed")
        return cls(
            simulator_id=cast(str, value["simulator_id"]),
            task_id=cast(str, value["task_id"]),
            observation_policy_id=cast(str, value["observation_policy_id"]),
            action_bank_id=cast(str, value["action_bank_id"]),
            metric_id=cast(str, value["metric_id"]),
            world_distribution_id=cast(str, value["world_distribution_id"]),
            statistical_unit=cast(str, value["statistical_unit"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            query_id=cast(str, value["query_id"]),
        )


@dataclass(frozen=True, slots=True)
class QueryCompetenceGateV1:
    """Frozen value, risk, support, and custody requirements for one query."""

    expected_group_count: int
    minimum_mean_gain: float
    require_positive_paired_lower_bound: bool
    maximum_harm_risk_upper: float
    minimum_downside_reduction_fraction: float
    minimum_retained_candidate_gain_fraction: float
    minimum_oracle_headroom_fraction: float
    maximum_technical_failures: int = 0
    maximum_retries: int = 0
    maximum_replacements: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    gate_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_group_count",
            genuine_integer(
                self.expected_group_count,
                name="expected_group_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "minimum_mean_gain",
            _finite_real(self.minimum_mean_gain, name="minimum_mean_gain"),
        )
        object.__setattr__(
            self,
            "require_positive_paired_lower_bound",
            genuine_boolean(
                self.require_positive_paired_lower_bound,
                name="require_positive_paired_lower_bound",
            ),
        )
        for name in (
            "maximum_harm_risk_upper",
            "minimum_downside_reduction_fraction",
            "minimum_retained_candidate_gain_fraction",
            "minimum_oracle_headroom_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _fraction(getattr(self, name), name=name),
            )
        for name in (
            "maximum_technical_failures",
            "maximum_retries",
            "maximum_replacements",
        ):
            object.__setattr__(
                self,
                name,
                genuine_integer(getattr(self, name), name=name, minimum=0),
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="competence gate metadata"),
        )
        expected_id = content_id(self.descriptor())
        supplied_id = self.gate_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="gate_id")
            if supplied_id != expected_id:
                raise ValueError("gate_id does not match competence gate content")
        object.__setattr__(self, "gate_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_COMPETENCE_GATE_SCHEMA,
            "schema_version": QUERY_COMPETENCE_GATE_VERSION,
            "expected_group_count": self.expected_group_count,
            "minimum_mean_gain": self.minimum_mean_gain,
            "require_positive_paired_lower_bound": (
                self.require_positive_paired_lower_bound
            ),
            "maximum_harm_risk_upper": self.maximum_harm_risk_upper,
            "minimum_downside_reduction_fraction": (
                self.minimum_downside_reduction_fraction
            ),
            "minimum_retained_candidate_gain_fraction": (
                self.minimum_retained_candidate_gain_fraction
            ),
            "minimum_oracle_headroom_fraction": (
                self.minimum_oracle_headroom_fraction
            ),
            "maximum_technical_failures": self.maximum_technical_failures,
            "maximum_retries": self.maximum_retries,
            "maximum_replacements": self.maximum_replacements,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "gate_id": self.gate_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query competence gate",
    ) -> QueryCompetenceGateV1:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "expected_group_count",
                    "minimum_mean_gain",
                    "require_positive_paired_lower_bound",
                    "maximum_harm_risk_upper",
                    "minimum_downside_reduction_fraction",
                    "minimum_retained_candidate_gain_fraction",
                    "minimum_oracle_headroom_fraction",
                    "maximum_technical_failures",
                    "maximum_retries",
                    "maximum_replacements",
                    "metadata",
                    "gate_id",
                }
            ),
            name=name,
        )
        if value["schema"] != QUERY_COMPETENCE_GATE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_COMPETENCE_GATE_VERSION:
            raise ValueError(f"{name} schema version changed")
        return cls(
            expected_group_count=cast(int, value["expected_group_count"]),
            minimum_mean_gain=cast(float, value["minimum_mean_gain"]),
            require_positive_paired_lower_bound=cast(
                bool, value["require_positive_paired_lower_bound"]
            ),
            maximum_harm_risk_upper=cast(
                float, value["maximum_harm_risk_upper"]
            ),
            minimum_downside_reduction_fraction=cast(
                float, value["minimum_downside_reduction_fraction"]
            ),
            minimum_retained_candidate_gain_fraction=cast(
                float, value["minimum_retained_candidate_gain_fraction"]
            ),
            minimum_oracle_headroom_fraction=cast(
                float, value["minimum_oracle_headroom_fraction"]
            ),
            maximum_technical_failures=cast(
                int, value["maximum_technical_failures"]
            ),
            maximum_retries=cast(int, value["maximum_retries"]),
            maximum_replacements=cast(int, value["maximum_replacements"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            gate_id=cast(str, value["gate_id"]),
        )


def _failed_competence_checks(
    *,
    gate: QueryCompetenceGateV1,
    group_count: int,
    technical_failures: int,
    retries: int,
    replacements: int,
    mean_gain: float,
    paired_gain_ci95: tuple[float, float],
    harm_risk_upper: float,
    downside_reduction_fraction: float,
    retained_candidate_gain_fraction: float,
    oracle_headroom_fraction: float,
    protocol_frozen_before_outcomes: bool,
    outcomes_used_for_policy_or_gate_selection: bool,
    independent_implementation_replay: bool,
    source_gate_passed: bool,
) -> tuple[str, ...]:
    failed: list[str] = []
    if group_count != gate.expected_group_count:
        failed.append("incomplete-group-denominator")
    if technical_failures > gate.maximum_technical_failures:
        failed.append("technical-failure-budget-exceeded")
    if retries > gate.maximum_retries:
        failed.append("retry-budget-exceeded")
    if replacements > gate.maximum_replacements:
        failed.append("replacement-budget-exceeded")
    if mean_gain < gate.minimum_mean_gain:
        failed.append("mean-gain-below-threshold")
    if gate.require_positive_paired_lower_bound and paired_gain_ci95[0] <= 0.0:
        failed.append("paired-gain-lower-bound-not-positive")
    if harm_risk_upper > gate.maximum_harm_risk_upper:
        failed.append("harm-risk-upper-bound-exceeded")
    if downside_reduction_fraction < gate.minimum_downside_reduction_fraction:
        failed.append("downside-reduction-below-threshold")
    if retained_candidate_gain_fraction < (
        gate.minimum_retained_candidate_gain_fraction
    ):
        failed.append("retained-value-below-threshold")
    if oracle_headroom_fraction < gate.minimum_oracle_headroom_fraction:
        failed.append("oracle-headroom-below-threshold")
    if not protocol_frozen_before_outcomes:
        failed.append("protocol-not-frozen-before-outcomes")
    if outcomes_used_for_policy_or_gate_selection:
        failed.append("outcomes-used-for-selection")
    if not independent_implementation_replay:
        failed.append("evidence-replay-not-verified")
    if not source_gate_passed:
        failed.append("registered-source-gate-rejected")
    return tuple(sorted(failed))


@dataclass(frozen=True, slots=True)
class QueryCompetenceCertificateV1:
    """Evidence-bound competence decision for one exact query and policy."""

    query_scope: SimulatorQueryScopeV1
    gate: QueryCompetenceGateV1
    candidate_policy_id: str
    baseline_policy_id: str
    protocol_id: str
    source_summary_artifact_id: str
    source_summary_sha256: str
    source_result_id: str
    verification_artifact_id: str
    verification_file_sha256: str
    verified_tree_id: str
    group_count: int
    technical_failures: int
    retries: int
    replacements: int
    mean_gain: float
    paired_gain_ci95: Sequence[float]
    harmful_group_count: int
    harm_confidence_level: float
    harm_risk_upper: float
    downside_reduction_fraction: float
    retained_candidate_gain_fraction: float
    oracle_headroom_fraction: float
    protocol_frozen_before_outcomes: bool
    outcomes_used_for_policy_or_gate_selection: bool
    independent_implementation_replay: bool
    source_gate_passed: bool
    failed_checks: Sequence[str]
    certified: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_scope, SimulatorQueryScopeV1):
            raise TypeError("query_scope must be a SimulatorQueryScopeV1")
        if self.query_scope.query_id is None:
            raise ValueError("query scope must have a content identity")
        if not isinstance(self.gate, QueryCompetenceGateV1):
            raise TypeError("gate must be a QueryCompetenceGateV1")
        for name in (
            "candidate_policy_id",
            "baseline_policy_id",
            "protocol_id",
            "source_summary_artifact_id",
            "source_summary_sha256",
            "source_result_id",
            "verification_artifact_id",
            "verification_file_sha256",
            "verified_tree_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in ("group_count", "technical_failures", "retries", "replacements"):
            object.__setattr__(
                self,
                name,
                genuine_integer(getattr(self, name), name=name, minimum=0),
            )
        harmful_group_count = genuine_integer(
            self.harmful_group_count,
            name="harmful_group_count",
            minimum=0,
        )
        if harmful_group_count > self.group_count:
            raise ValueError("harmful_group_count cannot exceed group_count")
        object.__setattr__(
            self,
            "harmful_group_count",
            harmful_group_count,
        )
        mean_gain = _finite_real(self.mean_gain, name="mean_gain")
        interval = _interval(self.paired_gain_ci95, name="paired_gain_ci95")
        confidence = _open_probability(
            self.harm_confidence_level,
            name="harm_confidence_level",
        )
        expected_harm_upper = one_sided_binomial_upper_bound(
            self.harmful_group_count,
            self.group_count,
            confidence,
        )
        supplied_harm_upper = _fraction(
            self.harm_risk_upper,
            name="harm_risk_upper",
        )
        if not math.isclose(
            supplied_harm_upper,
            expected_harm_upper,
            rel_tol=1e-12,
            abs_tol=1e-14,
        ):
            raise ValueError(
                "harm_risk_upper does not match exact Clopper-Pearson inversion"
            )
        downside = _fraction(
            self.downside_reduction_fraction,
            name="downside_reduction_fraction",
        )
        retained = _fraction(
            self.retained_candidate_gain_fraction,
            name="retained_candidate_gain_fraction",
        )
        oracle = _fraction(
            self.oracle_headroom_fraction,
            name="oracle_headroom_fraction",
        )
        protocol_frozen = genuine_boolean(
            self.protocol_frozen_before_outcomes,
            name="protocol_frozen_before_outcomes",
        )
        outcomes_used = genuine_boolean(
            self.outcomes_used_for_policy_or_gate_selection,
            name="outcomes_used_for_policy_or_gate_selection",
        )
        replayed = genuine_boolean(
            self.independent_implementation_replay,
            name="independent_implementation_replay",
        )
        source_passed = genuine_boolean(
            self.source_gate_passed,
            name="source_gate_passed",
        )
        expected_failed = _failed_competence_checks(
            gate=self.gate,
            group_count=self.group_count,
            technical_failures=self.technical_failures,
            retries=self.retries,
            replacements=self.replacements,
            mean_gain=mean_gain,
            paired_gain_ci95=interval,
            harm_risk_upper=expected_harm_upper,
            downside_reduction_fraction=downside,
            retained_candidate_gain_fraction=retained,
            oracle_headroom_fraction=oracle,
            protocol_frozen_before_outcomes=protocol_frozen,
            outcomes_used_for_policy_or_gate_selection=outcomes_used,
            independent_implementation_replay=replayed,
            source_gate_passed=source_passed,
        )
        failed = _canonical_reasons(self.failed_checks, name="failed_checks")
        if failed != expected_failed:
            raise ValueError("failed_checks do not match the frozen competence gate")
        expected_certified = not expected_failed
        certified = genuine_boolean(self.certified, name="certified")
        if certified != expected_certified:
            raise ValueError("certified does not match the frozen competence gate")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="query competence certificate metadata",
        )
        object.__setattr__(self, "mean_gain", mean_gain)
        object.__setattr__(self, "paired_gain_ci95", interval)
        object.__setattr__(self, "harm_confidence_level", confidence)
        object.__setattr__(self, "harm_risk_upper", expected_harm_upper)
        object.__setattr__(self, "downside_reduction_fraction", downside)
        object.__setattr__(self, "retained_candidate_gain_fraction", retained)
        object.__setattr__(self, "oracle_headroom_fraction", oracle)
        object.__setattr__(self, "protocol_frozen_before_outcomes", protocol_frozen)
        object.__setattr__(
            self,
            "outcomes_used_for_policy_or_gate_selection",
            outcomes_used,
        )
        object.__setattr__(self, "independent_implementation_replay", replayed)
        object.__setattr__(self, "source_gate_passed", source_passed)
        object.__setattr__(self, "failed_checks", expected_failed)
        object.__setattr__(self, "certified", expected_certified)
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match competence certificate")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_COMPETENCE_CERTIFICATE_SCHEMA,
            "schema_version": QUERY_COMPETENCE_CERTIFICATE_VERSION,
            "query_scope": self.query_scope.to_record(),
            "gate": self.gate.to_record(),
            "candidate_policy_id": self.candidate_policy_id,
            "baseline_policy_id": self.baseline_policy_id,
            "protocol_id": self.protocol_id,
            "source_summary_artifact_id": self.source_summary_artifact_id,
            "source_summary_sha256": self.source_summary_sha256,
            "source_result_id": self.source_result_id,
            "verification_artifact_id": self.verification_artifact_id,
            "verification_file_sha256": self.verification_file_sha256,
            "verified_tree_id": self.verified_tree_id,
            "group_count": self.group_count,
            "technical_failures": self.technical_failures,
            "retries": self.retries,
            "replacements": self.replacements,
            "mean_gain": self.mean_gain,
            "paired_gain_ci95": list(self.paired_gain_ci95),
            "harmful_group_count": self.harmful_group_count,
            "harm_confidence_level": self.harm_confidence_level,
            "harm_risk_upper": self.harm_risk_upper,
            "downside_reduction_fraction": self.downside_reduction_fraction,
            "retained_candidate_gain_fraction": (
                self.retained_candidate_gain_fraction
            ),
            "oracle_headroom_fraction": self.oracle_headroom_fraction,
            "protocol_frozen_before_outcomes": (
                self.protocol_frozen_before_outcomes
            ),
            "outcomes_used_for_policy_or_gate_selection": (
                self.outcomes_used_for_policy_or_gate_selection
            ),
            "independent_implementation_replay": (
                self.independent_implementation_replay
            ),
            "source_gate_passed": self.source_gate_passed,
            "failed_checks": list(self.failed_checks),
            "certified": self.certified,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query competence certificate",
    ) -> QueryCompetenceCertificateV1:
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "query_scope",
                "gate",
                "candidate_policy_id",
                "baseline_policy_id",
                "protocol_id",
                "source_summary_artifact_id",
                "source_summary_sha256",
                "source_result_id",
                "verification_artifact_id",
                "verification_file_sha256",
                "verified_tree_id",
                "group_count",
                "technical_failures",
                "retries",
                "replacements",
                "mean_gain",
                "paired_gain_ci95",
                "harmful_group_count",
                "harm_confidence_level",
                "harm_risk_upper",
                "downside_reduction_fraction",
                "retained_candidate_gain_fraction",
                "oracle_headroom_fraction",
                "protocol_frozen_before_outcomes",
                "outcomes_used_for_policy_or_gate_selection",
                "independent_implementation_replay",
                "source_gate_passed",
                "failed_checks",
                "certified",
                "metadata",
                "artifact_id",
            }
        )
        require_exact_fields(value, expected=expected, name=name)
        if value["schema"] != QUERY_COMPETENCE_CERTIFICATE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_COMPETENCE_CERTIFICATE_VERSION:
            raise ValueError(f"{name} schema version changed")
        scope_value = value["query_scope"]
        gate_value = value["gate"]
        if not isinstance(scope_value, Mapping) or not isinstance(gate_value, Mapping):
            raise ValueError(f"{name} scope and gate must be mappings")
        return cls(
            query_scope=SimulatorQueryScopeV1.from_mapping(
                cast(Mapping[str, object], scope_value),
                name=f"{name}.query_scope",
            ),
            gate=QueryCompetenceGateV1.from_mapping(
                cast(Mapping[str, object], gate_value),
                name=f"{name}.gate",
            ),
            candidate_policy_id=cast(str, value["candidate_policy_id"]),
            baseline_policy_id=cast(str, value["baseline_policy_id"]),
            protocol_id=cast(str, value["protocol_id"]),
            source_summary_artifact_id=cast(
                str, value["source_summary_artifact_id"]
            ),
            source_summary_sha256=cast(str, value["source_summary_sha256"]),
            source_result_id=cast(str, value["source_result_id"]),
            verification_artifact_id=cast(
                str, value["verification_artifact_id"]
            ),
            verification_file_sha256=cast(
                str, value["verification_file_sha256"]
            ),
            verified_tree_id=cast(str, value["verified_tree_id"]),
            group_count=cast(int, value["group_count"]),
            technical_failures=cast(int, value["technical_failures"]),
            retries=cast(int, value["retries"]),
            replacements=cast(int, value["replacements"]),
            mean_gain=cast(float, value["mean_gain"]),
            paired_gain_ci95=cast(Sequence[float], value["paired_gain_ci95"]),
            harmful_group_count=cast(int, value["harmful_group_count"]),
            harm_confidence_level=cast(float, value["harm_confidence_level"]),
            harm_risk_upper=cast(float, value["harm_risk_upper"]),
            downside_reduction_fraction=cast(
                float, value["downside_reduction_fraction"]
            ),
            retained_candidate_gain_fraction=cast(
                float, value["retained_candidate_gain_fraction"]
            ),
            oracle_headroom_fraction=cast(
                float, value["oracle_headroom_fraction"]
            ),
            protocol_frozen_before_outcomes=cast(
                bool, value["protocol_frozen_before_outcomes"]
            ),
            outcomes_used_for_policy_or_gate_selection=cast(
                bool, value["outcomes_used_for_policy_or_gate_selection"]
            ),
            independent_implementation_replay=cast(
                bool, value["independent_implementation_replay"]
            ),
            source_gate_passed=cast(bool, value["source_gate_passed"]),
            failed_checks=cast(Sequence[str], value["failed_checks"]),
            certified=cast(bool, value["certified"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )


@dataclass(frozen=True, slots=True)
class QueryCompetenceRegistryV1:
    """Independent query certificates without backend-wide pooling."""

    certificates: Mapping[str, QueryCompetenceCertificateV1]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.certificates, Mapping) or not self.certificates:
            raise ValueError("certificates must be a nonempty mapping")
        certificates: dict[str, QueryCompetenceCertificateV1] = {}
        for raw_query_id, certificate in self.certificates.items():
            query_id = sha256_digest(raw_query_id, name="certificate query key")
            if not isinstance(certificate, QueryCompetenceCertificateV1):
                raise TypeError(
                    "certificate values must be QueryCompetenceCertificateV1 records"
                )
            if certificate.query_scope.query_id != query_id:
                raise ValueError("certificate key does not match its query scope")
            if query_id in certificates:
                raise ValueError("certificates must not contain duplicate queries")
            certificates[query_id] = certificate
        certificates = dict(sorted(certificates.items()))
        object.__setattr__(
            self,
            "certificates",
            MappingProxyType(certificates),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="registry metadata"),
        )
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match competence registry")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def certified_query_ids(self) -> tuple[str, ...]:
        return tuple(
            query_id
            for query_id, certificate in self.certificates.items()
            if certificate.certified
        )

    @property
    def failed_query_ids(self) -> tuple[str, ...]:
        return tuple(
            query_id
            for query_id, certificate in self.certificates.items()
            if not certificate.certified
        )

    def certificate_for_query(
        self,
        query: SimulatorQueryScopeV1 | str,
    ) -> QueryCompetenceCertificateV1 | None:
        query_id = (
            query.query_id
            if isinstance(query, SimulatorQueryScopeV1)
            else sha256_digest(query, name="query_id")
        )
        if query_id is None:
            raise ValueError("query scope lacks a content identity")
        return self.certificates.get(query_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_COMPETENCE_REGISTRY_SCHEMA,
            "schema_version": QUERY_COMPETENCE_REGISTRY_VERSION,
            "certificates": [
                certificate.to_record()
                for certificate in self.certificates.values()
            ],
            "certified_query_ids": list(self.certified_query_ids),
            "failed_query_ids": list(self.failed_query_ids),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query competence registry",
    ) -> QueryCompetenceRegistryV1:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "certificates",
                    "certified_query_ids",
                    "failed_query_ids",
                    "metadata",
                    "artifact_id",
                }
            ),
            name=name,
        )
        if value["schema"] != QUERY_COMPETENCE_REGISTRY_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_COMPETENCE_REGISTRY_VERSION:
            raise ValueError(f"{name} schema version changed")
        raw_certificates = value["certificates"]
        if isinstance(raw_certificates, (str, bytes)) or not isinstance(
            raw_certificates, Sequence
        ):
            raise ValueError(f"{name}.certificates must be a sequence")
        certificates: dict[str, QueryCompetenceCertificateV1] = {}
        for index, raw in enumerate(tuple(raw_certificates)):
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name}.certificates[{index}] must be a mapping")
            certificate = QueryCompetenceCertificateV1.from_mapping(
                cast(Mapping[str, object], raw),
                name=f"{name}.certificates[{index}]",
            )
            query_id = str(certificate.query_scope.query_id)
            if query_id in certificates:
                raise ValueError(f"{name} contains duplicate query certificates")
            certificates[query_id] = certificate
        registry = cls(
            certificates=certificates,
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
        if value["certified_query_ids"] != list(registry.certified_query_ids):
            raise ValueError(f"{name} certified query roster changed")
        if value["failed_query_ids"] != list(registry.failed_query_ids):
            raise ValueError(f"{name} failed query roster changed")
        return registry


def build_query_competence_certificate(
    *,
    query_scope: SimulatorQueryScopeV1,
    gate: QueryCompetenceGateV1,
    candidate_policy_id: str,
    baseline_policy_id: str,
    protocol_id: str,
    source_summary_artifact_id: str,
    source_summary_sha256: str,
    source_result_id: str,
    verification_artifact_id: str,
    verification_file_sha256: str,
    verified_tree_id: str,
    group_count: int,
    technical_failures: int,
    retries: int,
    replacements: int,
    mean_gain: float,
    paired_gain_ci95: Sequence[float],
    harmful_group_count: int,
    harm_confidence_level: float,
    harm_risk_upper: float,
    downside_reduction_fraction: float,
    retained_candidate_gain_fraction: float,
    oracle_headroom_fraction: float,
    protocol_frozen_before_outcomes: bool,
    outcomes_used_for_policy_or_gate_selection: bool,
    independent_implementation_replay: bool,
    source_gate_passed: bool,
    metadata: Mapping[str, Any] | None = None,
) -> QueryCompetenceCertificateV1:
    """Build one certificate while deriving its fail-closed decision."""

    interval = _interval(paired_gain_ci95, name="paired_gain_ci95")
    expected_upper = one_sided_binomial_upper_bound(
        genuine_integer(harmful_group_count, name="harmful_group_count", minimum=0),
        genuine_integer(group_count, name="group_count", minimum=0),
        harm_confidence_level,
    )
    supplied_upper = _fraction(harm_risk_upper, name="harm_risk_upper")
    if not math.isclose(
        supplied_upper,
        expected_upper,
        rel_tol=1e-12,
        abs_tol=1e-14,
    ):
        raise ValueError(
            "harm_risk_upper does not match exact Clopper-Pearson inversion"
        )
    failed = _failed_competence_checks(
        gate=gate,
        group_count=group_count,
        technical_failures=technical_failures,
        retries=retries,
        replacements=replacements,
        mean_gain=mean_gain,
        paired_gain_ci95=interval,
        harm_risk_upper=expected_upper,
        downside_reduction_fraction=downside_reduction_fraction,
        retained_candidate_gain_fraction=retained_candidate_gain_fraction,
        oracle_headroom_fraction=oracle_headroom_fraction,
        protocol_frozen_before_outcomes=protocol_frozen_before_outcomes,
        outcomes_used_for_policy_or_gate_selection=(
            outcomes_used_for_policy_or_gate_selection
        ),
        independent_implementation_replay=independent_implementation_replay,
        source_gate_passed=source_gate_passed,
    )
    return QueryCompetenceCertificateV1(
        query_scope=query_scope,
        gate=gate,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        protocol_id=protocol_id,
        source_summary_artifact_id=source_summary_artifact_id,
        source_summary_sha256=source_summary_sha256,
        source_result_id=source_result_id,
        verification_artifact_id=verification_artifact_id,
        verification_file_sha256=verification_file_sha256,
        verified_tree_id=verified_tree_id,
        group_count=group_count,
        technical_failures=technical_failures,
        retries=retries,
        replacements=replacements,
        mean_gain=mean_gain,
        paired_gain_ci95=interval,
        harmful_group_count=harmful_group_count,
        harm_confidence_level=harm_confidence_level,
        harm_risk_upper=expected_upper,
        downside_reduction_fraction=downside_reduction_fraction,
        retained_candidate_gain_fraction=retained_candidate_gain_fraction,
        oracle_headroom_fraction=oracle_headroom_fraction,
        protocol_frozen_before_outcomes=protocol_frozen_before_outcomes,
        outcomes_used_for_policy_or_gate_selection=(
            outcomes_used_for_policy_or_gate_selection
        ),
        independent_implementation_replay=independent_implementation_replay,
        source_gate_passed=source_gate_passed,
        failed_checks=failed,
        certified=not failed,
        metadata={} if metadata is None else metadata,
    )


def select_query_competent_belief(
    baseline: BeliefT,
    candidate: BeliefT,
    registry: QueryCompetenceRegistryV1,
    *,
    query_scope: SimulatorQueryScopeV1,
    candidate_policy_id: str,
    baseline_policy_id: str,
    common_domain_id: str,
    inference_admissible: bool,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, CompleteBeliefSelectionV1]:
    """Admit only an exact certified query-policy match, else exact fallback."""

    if not isinstance(registry, QueryCompetenceRegistryV1):
        raise TypeError("registry must be a QueryCompetenceRegistryV1")
    if not isinstance(query_scope, SimulatorQueryScopeV1):
        raise TypeError("query_scope must be a SimulatorQueryScopeV1")
    candidate_policy = sha256_digest(
        candidate_policy_id,
        name="candidate_policy_id",
    )
    baseline_policy = sha256_digest(
        baseline_policy_id,
        name="baseline_policy_id",
    )
    common = sha256_digest(common_domain_id, name="common_domain_id")
    inference_ok = genuine_boolean(
        inference_admissible,
        name="inference_admissible",
    )
    certificate = registry.certificate_for_query(query_scope)
    if not inference_ok:
        reason = "inference-rejected"
    elif certificate is None:
        reason = "unknown-query"
    elif candidate_policy != certificate.candidate_policy_id:
        reason = "candidate-policy-mismatch"
    elif baseline_policy != certificate.baseline_policy_id:
        reason = "baseline-policy-mismatch"
    elif not certificate.certified:
        reason = "query-competence-rejected"
    else:
        reason = "query-competence-authorized"
    accepted = reason == "query-competence-authorized"
    caller_metadata = frozen_finite_json_mapping(
        metadata,
        name="query competence selection metadata",
    )
    routing_metadata = {
        "guard": QUERY_COMPETENCE_CERTIFICATE_SCHEMA,
        "query_id": query_scope.query_id,
        "registry_id": registry.artifact_id,
        "certificate_id": (
            None if certificate is None else certificate.artifact_id
        ),
        "candidate_policy_id": candidate_policy,
        "baseline_policy_id": baseline_policy,
        "query_certified": bool(certificate is not None and certificate.certified),
        "routing_reason": reason,
        "caller": plain_json(caller_metadata),
    }
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=common,
        certificate_id=(
            str(registry.artifact_id)
            if certificate is None
            else str(certificate.artifact_id)
        ),
        inference_admissible=inference_ok,
        regret_guard_accepted=accepted,
        reason=reason,
        metadata=routing_metadata,
    )
    return select_complete_belief(
        baseline,
        candidate,
        decision,
        metadata=routing_metadata,
    )


def save_query_competence_registry(
    registry: QueryCompetenceRegistryV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(registry, QueryCompetenceRegistryV1):
        raise TypeError("registry must be a QueryCompetenceRegistryV1")
    write_atomic_json(registry.to_record(), path, overwrite=overwrite)


def load_query_competence_registry(path: str | Path) -> QueryCompetenceRegistryV1:
    payload = load_strict_json_object(path, label="query competence registry")
    return QueryCompetenceRegistryV1.from_mapping(payload)


__all__ = [
    "QUERY_COMPETENCE_CERTIFICATE_SCHEMA",
    "QUERY_COMPETENCE_CERTIFICATE_VERSION",
    "QUERY_COMPETENCE_GATE_SCHEMA",
    "QUERY_COMPETENCE_GATE_VERSION",
    "QUERY_COMPETENCE_REGISTRY_SCHEMA",
    "QUERY_COMPETENCE_REGISTRY_VERSION",
    "QUERY_SCOPE_SCHEMA",
    "QUERY_SCOPE_VERSION",
    "QueryCompetenceCertificateV1",
    "QueryCompetenceGateV1",
    "QueryCompetenceRegistryV1",
    "SimulatorQueryScopeV1",
    "build_query_competence_certificate",
    "load_query_competence_registry",
    "save_query_competence_registry",
    "select_query_competent_belief",
]
