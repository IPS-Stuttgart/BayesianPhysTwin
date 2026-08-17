"""Calibration-frozen domain guard with exact complete-belief fallback.

A candidate may improve dynamic continuation while harming another regime such
as quasi-static contact. This module learns only a domain-level authorization
from independent calibration groups. Application receives a domain identifier,
not an application outcome, and rejection routes the exact registered baseline
belief through :mod:`bayesian_phystwin.complete_belief_selection`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import content_id, sha256_digest
from .complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    select_complete_belief,
)

CALIBRATION_DOMAIN_GUARD_SCHEMA = "bayesian_phystwin.calibration_domain_guard"
CALIBRATION_DOMAIN_GUARD_VERSION = 1
CALIBRATION_DOMAIN_DECISION_SCHEMA = "bayesian_phystwin.calibration_domain_decision"
CALIBRATION_DOMAIN_DECISION_VERSION = 1
CALIBRATION_DOMAIN_DATA_SCHEMA = "bayesian_phystwin.calibration_domain_data"
CALIBRATION_DOMAIN_DATA_VERSION = 1

BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _canonical_strings(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        source = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of strings") from error
    result = tuple(
        _canonical_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(source)
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite number")
    result = float(raw.item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _loss_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional numeric array")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return np.array(result, dtype=np.float64, copy=True, order="C")


def _relative_improvements(value: Sequence[float]) -> tuple[float, ...]:
    improvements = tuple(
        _finite_float(item, name=f"relative_improvements[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if not improvements:
        raise ValueError("relative_improvements must not be empty")
    if any(item > 1.0 for item in improvements):
        raise ValueError("relative improvements cannot exceed one")
    return improvements


@dataclass(frozen=True, slots=True)
class CalibrationDomainGuardConfigV1:
    """Frozen finite-group rule for one calibration-domain guard."""

    minimum_group_count: int = 3
    minimum_mean_relative_improvement: float = 0.05
    minimum_win_fraction: float = 2.0 / 3.0
    maximum_single_group_relative_regression: float = 0.05
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        minimum_group_count = genuine_integer(
            self.minimum_group_count,
            name="minimum_group_count",
            minimum=1,
        )
        minimum_mean = _finite_float(
            self.minimum_mean_relative_improvement,
            name="minimum_mean_relative_improvement",
        )
        minimum_win_fraction = _finite_float(
            self.minimum_win_fraction,
            name="minimum_win_fraction",
        )
        maximum_regression = _finite_float(
            self.maximum_single_group_relative_regression,
            name="maximum_single_group_relative_regression",
        )
        tolerance = _finite_float(
            self.numerical_tolerance,
            name="numerical_tolerance",
        )
        if not 0.0 <= minimum_mean <= 1.0:
            raise ValueError("minimum_mean_relative_improvement must lie in [0, 1]")
        if not 0.0 < minimum_win_fraction <= 1.0:
            raise ValueError("minimum_win_fraction must lie in (0, 1]")
        if not 0.0 <= maximum_regression <= 1.0:
            raise ValueError(
                "maximum_single_group_relative_regression must lie in [0, 1]"
            )
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("numerical_tolerance must lie in [0, 1)")
        object.__setattr__(self, "minimum_group_count", minimum_group_count)
        object.__setattr__(
            self,
            "minimum_mean_relative_improvement",
            minimum_mean,
        )
        object.__setattr__(self, "minimum_win_fraction", minimum_win_fraction)
        object.__setattr__(
            self,
            "maximum_single_group_relative_regression",
            maximum_regression,
        )
        object.__setattr__(self, "numerical_tolerance", tolerance)

    def descriptor(self) -> dict[str, object]:
        return {
            "minimum_group_count": self.minimum_group_count,
            "minimum_mean_relative_improvement": (
                self.minimum_mean_relative_improvement
            ),
            "minimum_win_fraction": self.minimum_win_fraction,
            "maximum_single_group_relative_regression": (
                self.maximum_single_group_relative_regression
            ),
            "numerical_tolerance": self.numerical_tolerance,
        }


@dataclass(frozen=True, slots=True)
class CalibrationDomainDecisionV1:
    """Calibration-only support decision for one declared domain."""

    domain_id: str
    group_ids: Sequence[str]
    relative_improvements: Sequence[float]
    mean_relative_improvement: float
    win_count: int
    required_win_count: int
    worst_relative_improvement: float
    calibration_supported: bool
    reasons: Sequence[str]
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        domain_id = _canonical_string(self.domain_id, name="domain_id")
        groups = _canonical_strings(self.group_ids, name="group_ids")
        if len(set(groups)) != len(groups):
            raise ValueError("group_ids must not contain duplicates")
        improvements = _relative_improvements(self.relative_improvements)
        if len(improvements) != len(groups):
            raise ValueError("relative_improvements length must match group_ids")
        order = tuple(sorted(range(len(groups)), key=groups.__getitem__))
        groups = tuple(groups[index] for index in order)
        improvements = tuple(improvements[index] for index in order)
        mean_improvement = _finite_float(
            self.mean_relative_improvement,
            name="mean_relative_improvement",
        )
        win_count = genuine_integer(
            self.win_count,
            name="win_count",
            minimum=0,
        )
        required_win_count = genuine_integer(
            self.required_win_count,
            name="required_win_count",
            minimum=0,
        )
        if win_count > len(groups) or required_win_count > len(groups):
            raise ValueError("win counts cannot exceed the calibration group count")
        worst_improvement = _finite_float(
            self.worst_relative_improvement,
            name="worst_relative_improvement",
        )
        supported = genuine_boolean(
            self.calibration_supported,
            name="calibration_supported",
        )
        reasons = _canonical_strings(self.reasons, name="reasons")
        if len(set(reasons)) != len(reasons):
            raise ValueError("reasons must not contain duplicates")
        reasons = tuple(sorted(reasons))
        object.__setattr__(self, "domain_id", domain_id)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "relative_improvements", improvements)
        object.__setattr__(
            self,
            "mean_relative_improvement",
            mean_improvement,
        )
        object.__setattr__(self, "win_count", win_count)
        object.__setattr__(self, "required_win_count", required_win_count)
        object.__setattr__(
            self,
            "worst_relative_improvement",
            worst_improvement,
        )
        object.__setattr__(self, "calibration_supported", supported)
        object.__setattr__(self, "reasons", reasons)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match domain decision")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_DOMAIN_DECISION_SCHEMA,
            "schema_version": CALIBRATION_DOMAIN_DECISION_VERSION,
            "domain_id": self.domain_id,
            "group_ids": list(self.group_ids),
            "relative_improvements": list(self.relative_improvements),
            "mean_relative_improvement": self.mean_relative_improvement,
            "win_count": self.win_count,
            "required_win_count": self.required_win_count,
            "worst_relative_improvement": self.worst_relative_improvement,
            "calibration_supported": self.calibration_supported,
            "reasons": list(self.reasons),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def _evaluate_domain(
    domain_id: str,
    group_ids: Sequence[str],
    relative_improvements: Sequence[float],
    config: CalibrationDomainGuardConfigV1,
) -> CalibrationDomainDecisionV1:
    groups = _canonical_strings(group_ids, name="group_ids")
    improvements = _relative_improvements(relative_improvements)
    if len(groups) != len(improvements):
        raise ValueError("relative_improvements length must match group_ids")
    count = len(groups)
    tolerance = config.numerical_tolerance
    required_win_count = int(math.ceil(config.minimum_win_fraction * count - tolerance))
    win_count = sum(item > tolerance for item in improvements)
    mean_improvement = float(np.mean(np.asarray(improvements, dtype=np.float64)))
    worst_improvement = min(improvements)
    reasons: list[str] = []
    if count < config.minimum_group_count:
        reasons.append("insufficient-calibration-groups")
    if mean_improvement + tolerance < config.minimum_mean_relative_improvement:
        reasons.append("mean-improvement-below-threshold")
    if win_count < required_win_count:
        reasons.append("insufficient-calibration-wins")
    if worst_improvement < -config.maximum_single_group_relative_regression - tolerance:
        reasons.append("single-group-regression-exceeds-limit")
    supported = not reasons
    if supported:
        reasons.append("calibration-criteria-passed")
    return CalibrationDomainDecisionV1(
        domain_id=domain_id,
        group_ids=groups,
        relative_improvements=improvements,
        mean_relative_improvement=mean_improvement,
        win_count=win_count,
        required_win_count=required_win_count,
        worst_relative_improvement=worst_improvement,
        calibration_supported=supported,
        reasons=reasons,
    )


@dataclass(frozen=True, slots=True)
class CalibrationDomainGuardCertificateV1:
    """Content-addressed calibration support and information-boundary record."""

    calibration_partition_id: str
    statistical_unit: str
    metric: str
    config: CalibrationDomainGuardConfigV1
    calibration_data_id: str
    decisions: Sequence[CalibrationDomainDecisionV1]
    guard_frozen_before_application_outcomes: bool
    application_outcomes_used_for_guard_selection: bool
    calibration_groups_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        partition_id = sha256_digest(
            self.calibration_partition_id,
            name="calibration_partition_id",
        )
        statistical_unit = _canonical_string(
            self.statistical_unit,
            name="statistical_unit",
        )
        metric = _canonical_string(self.metric, name="metric")
        if not isinstance(self.config, CalibrationDomainGuardConfigV1):
            raise TypeError("config must be a CalibrationDomainGuardConfigV1")
        calibration_data_id = sha256_digest(
            self.calibration_data_id,
            name="calibration_data_id",
        )
        decisions = tuple(self.decisions)
        if not decisions or any(
            not isinstance(item, CalibrationDomainDecisionV1) for item in decisions
        ):
            raise TypeError(
                "decisions must contain CalibrationDomainDecisionV1 records"
            )
        if len({item.domain_id for item in decisions}) != len(decisions):
            raise ValueError("decisions must not contain duplicate domains")
        decisions = tuple(sorted(decisions, key=lambda item: item.domain_id))
        for decision in decisions:
            expected = _evaluate_domain(
                decision.domain_id,
                decision.group_ids,
                decision.relative_improvements,
                self.config,
            )
            if decision.artifact_id != expected.artifact_id:
                raise ValueError(
                    f"decision for domain {decision.domain_id!r} "
                    "does not match the frozen configuration"
                )
        frozen_before = genuine_boolean(
            self.guard_frozen_before_application_outcomes,
            name="guard_frozen_before_application_outcomes",
        )
        application_used = genuine_boolean(
            self.application_outcomes_used_for_guard_selection,
            name="application_outcomes_used_for_guard_selection",
        )
        independent = genuine_boolean(
            self.calibration_groups_independent,
            name="calibration_groups_independent",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="calibration domain guard metadata",
        )
        object.__setattr__(self, "calibration_partition_id", partition_id)
        object.__setattr__(self, "statistical_unit", statistical_unit)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "calibration_data_id", calibration_data_id)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "guard_frozen_before_application_outcomes",
            frozen_before,
        )
        object.__setattr__(
            self,
            "application_outcomes_used_for_guard_selection",
            application_used,
        )
        object.__setattr__(self, "calibration_groups_independent", independent)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match domain guard")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def deployment_admissible(self) -> bool:
        return (
            self.guard_frozen_before_application_outcomes
            and not self.application_outcomes_used_for_guard_selection
            and self.calibration_groups_independent
        )

    @property
    def supported_domains(self) -> tuple[str, ...]:
        return tuple(
            decision.domain_id
            for decision in self.decisions
            if decision.calibration_supported
        )

    def decision_for_domain(
        self,
        domain_id: str,
    ) -> CalibrationDomainDecisionV1 | None:
        canonical = _canonical_string(domain_id, name="domain_id")
        return next(
            (
                decision
                for decision in self.decisions
                if decision.domain_id == canonical
            ),
            None,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_DOMAIN_GUARD_SCHEMA,
            "schema_version": CALIBRATION_DOMAIN_GUARD_VERSION,
            "calibration_partition_id": self.calibration_partition_id,
            "statistical_unit": self.statistical_unit,
            "metric": self.metric,
            "config": self.config.descriptor(),
            "calibration_data_id": self.calibration_data_id,
            "decisions": [decision.to_record() for decision in self.decisions],
            "information_boundary": {
                "guard_frozen_before_application_outcomes": (
                    self.guard_frozen_before_application_outcomes
                ),
                "application_outcomes_used_for_guard_selection": (
                    self.application_outcomes_used_for_guard_selection
                ),
                "calibration_groups_independent": (self.calibration_groups_independent),
                "deployment_admissible": self.deployment_admissible,
            },
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def fit_calibration_domain_guard(
    *,
    calibration_partition_id: str,
    statistical_unit: str,
    metric: str,
    group_ids: Sequence[str],
    domain_ids: Sequence[str],
    candidate_losses: object,
    fallback_losses: object,
    guard_frozen_before_application_outcomes: bool,
    application_outcomes_used_for_guard_selection: bool,
    calibration_groups_independent: bool,
    config: CalibrationDomainGuardConfigV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationDomainGuardCertificateV1:
    """Fit domain support using only one frozen calibration partition."""

    guard_config = CalibrationDomainGuardConfigV1() if config is None else config
    if not isinstance(guard_config, CalibrationDomainGuardConfigV1):
        raise TypeError("config must be a CalibrationDomainGuardConfigV1")
    partition_id = sha256_digest(
        calibration_partition_id,
        name="calibration_partition_id",
    )
    unit = _canonical_string(statistical_unit, name="statistical_unit")
    metric_name = _canonical_string(metric, name="metric")
    groups = _canonical_strings(group_ids, name="group_ids")
    domains = _canonical_strings(domain_ids, name="domain_ids")
    if len(set(groups)) != len(groups):
        raise ValueError("group_ids must not contain duplicates")
    candidate = _loss_vector(candidate_losses, name="candidate_losses")
    fallback = _loss_vector(fallback_losses, name="fallback_losses")
    if not (len(groups) == len(domains) == len(candidate) == len(fallback)):
        raise ValueError(
            "group_ids, domain_ids, candidate_losses, and fallback_losses "
            "must have equal lengths"
        )
    if np.any(fallback <= 0.0):
        raise ValueError("fallback_losses must be strictly positive")
    order = tuple(sorted(range(len(groups)), key=groups.__getitem__))
    records = [
        {
            "group_id": groups[index],
            "domain_id": domains[index],
            "candidate_loss": float(candidate[index]),
            "fallback_loss": float(fallback[index]),
        }
        for index in order
    ]
    calibration_data_id = content_id(
        {
            "schema": CALIBRATION_DOMAIN_DATA_SCHEMA,
            "schema_version": CALIBRATION_DOMAIN_DATA_VERSION,
            "calibration_partition_id": partition_id,
            "statistical_unit": unit,
            "metric": metric_name,
            "records": records,
        }
    )
    decisions: list[CalibrationDomainDecisionV1] = []
    for domain_id in sorted(set(domains)):
        indices = [index for index in order if domains[index] == domain_id]
        relative = tuple(
            float((fallback[index] - candidate[index]) / fallback[index])
            for index in indices
        )
        decisions.append(
            _evaluate_domain(
                domain_id,
                tuple(groups[index] for index in indices),
                relative,
                guard_config,
            )
        )
    return CalibrationDomainGuardCertificateV1(
        calibration_partition_id=partition_id,
        statistical_unit=unit,
        metric=metric_name,
        config=guard_config,
        calibration_data_id=calibration_data_id,
        decisions=decisions,
        guard_frozen_before_application_outcomes=(
            guard_frozen_before_application_outcomes
        ),
        application_outcomes_used_for_guard_selection=(
            application_outcomes_used_for_guard_selection
        ),
        calibration_groups_independent=calibration_groups_independent,
        metadata={} if metadata is None else metadata,
    )


def select_calibration_domain_guarded_belief(
    baseline: BeliefT,
    candidate: BeliefT,
    certificate: CalibrationDomainGuardCertificateV1,
    *,
    domain_id: str,
    common_domain_id: str,
    inference_admissible: bool,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, CompleteBeliefSelectionV1]:
    """Select one complete belief without receiving an application outcome."""

    if not isinstance(certificate, CalibrationDomainGuardCertificateV1):
        raise TypeError("certificate must be a CalibrationDomainGuardCertificateV1")
    domain = _canonical_string(domain_id, name="domain_id")
    common = sha256_digest(common_domain_id, name="common_domain_id")
    inference_ok = genuine_boolean(
        inference_admissible,
        name="inference_admissible",
    )
    decision = certificate.decision_for_domain(domain)
    calibration_supported = bool(
        decision is not None and decision.calibration_supported
    )
    accepted = (
        inference_ok and certificate.deployment_admissible and calibration_supported
    )
    if not inference_ok:
        reason = "inference-rejected"
    elif decision is None:
        reason = "unknown-calibration-domain"
    elif not certificate.deployment_admissible:
        reason = "calibration-information-boundary-rejected"
    elif not decision.calibration_supported:
        reason = "calibration-domain-rejected"
    else:
        reason = "calibration-domain-authorized"
    caller_metadata = frozen_finite_json_mapping(
        metadata,
        name="calibration domain selection metadata",
    )
    routing_metadata = {
        "guard": CALIBRATION_DOMAIN_GUARD_SCHEMA,
        "domain_id": domain,
        "domain_decision_id": None if decision is None else decision.artifact_id,
        "domain_reasons": [] if decision is None else list(decision.reasons),
        "calibration_supported": calibration_supported,
        "certificate_deployment_admissible": (certificate.deployment_admissible),
        "routing_reason": reason,
        "caller": plain_json(caller_metadata),
    }
    guard_decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=common,
        certificate_id=str(certificate.artifact_id),
        inference_admissible=inference_ok,
        regret_guard_accepted=accepted,
        reason=reason,
        metadata=routing_metadata,
    )
    return select_complete_belief(
        baseline,
        candidate,
        guard_decision,
        metadata=routing_metadata,
    )


__all__ = [
    "CALIBRATION_DOMAIN_DATA_SCHEMA",
    "CALIBRATION_DOMAIN_DATA_VERSION",
    "CALIBRATION_DOMAIN_DECISION_SCHEMA",
    "CALIBRATION_DOMAIN_DECISION_VERSION",
    "CALIBRATION_DOMAIN_GUARD_SCHEMA",
    "CALIBRATION_DOMAIN_GUARD_VERSION",
    "CalibrationDomainDecisionV1",
    "CalibrationDomainGuardCertificateV1",
    "CalibrationDomainGuardConfigV1",
    "fit_calibration_domain_guard",
    "select_calibration_domain_guarded_belief",
]
