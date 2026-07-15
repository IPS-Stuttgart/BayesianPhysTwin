"""Fail-closed timing gate for semantic task posteriors.

The timing contract deliberately uses one named monotonic clock.  Generation
duration covers the interval immediately before semantic-model invocation
through complete materialization of the :class:`TaskPosterior`.  Forecast age
starts when the observation/query snapshot is captured and ends when this gate
is evaluated.  The planning deadline is the latest time at which the gated
posterior may be handed to the planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar, Mapping

import numpy as np

from causal4d.contracts import TaskPosterior


SEMANTIC_TIMING_SCHEMA_VERSION = 1
SEMANTIC_TIMING_SCOPE = "semantic_generation_start_to_task_posterior_materialized"


def _finite_nonnegative_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("schema_version must be an integer")
    version = int(value)
    if version != SEMANTIC_TIMING_SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SEMANTIC_TIMING_SCHEMA_VERSION}")
    return version


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _trusted_clock_id(value: object) -> str:
    clock_id = _string(value, name="expected_clock_id")
    if not clock_id or clock_id != clock_id.strip():
        raise ValueError("expected_clock_id must be nonempty without outer whitespace")
    return clock_id


def _exact_mapping_keys(
    values: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    keys = set(values)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise ValueError(f"{name} keys are invalid; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class SemanticTimingMetadata:
    """Timestamps for one semantic forecast on a shared monotonic clock.

    ``query_created_monotonic_s`` identifies capture of the observation/query
    snapshot used by the semantic model.  ``generation_started_monotonic_s`` is
    sampled immediately before model invocation and
    ``generation_completed_monotonic_s`` only after the resulting task
    posterior is fully materialized.  ``current_monotonic_s`` is sampled at
    this gate.  ``planning_deadline_monotonic_s`` is the latest safe planner
    handoff time.
    """

    schema_version: int
    clock: str
    clock_id: str
    timing_scope: str
    query_created_monotonic_s: float
    generation_started_monotonic_s: float
    generation_completed_monotonic_s: float
    current_monotonic_s: float
    planning_deadline_monotonic_s: float

    _KEYS: ClassVar[set[str]] = {
        "schema_version",
        "clock",
        "clock_id",
        "timing_scope",
        "query_created_monotonic_s",
        "generation_started_monotonic_s",
        "generation_completed_monotonic_s",
        "current_monotonic_s",
        "planning_deadline_monotonic_s",
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.clock != "monotonic":
            raise ValueError("clock must be 'monotonic'")
        if (
            not isinstance(self.clock_id, str)
            or not self.clock_id
            or self.clock_id != self.clock_id.strip()
        ):
            raise ValueError("clock_id must be nonempty without outer whitespace")
        if self.timing_scope != SEMANTIC_TIMING_SCOPE:
            raise ValueError(f"timing_scope must be {SEMANTIC_TIMING_SCOPE!r}")

        names = (
            "query_created_monotonic_s",
            "generation_started_monotonic_s",
            "generation_completed_monotonic_s",
            "current_monotonic_s",
            "planning_deadline_monotonic_s",
        )
        for name in names:
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_number(getattr(self, name), name=name),
            )

        if self.query_created_monotonic_s > self.generation_started_monotonic_s:
            raise ValueError("query creation must not follow generation start")
        if self.generation_started_monotonic_s > self.generation_completed_monotonic_s:
            raise ValueError("generation start must not follow generation completion")
        if self.generation_completed_monotonic_s > self.current_monotonic_s:
            raise ValueError("generation completion must not follow gate evaluation")
        if self.planning_deadline_monotonic_s < self.query_created_monotonic_s:
            raise ValueError("planning deadline must not precede query creation")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SemanticTimingMetadata:
        """Parse strict runtime timing metadata without accepting silent extras."""

        _exact_mapping_keys(values, cls._KEYS, name="semantic timing metadata")
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            clock=_string(values["clock"], name="clock"),
            clock_id=_string(values["clock_id"], name="clock_id"),
            timing_scope=_string(values["timing_scope"], name="timing_scope"),
            query_created_monotonic_s=_finite_nonnegative_number(
                values["query_created_monotonic_s"],
                name="query_created_monotonic_s",
            ),
            generation_started_monotonic_s=_finite_nonnegative_number(
                values["generation_started_monotonic_s"],
                name="generation_started_monotonic_s",
            ),
            generation_completed_monotonic_s=_finite_nonnegative_number(
                values["generation_completed_monotonic_s"],
                name="generation_completed_monotonic_s",
            ),
            current_monotonic_s=_finite_nonnegative_number(
                values["current_monotonic_s"],
                name="current_monotonic_s",
            ),
            planning_deadline_monotonic_s=_finite_nonnegative_number(
                values["planning_deadline_monotonic_s"],
                name="planning_deadline_monotonic_s",
            ),
        )

    @property
    def inference_duration_s(self) -> float:
        return (
            self.generation_completed_monotonic_s - self.generation_started_monotonic_s
        )

    @property
    def forecast_age_s(self) -> float:
        return self.current_monotonic_s - self.query_created_monotonic_s

    @property
    def planning_deadline_margin_s(self) -> float:
        return self.planning_deadline_monotonic_s - self.current_monotonic_s


@dataclass(frozen=True)
class SemanticFreshnessLimits:
    """Maximum accepted generation duration and query age, in seconds."""

    schema_version: int
    max_inference_duration_s: float
    max_forecast_age_s: float

    _KEYS: ClassVar[set[str]] = {
        "schema_version",
        "max_inference_duration_s",
        "max_forecast_age_s",
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        for name in ("max_inference_duration_s", "max_forecast_age_s"):
            value = _finite_nonnegative_number(getattr(self, name), name=name)
            if value == 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SemanticFreshnessLimits:
        """Parse strict policy limits without accepting silent extras."""

        _exact_mapping_keys(values, cls._KEYS, name="semantic freshness limits")
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            max_inference_duration_s=_finite_nonnegative_number(
                values["max_inference_duration_s"],
                name="max_inference_duration_s",
            ),
            max_forecast_age_s=_finite_nonnegative_number(
                values["max_forecast_age_s"],
                name="max_forecast_age_s",
            ),
        )


@dataclass(frozen=True)
class SemanticFreshnessDecision:
    """Auditable outcome of the freshness gate."""

    accepted: bool
    applied_beta: float
    reasons: tuple[str, ...]
    inference_duration_s: float | None
    forecast_age_s: float | None
    planning_deadline_margin_s: float | None
    clock_id: str | None
    expected_clock_id: str | None
    timing_scope: str
    validation_error: str | None = None


def _physical_only_task(task: TaskPosterior) -> TaskPosterior:
    fallback = TaskPosterior(
        context=task.context,
        physical_posterior_id=task.physical_posterior_id,
        component_ids=task.component_ids,
        physical_weights=task.physical_weights,
        task_weights=task.physical_weights,
        semantic_log_scores=task.semantic_log_scores,
        beta=0.0,
        query_node_indices=task.query_node_indices,
        semantic_source=task.semantic_source,
        metadata=task.metadata,
    )
    if fallback.physical_weights.tobytes() != fallback.task_weights.tobytes():
        raise RuntimeError("semantic freshness fallback changed physical weights")
    return fallback


def _rejected_decision(
    task: TaskPosterior,
    *,
    reason: str,
    validation_error: str,
    expected_clock_id: str | None = None,
) -> tuple[TaskPosterior, SemanticFreshnessDecision]:
    fallback = _physical_only_task(task)
    return fallback, SemanticFreshnessDecision(
        accepted=False,
        applied_beta=0.0,
        reasons=(reason,),
        inference_duration_s=None,
        forecast_age_s=None,
        planning_deadline_margin_s=None,
        clock_id=None,
        expected_clock_id=expected_clock_id,
        timing_scope=SEMANTIC_TIMING_SCOPE,
        validation_error=validation_error,
    )


def apply_semantic_freshness_gate(
    task: TaskPosterior,
    timing: SemanticTimingMetadata | Mapping[str, object],
    limits: SemanticFreshnessLimits | Mapping[str, object],
    *,
    expected_clock_id: str | None = None,
) -> tuple[TaskPosterior, SemanticFreshnessDecision]:
    """Preserve fresh evidence and disable stale or malformed semantic evidence.

    Runtime callers should pass raw mappings so malformed telemetry is converted
    to a physical-only ``beta=0`` result rather than raising into the planner.
    ``expected_clock_id`` must come from trusted planner/runtime context and must
    never be copied from the untrusted timing payload.  Omitting it deliberately
    fails closed.  Accepted inputs return the exact original ``TaskPosterior``
    object.  Rejected inputs retain its provenance and semantic scores but copy
    physical weights byte-for-byte into task weights.
    """

    try:
        trusted_clock_id = _trusted_clock_id(expected_clock_id)
    except (TypeError, ValueError) as error:
        return _rejected_decision(
            task,
            reason="invalid_expected_clock_id",
            validation_error=str(error),
        )

    try:
        checked_limits = (
            limits
            if isinstance(limits, SemanticFreshnessLimits)
            else SemanticFreshnessLimits.from_mapping(limits)
        )
    except (KeyError, TypeError, ValueError) as error:
        return _rejected_decision(
            task,
            reason="invalid_freshness_limits",
            validation_error=str(error),
            expected_clock_id=trusted_clock_id,
        )

    try:
        checked_timing = (
            timing
            if isinstance(timing, SemanticTimingMetadata)
            else SemanticTimingMetadata.from_mapping(timing)
        )
    except (KeyError, TypeError, ValueError) as error:
        return _rejected_decision(
            task,
            reason="invalid_timing_metadata",
            validation_error=str(error),
            expected_clock_id=trusted_clock_id,
        )

    reasons: list[str] = []
    if checked_timing.clock_id != trusted_clock_id:
        reasons.append("clock_id_mismatch")
    if checked_timing.inference_duration_s > checked_limits.max_inference_duration_s:
        reasons.append("semantic_inference_timeout")
    if checked_timing.forecast_age_s > checked_limits.max_forecast_age_s:
        reasons.append("stale_semantic_forecast")
    if checked_timing.planning_deadline_margin_s < 0.0:
        reasons.append("planning_deadline_missed")

    if reasons:
        fallback = _physical_only_task(task)
        applied_task = fallback
        applied_beta = 0.0
    else:
        applied_task = task
        applied_beta = task.beta

    decision = SemanticFreshnessDecision(
        accepted=not reasons,
        applied_beta=applied_beta,
        reasons=tuple(reasons),
        inference_duration_s=checked_timing.inference_duration_s,
        forecast_age_s=checked_timing.forecast_age_s,
        planning_deadline_margin_s=checked_timing.planning_deadline_margin_s,
        clock_id=checked_timing.clock_id,
        expected_clock_id=trusted_clock_id,
        timing_scope=checked_timing.timing_scope,
    )
    if reasons and not np.array_equal(
        applied_task.physical_weights, applied_task.task_weights
    ):
        raise RuntimeError("semantic freshness rejection failed closed")
    return applied_task, decision
