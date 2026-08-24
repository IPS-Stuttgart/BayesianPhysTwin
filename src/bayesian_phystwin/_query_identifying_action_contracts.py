"""Immutable contracts for target-closed query-identifying action design."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .nuisance_aware_information import NuisanceAwareInformationState

QUERY_IDENTIFYING_ACTION_SCHEMA: Final = (
    "bayesian_phystwin.query_identifying_action_candidate"
)
QUERY_IDENTIFYING_ACTION_VERSION: Final = 1
QUERY_IDENTIFYING_ACTION_EVALUATION_SCHEMA: Final = (
    "bayesian_phystwin.query_identifying_action_evaluation"
)
QUERY_IDENTIFYING_ACTION_EVALUATION_VERSION: Final = 1
QUERY_IDENTIFYING_ACTION_DESIGN_SCHEMA: Final = (
    "bayesian_phystwin.query_identifying_action_design"
)
QUERY_IDENTIFYING_ACTION_DESIGN_VERSION: Final = 1
QUERY_IDENTIFYING_ACTION_DESIGN_SEMANTICS: Final = (
    "finite-action-query-covariance-contraction-after-nuisance-marginalization-v1"
)
QUERY_IDENTIFYING_ACTION_DESIGN_CLAIM_BOUNDARY: Final = (
    "Local linear-Gaussian ranking of an externally supplied finite action "
    "roster under the exact registered prior information state, query, "
    "prospective observation blocks, reliability, costs, risks, and thresholds "
    "only. The decision does not generate or execute an action, access an "
    "outcome, establish global or nonlinear identifiability, validate a provider "
    "or simulator, calibrate deployment uncertainty, establish unseen-object "
    "transfer, certify physical safety, or demonstrate Causal4D benefit."
)


class QueryIdentifyingActionStatus(str, Enum):
    """Per-action admission state under the frozen design objective."""

    ELIGIBLE = "eligible"
    SAFETY_REJECTED = "safety_rejected"
    RISK_REJECTED = "risk_rejected"
    INSUFFICIENT_GAIN = "insufficient_gain"
    TRIVIAL_QUERY = "trivial_query"


class QueryIdentifyingDesignStatus(str, Enum):
    """Finite-roster design decision."""

    ACTION_SELECTED = "action_selected"
    NO_ELIGIBLE_ACTION = "no_eligible_action"
    INSUFFICIENT_GAIN = "insufficient_gain"
    TRIVIAL_QUERY = "trivial_query"


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _real_matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _exact_symmetric(value: object, *, name: str) -> np.ndarray:
    matrix = _real_matrix(value, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError(f"{name} must be exactly symmetric")
    return matrix


def _positive_definite(value: object, *, name: str) -> np.ndarray:
    matrix = _exact_symmetric(value, name=name)
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must be nonempty")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


def _reliability_vector(value: object, row_count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("reliability must contain real numeric values")
    reliability = np.asarray(raw, dtype=np.float64)
    if reliability.ndim == 0:
        reliability = np.full(row_count, float(reliability), dtype=np.float64)
    if reliability.shape != (row_count,):
        raise ValueError(
            "reliability must be a scalar or one value per observation row"
        )
    if not np.all(np.isfinite(reliability)) or not np.all(
        (reliability >= 0.0) & (reliability <= 1.0)
    ):
        raise ValueError("reliability must lie in [0, 1]")
    return reliability


def _immutable_float64(value: object) -> np.ndarray:
    return cast(np.ndarray, immutable_array(value, dtype=np.float64))


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _marginal_state_covariance(state: NuisanceAwareInformationState) -> np.ndarray:
    precision = state.marginal_state_precision()
    factor = np.linalg.cholesky(precision)
    identity = np.eye(state.state_dimension, dtype=np.float64)
    intermediate = np.linalg.solve(factor, identity)
    return _symmetric(np.linalg.solve(factor.T, intermediate))


def _query_covariance(
    state: NuisanceAwareInformationState,
    query_jacobian: np.ndarray,
) -> np.ndarray:
    covariance = query_jacobian @ _marginal_state_covariance(state) @ query_jacobian.T
    return _symmetric(covariance)


def _normalized_covariance(
    covariance: np.ndarray,
    scale_cholesky: np.ndarray,
) -> np.ndarray:
    left = np.linalg.solve(scale_cholesky, covariance)
    normalized = np.linalg.solve(scale_cholesky, left.T).T
    return _symmetric(normalized)


def _information_state_record(
    state: NuisanceAwareInformationState,
) -> dict[str, object]:
    return {
        "state_precision": _array_record(state.state_precision),
        "nuisance_precision": _array_record(state.nuisance_precision),
        "state_nuisance_precision": _array_record(
            state.state_nuisance_precision
        ),
    }


@dataclass(frozen=True, slots=True)
class QueryIdentifyingActionCandidateV1:
    """One externally proposed action and its prospective observation block."""

    action_id: str
    state_jacobian: np.ndarray = field(repr=False)
    nuisance_jacobian: np.ndarray = field(repr=False)
    observation_covariance: np.ndarray = field(repr=False)
    reliability: float | np.ndarray = field(default=1.0, repr=False)
    dimensionless_cost: float = 0.0
    dimensionless_risk: float = 0.0
    safety_admissible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        action_id = _nonempty_string(self.action_id, name="action_id")
        state_jacobian = _real_matrix(self.state_jacobian, name="state_jacobian")
        nuisance_jacobian = _real_matrix(
            self.nuisance_jacobian,
            name="nuisance_jacobian",
        )
        covariance = _positive_definite(
            self.observation_covariance,
            name="observation_covariance",
        )
        if state_jacobian.shape[0] == 0 or state_jacobian.shape[1] == 0:
            raise ValueError("state_jacobian must be nonempty")
        if nuisance_jacobian.shape[0] != state_jacobian.shape[0]:
            raise ValueError(
                "nuisance_jacobian must have the same observation rows as "
                "state_jacobian"
            )
        if covariance.shape != (state_jacobian.shape[0], state_jacobian.shape[0]):
            raise ValueError(
                "observation_covariance must have one row and column per "
                "prospective observation coordinate"
            )
        reliability = _reliability_vector(
            self.reliability,
            state_jacobian.shape[0],
        )
        cost = _finite_nonnegative(self.dimensionless_cost, name="dimensionless_cost")
        risk = _finite_nonnegative(self.dimensionless_risk, name="dimensionless_risk")
        safety = genuine_boolean(self.safety_admissible, name="safety_admissible")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="query-identifying action metadata",
        )
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "state_jacobian", _immutable_float64(state_jacobian))
        object.__setattr__(
            self,
            "nuisance_jacobian",
            _immutable_float64(nuisance_jacobian),
        )
        object.__setattr__(
            self,
            "observation_covariance",
            _immutable_float64(covariance),
        )
        object.__setattr__(self, "reliability", _immutable_float64(reliability))
        object.__setattr__(self, "dimensionless_cost", cost)
        object.__setattr__(self, "dimensionless_risk", risk)
        object.__setattr__(self, "safety_admissible", safety)
        object.__setattr__(self, "metadata", metadata)

        expected_id = cast(str, content_id(self.descriptor()))
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = cast(
                str,
                literal_lower_hex(
                    supplied_id,
                    name="artifact_id",
                    lengths={64},
                ),
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "query-identifying action artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def observation_dimension(self) -> int:
        return int(self.state_jacobian.shape[0])

    @property
    def state_dimension(self) -> int:
        return int(self.state_jacobian.shape[1])

    @property
    def nuisance_dimension(self) -> int:
        return int(self.nuisance_jacobian.shape[1])

    @property
    def reliability_vector(self) -> np.ndarray:
        return cast(np.ndarray, self.reliability)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_IDENTIFYING_ACTION_SCHEMA,
            "schema_version": QUERY_IDENTIFYING_ACTION_VERSION,
            "action_id": self.action_id,
            "state_jacobian": _array_record(self.state_jacobian),
            "nuisance_jacobian": _array_record(self.nuisance_jacobian),
            "observation_covariance": _array_record(self.observation_covariance),
            "reliability": _array_record(self.reliability_vector),
            "dimensionless_cost": self.dimensionless_cost,
            "dimensionless_risk": self.dimensionless_risk,
            "safety_admissible": self.safety_admissible,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class QueryIdentifyingActionEvaluationV1:
    """Content-bound query contraction and selection score for one action."""

    candidate: QueryIdentifyingActionCandidateV1
    status: QueryIdentifyingActionStatus
    posterior_query_covariance: np.ndarray = field(repr=False)
    ideal_posterior_query_covariance: np.ndarray = field(repr=False)
    normalized_query_covariance: np.ndarray = field(repr=False)
    normalized_ideal_query_covariance: np.ndarray = field(repr=False)
    baseline_normalized_query_trace: float
    posterior_normalized_query_trace: float
    ideal_normalized_query_trace: float
    query_trace_reduction: float
    ideal_query_trace_reduction: float
    nuisance_trace_effect: float
    normalized_query_maximum_eigenvalue: float
    marginal_state_information_gain_nats: float
    dimensionless_objective: float
    objective_improvement: float
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, QueryIdentifyingActionCandidateV1):
            raise TypeError("candidate must be QueryIdentifyingActionCandidateV1")
        if not isinstance(self.status, QueryIdentifyingActionStatus):
            raise TypeError("status must be QueryIdentifyingActionStatus")
        for name in (
            "posterior_query_covariance",
            "ideal_posterior_query_covariance",
            "normalized_query_covariance",
            "normalized_ideal_query_covariance",
        ):
            value = _exact_symmetric(getattr(self, name), name=name)
            object.__setattr__(self, name, _immutable_float64(value))
        for name in (
            "baseline_normalized_query_trace",
            "posterior_normalized_query_trace",
            "ideal_normalized_query_trace",
            "query_trace_reduction",
            "ideal_query_trace_reduction",
            "normalized_query_maximum_eigenvalue",
            "marginal_state_information_gain_nats",
            "dimensionless_objective",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        nuisance_effect = self.nuisance_trace_effect
        if isinstance(nuisance_effect, (bool, np.bool_)) or not isinstance(
            nuisance_effect,
            Real,
        ):
            raise ValueError("nuisance_trace_effect must be a finite real number")
        nuisance_effect_float = float(nuisance_effect)
        if not np.isfinite(nuisance_effect_float):
            raise ValueError("nuisance_trace_effect must be a finite real number")
        object.__setattr__(
            self,
            "nuisance_trace_effect",
            nuisance_effect_float,
        )

        improvement = self.objective_improvement
        if isinstance(improvement, (bool, np.bool_)) or not isinstance(
            improvement,
            Real,
        ):
            raise ValueError("objective_improvement must be a finite real number")
        improvement_float = float(improvement)
        if not np.isfinite(improvement_float):
            raise ValueError("objective_improvement must be a finite real number")
        object.__setattr__(self, "objective_improvement", improvement_float)

        expected_id = cast(str, content_id(self.descriptor()))
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = cast(
                str,
                literal_lower_hex(
                    supplied_id,
                    name="artifact_id",
                    lengths={64},
                ),
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "query-identifying evaluation artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_IDENTIFYING_ACTION_EVALUATION_SCHEMA,
            "schema_version": QUERY_IDENTIFYING_ACTION_EVALUATION_VERSION,
            "candidate_id": self.candidate.artifact_id,
            "action_id": self.candidate.action_id,
            "status": self.status.value,
            "posterior_query_covariance": _array_record(
                self.posterior_query_covariance
            ),
            "ideal_posterior_query_covariance": _array_record(
                self.ideal_posterior_query_covariance
            ),
            "normalized_query_covariance": _array_record(
                self.normalized_query_covariance
            ),
            "normalized_ideal_query_covariance": _array_record(
                self.normalized_ideal_query_covariance
            ),
            "baseline_normalized_query_trace": (
                self.baseline_normalized_query_trace
            ),
            "posterior_normalized_query_trace": (
                self.posterior_normalized_query_trace
            ),
            "ideal_normalized_query_trace": self.ideal_normalized_query_trace,
            "query_trace_reduction": self.query_trace_reduction,
            "ideal_query_trace_reduction": self.ideal_query_trace_reduction,
            "nuisance_trace_effect": self.nuisance_trace_effect,
            "normalized_query_maximum_eigenvalue": (
                self.normalized_query_maximum_eigenvalue
            ),
            "marginal_state_information_gain_nats": (
                self.marginal_state_information_gain_nats
            ),
            "dimensionless_objective": self.dimensionless_objective,
            "objective_improvement": self.objective_improvement,
        }

    def summary(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "candidate_id": self.candidate.artifact_id,
            "action_id": self.candidate.action_id,
            "status": self.status.value,
            "query_trace_reduction": self.query_trace_reduction,
            "nuisance_trace_effect": self.nuisance_trace_effect,
            "marginal_state_information_gain_nats": (
                self.marginal_state_information_gain_nats
            ),
            "dimensionless_objective": self.dimensionless_objective,
            "objective_improvement": self.objective_improvement,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


__all__ = [
    "QUERY_IDENTIFYING_ACTION_DESIGN_CLAIM_BOUNDARY",
    "QUERY_IDENTIFYING_ACTION_DESIGN_SCHEMA",
    "QUERY_IDENTIFYING_ACTION_DESIGN_SEMANTICS",
    "QUERY_IDENTIFYING_ACTION_DESIGN_VERSION",
    "QUERY_IDENTIFYING_ACTION_EVALUATION_SCHEMA",
    "QUERY_IDENTIFYING_ACTION_EVALUATION_VERSION",
    "QUERY_IDENTIFYING_ACTION_SCHEMA",
    "QUERY_IDENTIFYING_ACTION_VERSION",
    "QueryIdentifyingActionCandidateV1",
    "QueryIdentifyingActionEvaluationV1",
    "QueryIdentifyingActionStatus",
    "QueryIdentifyingDesignStatus",
]
