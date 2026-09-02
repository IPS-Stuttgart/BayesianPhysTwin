"""Robust, open-set interventional attribution of physical-twin error.

A nominal interventional certificate can distinguish registered cause signatures
when those signatures are treated as exact. Real signatures are estimated. This
module turns deterministic bounds on signature error, coefficient magnitude, and
observation noise into:

* a finite error bound for every identifiable cause query;
* an explicit ``unregistered_cause`` result when the registered family cannot
  explain the stacked intervention response within its declared uncertainty; and
* minimum-cardinality and minimum-cost intervention sets computed before outcome
  access.

The result is local and finite-family. Passing the family-closure test means only
that the registered family was not falsified at the supplied uncertainty radius.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Final

import numpy as np

ROBUST_ATTRIBUTION_SCHEMA: Final = (
    "bayesian_phystwin.robust_interventional_cause_attribution"
)
ROBUST_ATTRIBUTION_VERSION: Final = 1
ROBUST_ATTRIBUTION_SEMANTICS: Final = (
    "pre-outcome-bounded-signature-attribution-with-open-set-closure-v1"
)
ROBUST_ATTRIBUTION_CLAIM_BOUNDARY: Final = (
    "A passing decision bounds one registered cause query under the exact nominal "
    "intervention signatures, deterministic signature-error budgets, coefficient "
    "norm bounds, observation-noise radii, whitening, and query tolerance. A "
    "failed closure test rejects the registered family. A passed closure test "
    "does not prove family completeness, unique physical causation, nonlinear "
    "validity, unseen-object transfer, safe control, or deployment safety."
)


class RobustAttributionStatus(str, Enum):
    """Fail-closed attribution states for one registered cause query."""

    ROBUSTLY_ATTRIBUTABLE = "robustly_attributable"
    IDENTIFIABLE_BUT_UNSTABLE = "identifiable_but_unstable"
    CONFOUNDED = "confounded"
    TRIVIAL_QUERY = "trivial_query"
    UNREGISTERED_CAUSE = "unregistered_cause"


def _literal(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _matrix(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector")
    return result


def _immutable(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )


def _freeze_json(value: object, name: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Real):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must contain finite JSON values")
        return result
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{name} keys must be strings")
            output[key] = _freeze_json(item, name)
        return MappingProxyType(output)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item, name) for item in value)
    raise ValueError(f"{name} must contain finite JSON values")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _content_id(value: object) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _basis(design: np.ndarray, relative: float, absolute: float) -> np.ndarray:
    if design.shape[1] == 0:
        return np.empty((design.shape[0], 0), dtype=np.float64)
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    leading = float(singular[0]) if len(singular) else 0.0
    tolerance = max(absolute, relative * leading)
    return left[:, singular > tolerance]


def _complement(design: np.ndarray, relative: float, absolute: float) -> np.ndarray:
    basis = _basis(design, relative, absolute)
    return np.eye(design.shape[0], dtype=np.float64) - basis @ basis.T


def _pinv(design: np.ndarray, relative: float, absolute: float) -> np.ndarray:
    left, singular, right_t = np.linalg.svd(design, full_matrices=False)
    leading = float(singular[0]) if len(singular) else 0.0
    tolerance = max(absolute, relative * leading)
    inverse = np.zeros_like(singular)
    retained = singular > tolerance
    inverse[retained] = 1.0 / singular[retained]
    return (right_t.T * inverse) @ left.T


def _stack_bound(values: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _operator_norm(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(np.linalg.norm(value, ord=2))


@dataclass(frozen=True, slots=True)
class RobustCauseModelV1:
    """One registered cause response and one task-facing cause query."""

    cause_id: str
    intervention_ids: Sequence[str]
    response_blocks: Sequence[np.ndarray]
    query_map: np.ndarray
    signature_error_bounds: Sequence[float]
    coefficient_norm_bound: float
    query_error_tolerance: float
    minimum_effect_norm: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cause_id = _literal(self.cause_id, "cause_id")
        ids = tuple(
            _literal(value, "intervention_id") for value in self.intervention_ids
        )
        if (
            len(ids) < 2
            or ids != tuple(sorted(ids))
            or len(ids) != len(set(ids))
        ):
            raise ValueError(
                "intervention_ids must be sorted, unique, and contain at least "
                "two values"
            )
        blocks = tuple(
            _matrix(value, "response_block") for value in self.response_blocks
        )
        if len(blocks) != len(ids) or any(
            block.shape[0] == 0 or block.shape[1] == 0 for block in blocks
        ):
            raise ValueError(
                "one nonempty response block is required per intervention"
            )
        latent = blocks[0].shape[1]
        if any(block.shape[1] != latent for block in blocks):
            raise ValueError("response blocks must share the latent dimension")
        query = _matrix(self.query_map, "query_map")
        if query.shape[0] == 0 or query.shape[1] != latent:
            raise ValueError(
                "query_map must have one column per cause coordinate"
            )
        errors = tuple(
            _finite_nonnegative(value, "signature_error_bound")
            for value in self.signature_error_bounds
        )
        if len(errors) != len(ids):
            raise ValueError(
                "one signature error bound is required per intervention"
            )
        metadata = _freeze_json(self.metadata, "metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "cause_id", cause_id)
        object.__setattr__(self, "intervention_ids", ids)
        object.__setattr__(
            self,
            "response_blocks",
            tuple(_immutable(value) for value in blocks),
        )
        object.__setattr__(self, "query_map", _immutable(query))
        object.__setattr__(self, "signature_error_bounds", errors)
        object.__setattr__(
            self,
            "coefficient_norm_bound",
            _finite_nonnegative(
                self.coefficient_norm_bound,
                "coefficient_norm_bound",
            ),
        )
        object.__setattr__(
            self,
            "query_error_tolerance",
            _finite_nonnegative(
                self.query_error_tolerance,
                "query_error_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "minimum_effect_norm",
            _finite_nonnegative(
                self.minimum_effect_norm,
                "minimum_effect_norm",
            ),
        )
        object.__setattr__(self, "metadata", metadata)

    @property
    def observation_dimensions(self) -> tuple[int, ...]:
        return tuple(int(block.shape[0]) for block in self.response_blocks)

    def descriptor(self) -> dict[str, object]:
        return {
            "cause_id": self.cause_id,
            "intervention_ids": list(self.intervention_ids),
            "response_blocks": [
                _array_record(value) for value in self.response_blocks
            ],
            "query_map": _array_record(self.query_map),
            "signature_error_bounds": list(self.signature_error_bounds),
            "coefficient_norm_bound": self.coefficient_norm_bound,
            "query_error_tolerance": self.query_error_tolerance,
            "minimum_effect_norm": self.minimum_effect_norm,
            "metadata": _plain(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RobustObservationDesignV1:
    """Shared nuisance, noise, and cost contract for registered interventions."""

    intervention_ids: Sequence[str]
    nuisance_blocks: Sequence[np.ndarray]
    observation_noise_radii: Sequence[float]
    nuisance_signature_error_bounds: Sequence[float]
    nuisance_coefficient_norm_bound: float
    intervention_costs: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(
            _literal(value, "intervention_id") for value in self.intervention_ids
        )
        if (
            len(ids) < 2
            or ids != tuple(sorted(ids))
            or len(ids) != len(set(ids))
        ):
            raise ValueError(
                "intervention_ids must be sorted, unique, and contain at least "
                "two values"
            )
        blocks = tuple(
            _matrix(value, "nuisance_block") for value in self.nuisance_blocks
        )
        if len(blocks) != len(ids) or any(
            block.shape[0] == 0 for block in blocks
        ):
            raise ValueError(
                "one nuisance block with positive row count is required per "
                "intervention"
            )
        nuisance_dimension = blocks[0].shape[1]
        if any(block.shape[1] != nuisance_dimension for block in blocks):
            raise ValueError("nuisance blocks must share the nuisance dimension")
        noise = tuple(
            _finite_nonnegative(value, "observation_noise_radius")
            for value in self.observation_noise_radii
        )
        nuisance_errors = tuple(
            _finite_nonnegative(value, "nuisance_signature_error_bound")
            for value in self.nuisance_signature_error_bounds
        )
        costs = tuple(
            _finite_nonnegative(value, "intervention_cost")
            for value in self.intervention_costs
        )
        if len(noise) != len(ids):
            raise ValueError(
                "one observation noise radius is required per intervention"
            )
        if len(nuisance_errors) != len(ids):
            raise ValueError(
                "one nuisance signature error bound is required per intervention"
            )
        if len(costs) != len(ids):
            raise ValueError("one intervention cost is required per intervention")
        metadata = _freeze_json(self.metadata, "metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "intervention_ids", ids)
        object.__setattr__(
            self,
            "nuisance_blocks",
            tuple(_immutable(value) for value in blocks),
        )
        object.__setattr__(self, "observation_noise_radii", noise)
        object.__setattr__(
            self,
            "nuisance_signature_error_bounds",
            nuisance_errors,
        )
        object.__setattr__(
            self,
            "nuisance_coefficient_norm_bound",
            _finite_nonnegative(
                self.nuisance_coefficient_norm_bound,
                "nuisance_coefficient_norm_bound",
            ),
        )
        object.__setattr__(self, "intervention_costs", costs)
        object.__setattr__(self, "metadata", metadata)

    @property
    def observation_dimensions(self) -> tuple[int, ...]:
        return tuple(int(block.shape[0]) for block in self.nuisance_blocks)

    def descriptor(self) -> dict[str, object]:
        return {
            "intervention_ids": list(self.intervention_ids),
            "nuisance_blocks": [
                _array_record(value) for value in self.nuisance_blocks
            ],
            "observation_noise_radii": list(self.observation_noise_radii),
            "nuisance_signature_error_bounds": list(
                self.nuisance_signature_error_bounds
            ),
            "nuisance_coefficient_norm_bound": (
                self.nuisance_coefficient_norm_bound
            ),
            "intervention_costs": list(self.intervention_costs),
            "metadata": _plain(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RobustInterventionSetResultV1:
    """Pre-outcome robust identifiability result for one intervention subset."""

    intervention_ids: tuple[str, ...]
    status: RobustAttributionStatus
    nominally_identifiable: bool
    query_reconstruction_residual: float
    query_operator_norm: float
    effective_response_error_bound: float
    query_error_bound: float
    intervention_cost: float
    query_operator: np.ndarray = field(repr=False)
    competitor_projector: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_operator", _immutable(self.query_operator))
        object.__setattr__(
            self,
            "competitor_projector",
            _immutable(self.competitor_projector),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "intervention_ids": list(self.intervention_ids),
            "status": self.status.value,
            "nominally_identifiable": self.nominally_identifiable,
            "query_reconstruction_residual": (
                self.query_reconstruction_residual
            ),
            "query_operator_norm": self.query_operator_norm,
            "effective_response_error_bound": (
                self.effective_response_error_bound
            ),
            "query_error_bound": self.query_error_bound,
            "intervention_cost": self.intervention_cost,
        }


@dataclass(frozen=True, slots=True)
class RobustCausePlanV1:
    """Outcome-independent intervention plan for one cause query."""

    cause_id: str
    full_intervention_result: RobustInterventionSetResultV1
    intervention_set_results: tuple[RobustInterventionSetResultV1, ...]
    minimum_robust_intervention_count: int | None
    minimal_robust_intervention_sets: tuple[tuple[str, ...], ...]
    minimum_robust_intervention_cost: float | None
    minimum_cost_robust_intervention_sets: tuple[tuple[str, ...], ...]

    @property
    def full_status(self) -> RobustAttributionStatus:
        return self.full_intervention_result.status

    @property
    def query_error_bound(self) -> float:
        return self.full_intervention_result.query_error_bound

    def to_record(self) -> dict[str, object]:
        return {
            "cause_id": self.cause_id,
            "full_status": self.full_status.value,
            "full_query_error_bound": self.query_error_bound,
            "minimum_robust_intervention_count": (
                self.minimum_robust_intervention_count
            ),
            "minimal_robust_intervention_sets": [
                list(value) for value in self.minimal_robust_intervention_sets
            ],
            "minimum_robust_intervention_cost": (
                self.minimum_robust_intervention_cost
            ),
            "minimum_cost_robust_intervention_sets": [
                list(value)
                for value in self.minimum_cost_robust_intervention_sets
            ],
            "intervention_set_results": [
                value.to_record() for value in self.intervention_set_results
            ],
        }


@dataclass(frozen=True, slots=True)
class RobustCauseDecisionV1:
    """Observed robust attribution result for one cause query."""

    cause_id: str
    status: RobustAttributionStatus
    query_estimate: np.ndarray
    query_error_bound: float
    effect_present_certified: bool
    effect_absent_certified: bool
    minimum_effect_norm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_estimate", _immutable(self.query_estimate))

    def to_record(self) -> dict[str, object]:
        return {
            "cause_id": self.cause_id,
            "status": self.status.value,
            "query_estimate": self.query_estimate.tolist(),
            "query_error_bound": self.query_error_bound,
            "effect_present_certified": self.effect_present_certified,
            "effect_absent_certified": self.effect_absent_certified,
            "minimum_effect_norm": self.minimum_effect_norm,
        }


@dataclass(frozen=True, slots=True)
class RobustAttributionDecisionV1:
    """Complete open-set decision after reading registered response blocks."""

    plan_id: str
    cause_family_id: str
    cause_decisions: tuple[RobustCauseDecisionV1, ...]
    registered_family_falsified: bool
    family_closure_residual_norm: float
    family_closure_error_bound: float
    claim_boundary: str = ROBUST_ATTRIBUTION_CLAIM_BOUNDARY

    def result_for(self, cause_id: str) -> RobustCauseDecisionV1:
        requested = _literal(cause_id, "cause_id")
        for result in self.cause_decisions:
            if result.cause_id == requested:
                return result
        raise KeyError(requested)

    def to_record(self) -> dict[str, object]:
        record = {
            "schema": ROBUST_ATTRIBUTION_SCHEMA,
            "schema_version": ROBUST_ATTRIBUTION_VERSION,
            "semantics": ROBUST_ATTRIBUTION_SEMANTICS,
            "plan_id": self.plan_id,
            "cause_family_id": self.cause_family_id,
            "registered_family_falsified": (
                self.registered_family_falsified
            ),
            "family_closure_residual_norm": (
                self.family_closure_residual_norm
            ),
            "family_closure_error_bound": self.family_closure_error_bound,
            "cause_decisions": [
                value.to_record() for value in self.cause_decisions
            ],
            "claim_boundary": self.claim_boundary,
        }
        record["decision_id"] = _content_id(record)
        return record


@dataclass(frozen=True, slots=True)
class RobustAttributionPlanV1:
    """Frozen attribution, open-set, and intervention-design contract."""

    observation_design: RobustObservationDesignV1
    cause_models: Sequence[RobustCauseModelV1]
    cause_family_id: str
    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 1e-12
    maximum_intervention_subsets: int = 4095
    cause_plans: tuple[RobustCausePlanV1, ...] = field(
        init=False,
        repr=False,
    )
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation_design, RobustObservationDesignV1):
            raise TypeError(
                "observation_design must be RobustObservationDesignV1"
            )
        family_id = _literal(self.cause_family_id, "cause_family_id")
        causes = tuple(self.cause_models)
        if not causes or any(
            not isinstance(value, RobustCauseModelV1) for value in causes
        ):
            raise TypeError(
                "cause_models must contain RobustCauseModelV1 values"
            )
        cause_ids = tuple(value.cause_id for value in causes)
        if cause_ids != tuple(sorted(cause_ids)) or len(cause_ids) != len(
            set(cause_ids)
        ):
            raise ValueError("cause_models must be sorted and unique by cause_id")
        design = self.observation_design
        for cause in causes:
            if cause.intervention_ids != design.intervention_ids:
                raise ValueError(
                    "every cause model must use the observation intervention_ids"
                )
            if cause.observation_dimensions != design.observation_dimensions:
                raise ValueError(
                    "cause and nuisance blocks must share observation dimensions"
                )
        relative = _positive(self.relative_tolerance, "relative_tolerance")
        absolute = _positive(self.absolute_tolerance, "absolute_tolerance")
        maximum = _positive_integer(
            self.maximum_intervention_subsets,
            "maximum_intervention_subsets",
        )
        subset_count = (1 << len(design.intervention_ids)) - 1
        if subset_count > maximum:
            raise ValueError(
                "registered intervention set exceeds maximum_intervention_subsets"
            )
        object.__setattr__(self, "cause_models", causes)
        object.__setattr__(self, "cause_family_id", family_id)
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "absolute_tolerance", absolute)
        object.__setattr__(self, "maximum_intervention_subsets", maximum)
        plans = tuple(self._build_cause_plan(cause) for cause in causes)
        object.__setattr__(self, "cause_plans", plans)
        descriptor = {
            "schema": ROBUST_ATTRIBUTION_SCHEMA,
            "schema_version": ROBUST_ATTRIBUTION_VERSION,
            "semantics": ROBUST_ATTRIBUTION_SEMANTICS,
            "cause_family_id": family_id,
            "observation_design": design.descriptor(),
            "cause_models": [value.descriptor() for value in causes],
            "relative_tolerance": relative,
            "absolute_tolerance": absolute,
            "maximum_intervention_subsets": maximum,
        }
        object.__setattr__(self, "plan_id", _content_id(descriptor))

    def _indices(self, intervention_ids: tuple[str, ...]) -> tuple[int, ...]:
        lookup = {
            value: index
            for index, value in enumerate(
                self.observation_design.intervention_ids
            )
        }
        return tuple(lookup[value] for value in intervention_ids)

    def _stack_cause(
        self,
        cause: RobustCauseModelV1,
        indices: tuple[int, ...],
    ) -> np.ndarray:
        return np.vstack(tuple(cause.response_blocks[index] for index in indices))

    def _stack_nuisance(self, indices: tuple[int, ...]) -> np.ndarray:
        return np.vstack(
            tuple(
                self.observation_design.nuisance_blocks[index]
                for index in indices
            )
        )

    def _effective_error_bound(self, indices: tuple[int, ...]) -> float:
        design = self.observation_design
        total = _stack_bound(
            tuple(design.observation_noise_radii[index] for index in indices)
        )
        total += design.nuisance_coefficient_norm_bound * _stack_bound(
            tuple(
                design.nuisance_signature_error_bounds[index]
                for index in indices
            )
        )
        for cause in self.cause_models:
            total += cause.coefficient_norm_bound * _stack_bound(
                tuple(cause.signature_error_bounds[index] for index in indices)
            )
        return float(total)

    def _subset_result(
        self,
        target: RobustCauseModelV1,
        intervention_ids: tuple[str, ...],
    ) -> RobustInterventionSetResultV1:
        indices = self._indices(intervention_ids)
        target_design = self._stack_cause(target, indices)
        competitors = [self._stack_nuisance(indices)]
        competitors.extend(
            self._stack_cause(cause, indices)
            for cause in self.cause_models
            if cause.cause_id != target.cause_id
        )
        competitor_design = np.column_stack(competitors)
        projector = _complement(
            competitor_design,
            self.relative_tolerance,
            self.absolute_tolerance,
        )
        residualized = projector @ target_design
        query_norm = _operator_norm(target.query_map)
        if query_norm <= self.absolute_tolerance:
            status = RobustAttributionStatus.TRIVIAL_QUERY
            operator = np.zeros(
                (target.query_map.shape[0], residualized.shape[0]),
                dtype=np.float64,
            )
            reconstruction_residual = 0.0
            nominally_identifiable = True
        else:
            operator = target.query_map @ _pinv(
                residualized,
                self.relative_tolerance,
                self.absolute_tolerance,
            )
            reconstruction = operator @ residualized
            reconstruction_residual = _operator_norm(
                target.query_map - reconstruction
            )
            threshold = self.absolute_tolerance + (
                self.relative_tolerance * max(1.0, query_norm)
            )
            nominally_identifiable = reconstruction_residual <= threshold
            if nominally_identifiable:
                status = RobustAttributionStatus.ROBUSTLY_ATTRIBUTABLE
            else:
                status = RobustAttributionStatus.CONFOUNDED
        effective = self._effective_error_bound(indices)
        query_error = _operator_norm(operator) * effective
        if (
            status is RobustAttributionStatus.ROBUSTLY_ATTRIBUTABLE
            and query_error
            > target.query_error_tolerance
            + self.absolute_tolerance
            + self.relative_tolerance * max(1.0, target.query_error_tolerance)
        ):
            status = RobustAttributionStatus.IDENTIFIABLE_BUT_UNSTABLE
        cost_lookup = dict(
            zip(
                self.observation_design.intervention_ids,
                self.observation_design.intervention_costs,
                strict=True,
            )
        )
        return RobustInterventionSetResultV1(
            intervention_ids=intervention_ids,
            status=status,
            nominally_identifiable=nominally_identifiable,
            query_reconstruction_residual=reconstruction_residual,
            query_operator_norm=_operator_norm(operator),
            effective_response_error_bound=effective,
            query_error_bound=query_error,
            intervention_cost=float(
                sum(cost_lookup[value] for value in intervention_ids)
            ),
            query_operator=operator,
            competitor_projector=projector,
        )

    def _build_cause_plan(
        self,
        target: RobustCauseModelV1,
    ) -> RobustCausePlanV1:
        ids = self.observation_design.intervention_ids
        results = tuple(
            self._subset_result(target, subset)
            for count in range(1, len(ids) + 1)
            for subset in itertools.combinations(ids, count)
        )
        robust = tuple(
            value
            for value in results
            if value.status is RobustAttributionStatus.ROBUSTLY_ATTRIBUTABLE
        )
        if robust:
            minimum_count = min(len(value.intervention_ids) for value in robust)
            minimal_sets = tuple(
                value.intervention_ids
                for value in robust
                if len(value.intervention_ids) == minimum_count
            )
            minimum_cost = min(value.intervention_cost for value in robust)
            cost_tolerance = self.absolute_tolerance + (
                self.relative_tolerance * max(1.0, minimum_cost)
            )
            minimum_cost_sets = tuple(
                value.intervention_ids
                for value in robust
                if abs(value.intervention_cost - minimum_cost)
                <= cost_tolerance
            )
        else:
            minimum_count = None
            minimal_sets = ()
            minimum_cost = None
            minimum_cost_sets = ()
        full_result = next(
            value for value in results if value.intervention_ids == ids
        )
        return RobustCausePlanV1(
            cause_id=target.cause_id,
            full_intervention_result=full_result,
            intervention_set_results=results,
            minimum_robust_intervention_count=minimum_count,
            minimal_robust_intervention_sets=minimal_sets,
            minimum_robust_intervention_cost=minimum_cost,
            minimum_cost_robust_intervention_sets=minimum_cost_sets,
        )

    def result_for(self, cause_id: str) -> RobustCausePlanV1:
        requested = _literal(cause_id, "cause_id")
        for result in self.cause_plans:
            if result.cause_id == requested:
                return result
        raise KeyError(requested)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": ROBUST_ATTRIBUTION_SCHEMA,
            "schema_version": ROBUST_ATTRIBUTION_VERSION,
            "semantics": ROBUST_ATTRIBUTION_SEMANTICS,
            "plan_id": self.plan_id,
            "cause_family_id": self.cause_family_id,
            "observation_design": self.observation_design.descriptor(),
            "cause_models": [
                value.descriptor() for value in self.cause_models
            ],
            "cause_plans": [value.to_record() for value in self.cause_plans],
            "claim_boundary": ROBUST_ATTRIBUTION_CLAIM_BOUNDARY,
        }

    def _observations(
        self,
        observation_blocks: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        blocks = tuple(
            _vector(value, "observation_block") for value in observation_blocks
        )
        if len(blocks) != len(self.observation_design.intervention_ids):
            raise ValueError(
                "one observation block is required per registered intervention"
            )
        for block, expected in zip(
            blocks,
            self.observation_design.observation_dimensions,
            strict=True,
        ):
            if block.shape != (expected,):
                raise ValueError(
                    "observation block dimension does not match registered design"
                )
        return blocks

    def evaluate(
        self,
        observation_blocks: Sequence[np.ndarray],
    ) -> RobustAttributionDecisionV1:
        blocks = self._observations(observation_blocks)
        observation = np.concatenate(blocks)
        indices = tuple(range(len(blocks)))
        family_columns = [self._stack_nuisance(indices)]
        family_columns.extend(
            self._stack_cause(cause, indices) for cause in self.cause_models
        )
        family_design = np.column_stack(family_columns)
        family_projector = _complement(
            family_design,
            self.relative_tolerance,
            self.absolute_tolerance,
        )
        closure_residual = float(
            np.linalg.norm(family_projector @ observation)
        )
        closure_bound = self._effective_error_bound(indices)
        closure_tolerance = self.absolute_tolerance + (
            self.relative_tolerance * max(1.0, float(np.linalg.norm(observation)))
        )
        falsified = closure_residual > closure_bound + closure_tolerance
        decisions: list[RobustCauseDecisionV1] = []
        for cause, plan in zip(self.cause_models, self.cause_plans, strict=True):
            full = plan.full_intervention_result
            estimate = full.query_operator @ (
                full.competitor_projector @ observation
            )
            status = (
                RobustAttributionStatus.UNREGISTERED_CAUSE
                if falsified
                else full.status
            )
            estimate_norm = float(np.linalg.norm(estimate))
            lower = max(0.0, estimate_norm - full.query_error_bound)
            upper = estimate_norm + full.query_error_bound
            present = (
                status is RobustAttributionStatus.ROBUSTLY_ATTRIBUTABLE
                and lower >= cause.minimum_effect_norm
            )
            absent = (
                status is RobustAttributionStatus.ROBUSTLY_ATTRIBUTABLE
                and upper < cause.minimum_effect_norm
            )
            decisions.append(
                RobustCauseDecisionV1(
                    cause_id=cause.cause_id,
                    status=status,
                    query_estimate=estimate,
                    query_error_bound=full.query_error_bound,
                    effect_present_certified=present,
                    effect_absent_certified=absent,
                    minimum_effect_norm=cause.minimum_effect_norm,
                )
            )
        return RobustAttributionDecisionV1(
            plan_id=self.plan_id,
            cause_family_id=self.cause_family_id,
            cause_decisions=tuple(decisions),
            registered_family_falsified=falsified,
            family_closure_residual_norm=closure_residual,
            family_closure_error_bound=closure_bound,
        )
