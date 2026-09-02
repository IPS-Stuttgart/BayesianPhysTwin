"""Fail-closed tiered transport for learned physical corrections.

A learned correction is not one indivisible transferable object.  Depending on
what is justified before target outcomes are opened, a physical twin may reuse
exact coefficients, a query-identifiable effect, a low-dimensional
recalibration, uncertainty structure only, or merely the fitting procedure.

This module chooses the strongest justified tier for one registered physical
query and finite affine-loss action portfolio.  Deterministic mean-transport
tiers must additionally identify a unique action whose worst-case regret is
within the supplied tolerance for every query error in a Euclidean ball.
Belief-only and procedure-only tiers preserve the exact caller-owned fallback
action.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final

import numpy as np

TRANSPORT4D_SCHEMA: Final = "bayesian_phystwin.transport4d_tiered_certificate"
TRANSPORT4D_VERSION: Final = 1
TRANSPORT4D_SEMANTICS: Final = (
    "highest-source-justified-query-conditional-transport-tier-v1"
)
TRANSPORT4D_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied tier candidates, evidence "
    "checks, query effects, Euclidean error radii, affine action losses, and "
    "regret tolerance. It does not validate a physical transformation, infer "
    "an error radius, establish exchangeability, prove nonlinear closure, "
    "authorize target-data access, certify deployment safety, or establish "
    "state of the art."
)


class TransportTier(str, Enum):
    """Descending specificity of a transported learned correction."""

    EXACT_COEFFICIENTS = "exact_coefficients"
    QUERY_IDENTIFIABLE_EFFECT = "query_identifiable_effect"
    LOW_DIMENSIONAL_CORRECTION = "low_dimensional_correction"
    UNCERTAINTY_ONLY = "uncertainty_only"
    PROCEDURE_ONLY = "procedure_only"


TRANSPORT_TIER_ORDER: Final = (
    TransportTier.EXACT_COEFFICIENTS,
    TransportTier.QUERY_IDENTIFIABLE_EFFECT,
    TransportTier.LOW_DIMENSIONAL_CORRECTION,
    TransportTier.UNCERTAINTY_ONLY,
    TransportTier.PROCEDURE_ONLY,
)
_TIER_RANK: Final = {tier: index for index, tier in enumerate(TRANSPORT_TIER_ORDER)}


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _name(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def _matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] == 0
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return result


def _immutable(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    copied = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    if not isinstance(copied, dict):
        raise TypeError("metadata must encode to a JSON object")
    return copied


def _canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TransportCandidateV1:
    """One source-frozen candidate in the transportability ladder."""

    tier: TransportTier
    evidence_id: str
    checks: Mapping[str, bool]
    target_outcome_blind: bool
    adaptation_dimension: int
    transports_mean: bool
    transports_uncertainty: bool
    query_effect: np.ndarray | None = None
    query_error_radius: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        try:
            tier = TransportTier(self.tier)
        except ValueError as error:
            raise ValueError("unknown transport tier") from error
        evidence_id = _digest(self.evidence_id, name="evidence_id")
        if type(self.target_outcome_blind) is not bool:
            raise ValueError("target_outcome_blind must be a bool")
        if isinstance(self.adaptation_dimension, bool) or not isinstance(
            self.adaptation_dimension, int
        ):
            raise ValueError("adaptation_dimension must be a nonnegative integer")
        if self.adaptation_dimension < 0:
            raise ValueError("adaptation_dimension must be a nonnegative integer")
        if type(self.transports_mean) is not bool:
            raise ValueError("transports_mean must be a bool")
        if type(self.transports_uncertainty) is not bool:
            raise ValueError("transports_uncertainty must be a bool")
        if not isinstance(self.checks, Mapping) or not self.checks:
            raise ValueError("checks must be a nonempty mapping")
        checks: dict[str, bool] = {}
        for raw_name, raw_value in self.checks.items():
            check_name = _name(raw_name, name="check name")
            if type(raw_value) is not bool:
                raise ValueError("every transport check must be a bool")
            checks[check_name] = raw_value
        if len(checks) != len(self.checks):
            raise ValueError("transport check names must be unique")

        mean_tier = tier in {
            TransportTier.EXACT_COEFFICIENTS,
            TransportTier.QUERY_IDENTIFIABLE_EFFECT,
            TransportTier.LOW_DIMENSIONAL_CORRECTION,
        }
        effect: np.ndarray | None = None
        radius: float | None = None
        if mean_tier:
            if not self.transports_mean:
                raise ValueError("mean transport tiers must transport a query mean")
            if self.query_effect is None or self.query_error_radius is None:
                raise ValueError("mean transport tiers require effect and error radius")
            effect = _immutable(_vector(self.query_effect, name="query_effect"))
            radius = _finite_nonnegative(
                self.query_error_radius,
                name="query_error_radius",
            )
        else:
            if self.transports_mean:
                raise ValueError("belief/procedure tiers cannot transport a mean")
            if self.query_effect is not None or self.query_error_radius is not None:
                raise ValueError(
                    "belief/procedure tiers must not carry deterministic query effects"
                )

        if (
            tier
            in {
                TransportTier.EXACT_COEFFICIENTS,
                TransportTier.QUERY_IDENTIFIABLE_EFFECT,
                TransportTier.UNCERTAINTY_ONLY,
            }
            and self.adaptation_dimension != 0
        ):
            raise ValueError(f"{tier.value} must use zero target-fit dimensions")
        if tier is TransportTier.LOW_DIMENSIONAL_CORRECTION:
            _positive_integer(
                self.adaptation_dimension,
                name="low-dimensional adaptation_dimension",
            )
        if tier is TransportTier.UNCERTAINTY_ONLY:
            if not self.transports_uncertainty:
                raise ValueError("uncertainty_only must transport uncertainty")
        if tier is TransportTier.PROCEDURE_ONLY:
            if self.transports_uncertainty:
                raise ValueError("procedure_only cannot transport target uncertainty")
            _positive_integer(
                self.adaptation_dimension,
                name="procedure-only adaptation_dimension",
            )

        metadata = _json_copy(self.metadata)
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "query_effect", effect)
        object.__setattr__(self, "query_error_radius", radius)
        object.__setattr__(self, "metadata", metadata)

        expected = _canonical_id(self.descriptor())
        supplied = self.candidate_id
        if supplied is not None:
            supplied = _digest(supplied, name="candidate_id")
            if supplied != expected:
                raise ValueError("candidate_id does not match candidate content")
        object.__setattr__(self, "candidate_id", expected)

    @property
    def checks_passed(self) -> bool:
        return all(self.checks.values())

    @property
    def structural_eligibility(self) -> bool:
        return self.target_outcome_blind and self.checks_passed

    @property
    def resolved_candidate_id(self) -> str:
        candidate_id = self.candidate_id
        if candidate_id is None:
            raise RuntimeError("candidate_id was not initialized")
        return candidate_id

    def descriptor(self) -> dict[str, object]:
        return {
            "tier": self.tier.value,
            "evidence_id": self.evidence_id,
            "checks": dict(sorted(self.checks.items())),
            "target_outcome_blind": self.target_outcome_blind,
            "adaptation_dimension": self.adaptation_dimension,
            "transports_mean": self.transports_mean,
            "transports_uncertainty": self.transports_uncertainty,
            "query_effect": (
                _array_record(self.query_effect)
                if self.query_effect is not None
                else None
            ),
            "query_error_radius": self.query_error_radius,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class TransportTierEvaluationV1:
    """Evaluation of one candidate against the registered query/action contract."""

    tier: TransportTier
    candidate_id: str
    structurally_eligible: bool
    failed_checks: tuple[str, ...]
    target_outcome_blind: bool
    action_certified: bool
    action_name: str | None
    minimax_regret_upper: float | None
    robust_regret_upper_by_action: tuple[float, ...]
    selectable: bool
    selected: bool
    reason_code: str

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier.value,
            "candidate_id": self.candidate_id,
            "structurally_eligible": self.structurally_eligible,
            "failed_checks": list(self.failed_checks),
            "target_outcome_blind": self.target_outcome_blind,
            "action_certified": self.action_certified,
            "action_name": self.action_name,
            "minimax_regret_upper": self.minimax_regret_upper,
            "robust_regret_upper_by_action": list(self.robust_regret_upper_by_action),
            "selectable": self.selectable,
            "selected": self.selected,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class _ProvisionalTierEvaluation:
    candidate: TransportCandidateV1
    structurally_eligible: bool
    failed_checks: tuple[str, ...]
    action_certified: bool
    action_name: str | None
    minimax_regret_upper: float | None
    robust_regret_upper_by_action: tuple[float, ...]
    selectable: bool
    reason_code: str


@dataclass(frozen=True, slots=True)
class TieredTransportCertificateV1:
    """Select the strongest justified transport tier for one physical query."""

    query_id: str
    query_contract_id: str
    baseline_belief_id: str
    action_portfolio_id: str
    baseline_query: np.ndarray
    action_names: Sequence[str]
    action_weights: np.ndarray
    action_offsets: np.ndarray
    fallback_action_name: str
    candidates: Sequence[TransportCandidateV1]
    regret_tolerance: float
    action_tie_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    candidate_order: tuple[TransportCandidateV1, ...] = field(init=False)
    tier_evaluations: tuple[TransportTierEvaluationV1, ...] = field(init=False)
    selected_tier: TransportTier | None = field(init=False)
    selected_candidate_id: str | None = field(init=False)
    selected_action_name: str = field(init=False)
    used_exact_fallback: bool = field(init=False)
    belief_transport_only: bool = field(init=False)

    def __post_init__(self) -> None:
        query_id = _name(self.query_id, name="query_id")
        query_contract_id = _digest(self.query_contract_id, name="query_contract_id")
        baseline_belief_id = _digest(
            self.baseline_belief_id,
            name="baseline_belief_id",
        )
        action_portfolio_id = _digest(
            self.action_portfolio_id,
            name="action_portfolio_id",
        )
        baseline = _immutable(_vector(self.baseline_query, name="baseline_query"))
        weights = _immutable(_matrix(self.action_weights, name="action_weights"))
        offsets = _immutable(_vector(self.action_offsets, name="action_offsets"))
        names = tuple(_name(item, name="action name") for item in self.action_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("action_names must contain at least two unique actions")
        if weights.shape != (len(names), baseline.size):
            raise ValueError(
                "action_weights must have shape (actions, query_dimension)"
            )
        if offsets.shape != (len(names),):
            raise ValueError("action_offsets must contain one value per action")
        fallback = _name(self.fallback_action_name, name="fallback_action_name")
        if fallback not in names:
            raise ValueError("fallback_action_name is absent from action_names")
        regret_tolerance = _finite_nonnegative(
            self.regret_tolerance,
            name="regret_tolerance",
        )
        tie_tolerance = _finite_nonnegative(
            self.action_tie_tolerance,
            name="action_tie_tolerance",
        )
        if tie_tolerance == 0.0:
            raise ValueError("action_tie_tolerance must be positive")

        raw_candidates = tuple(self.candidates)
        if not raw_candidates:
            raise ValueError("at least one transport candidate is required")
        if not all(isinstance(item, TransportCandidateV1) for item in raw_candidates):
            raise TypeError("candidates must contain TransportCandidateV1 values")
        tiers = [item.tier for item in raw_candidates]
        if len(tiers) != len(set(tiers)):
            raise ValueError("only one candidate per transport tier is supported")
        for item in raw_candidates:
            if (
                item.query_effect is not None
                and item.query_effect.shape != baseline.shape
            ):
                raise ValueError(
                    "candidate query effect dimension differs from baseline"
                )
        candidates = tuple(
            sorted(raw_candidates, key=lambda item: _TIER_RANK[item.tier])
        )
        metadata = _json_copy(self.metadata)

        provisional: list[_ProvisionalTierEvaluation] = []
        selected_candidate: TransportCandidateV1 | None = None
        selected_action_index: int | None = None
        for candidate in candidates:
            failed = tuple(
                sorted(name for name, passed in candidate.checks.items() if not passed)
            )
            structural = candidate.structural_eligibility
            action_certified = False
            action_name: str | None = None
            minimax: float | None = None
            regrets: tuple[float, ...] = ()
            if candidate.transports_mean:
                assert candidate.query_effect is not None
                assert candidate.query_error_radius is not None
                query_center = baseline + candidate.query_effect
                regret_values = self._robust_regrets(
                    query_center,
                    candidate.query_error_radius,
                    weights,
                    offsets,
                )
                regrets = tuple(float(value) for value in regret_values)
                minimax = float(np.min(regret_values))
                minimizers = np.flatnonzero(regret_values <= minimax + tie_tolerance)
                if minimizers.size == 1:
                    action_index = int(minimizers[0])
                    action_name = names[action_index]
                    action_certified = minimax <= regret_tolerance
                selectable = structural and action_certified
                if not candidate.target_outcome_blind:
                    reason = "target-outcome-contaminated"
                elif failed:
                    reason = "registered-check-failed"
                elif minimizers.size != 1:
                    reason = "action-not-unique"
                elif not action_certified:
                    reason = "regret-budget-exceeded"
                else:
                    reason = "mean-transport-and-action-certified"
            elif candidate.tier is TransportTier.UNCERTAINTY_ONLY:
                selectable = structural
                reason = (
                    "belief-only-transport-eligible"
                    if selectable
                    else (
                        "target-outcome-contaminated"
                        if not candidate.target_outcome_blind
                        else "registered-check-failed"
                    )
                )
            else:
                selectable = structural
                reason = (
                    "procedure-only-refit-required"
                    if selectable
                    else (
                        "target-outcome-contaminated"
                        if not candidate.target_outcome_blind
                        else "registered-check-failed"
                    )
                )

            if selected_candidate is None and selectable:
                selected_candidate = candidate
                if candidate.transports_mean:
                    assert action_name is not None
                    selected_action_index = names.index(action_name)
            provisional.append(
                _ProvisionalTierEvaluation(
                    candidate=candidate,
                    structurally_eligible=structural,
                    failed_checks=failed,
                    action_certified=action_certified,
                    action_name=action_name,
                    minimax_regret_upper=minimax,
                    robust_regret_upper_by_action=regrets,
                    selectable=selectable,
                    reason_code=reason,
                )
            )

        if selected_candidate is None:
            selected_tier = None
            selected_candidate_id = None
            selected_action = fallback
            belief_only = False
        elif selected_candidate.transports_mean:
            assert selected_action_index is not None
            selected_tier = selected_candidate.tier
            selected_candidate_id = selected_candidate.resolved_candidate_id
            selected_action = names[selected_action_index]
            belief_only = False
        else:
            selected_tier = selected_candidate.tier
            selected_candidate_id = selected_candidate.resolved_candidate_id
            selected_action = fallback
            belief_only = selected_candidate.tier is TransportTier.UNCERTAINTY_ONLY

        evaluations = tuple(
            TransportTierEvaluationV1(
                tier=item.candidate.tier,
                candidate_id=item.candidate.resolved_candidate_id,
                structurally_eligible=item.structurally_eligible,
                failed_checks=item.failed_checks,
                target_outcome_blind=item.candidate.target_outcome_blind,
                action_certified=item.action_certified,
                action_name=item.action_name,
                minimax_regret_upper=item.minimax_regret_upper,
                robust_regret_upper_by_action=(item.robust_regret_upper_by_action),
                selectable=item.selectable,
                selected=item.candidate is selected_candidate,
                reason_code=item.reason_code,
            )
            for item in provisional
        )

        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "query_contract_id", query_contract_id)
        object.__setattr__(self, "baseline_belief_id", baseline_belief_id)
        object.__setattr__(self, "action_portfolio_id", action_portfolio_id)
        object.__setattr__(self, "baseline_query", baseline)
        object.__setattr__(self, "action_names", names)
        object.__setattr__(self, "action_weights", weights)
        object.__setattr__(self, "action_offsets", offsets)
        object.__setattr__(self, "fallback_action_name", fallback)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "candidate_order", candidates)
        object.__setattr__(self, "regret_tolerance", regret_tolerance)
        object.__setattr__(self, "action_tie_tolerance", tie_tolerance)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "tier_evaluations", evaluations)
        object.__setattr__(self, "selected_tier", selected_tier)
        object.__setattr__(self, "selected_candidate_id", selected_candidate_id)
        object.__setattr__(self, "selected_action_name", selected_action)
        object.__setattr__(
            self,
            "used_exact_fallback",
            selected_action == fallback,
        )
        object.__setattr__(self, "belief_transport_only", belief_only)

        expected = _canonical_id(self.descriptor())
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match certificate content")
        object.__setattr__(self, "artifact_id", expected)

    @staticmethod
    def _robust_regrets(
        query_center: np.ndarray,
        radius: float,
        weights: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        """Exact worst-case affine-loss regret over one Euclidean query ball."""

        action_count = weights.shape[0]
        result = np.empty(action_count, dtype=np.float64)
        for action in range(action_count):
            weight_differences = weights[action] - weights
            center_gaps = weight_differences @ query_center + offsets[action] - offsets
            ball_support = np.linalg.norm(weight_differences, axis=1) * radius
            result[action] = float(np.max(center_gaps + ball_support))
        return result

    def evaluation_for(self, tier: TransportTier) -> TransportTierEvaluationV1:
        requested = TransportTier(tier)
        for evaluation in self.tier_evaluations:
            if evaluation.tier is requested:
                return evaluation
        raise KeyError(requested.value)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": TRANSPORT4D_SCHEMA,
            "schema_version": TRANSPORT4D_VERSION,
            "semantics": TRANSPORT4D_SEMANTICS,
            "query_id": self.query_id,
            "query_contract_id": self.query_contract_id,
            "baseline_belief_id": self.baseline_belief_id,
            "action_portfolio_id": self.action_portfolio_id,
            "baseline_query": _array_record(self.baseline_query),
            "action_names": list(self.action_names),
            "action_weights": _array_record(self.action_weights),
            "action_offsets": _array_record(self.action_offsets),
            "fallback_action_name": self.fallback_action_name,
            "candidates": [
                candidate.descriptor() for candidate in self.candidate_order
            ],
            "regret_tolerance": self.regret_tolerance,
            "action_tie_tolerance": self.action_tie_tolerance,
            "metadata": self.metadata,
            "selected_tier": self.selected_tier.value if self.selected_tier else None,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_action_name": self.selected_action_name,
            "used_exact_fallback": self.used_exact_fallback,
            "belief_transport_only": self.belief_transport_only,
            "tier_evaluations": [
                evaluation.to_record() for evaluation in self.tier_evaluations
            ],
            "claim_boundary": TRANSPORT4D_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    def arrays(self) -> Mapping[str, np.ndarray]:
        result: dict[str, np.ndarray] = {
            "baseline_query": self.baseline_query,
            "action_weights": self.action_weights,
            "action_offsets": self.action_offsets,
        }
        for candidate in self.candidate_order:
            if candidate.query_effect is not None:
                result[f"query_effect::{candidate.tier.value}"] = candidate.query_effect
        return result
