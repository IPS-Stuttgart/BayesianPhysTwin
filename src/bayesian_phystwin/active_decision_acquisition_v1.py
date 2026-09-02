"""Active acquisition for finite decision-identifiable physical beliefs.

The current physical evidence may determine only quotient-class masses over a
finite set of prior-supported hypotheses.  A deterministic probe partitions
those hypotheses by its possible outcomes.  This module computes the exact
post-outcome worst-case decision regret without inventing within-class odds,
and synthesizes a minimum worst-case-cost adaptive probe policy for small probe
sets.

For a current quotient class ``c`` with fixed mass ``lambda_c`` and an observed
probe-history event ``E``, the amount of class mass surviving ``E`` is:

* exactly ``lambda_c`` if every supported member of class ``c`` lies in ``E``;
* any value in ``[0, lambda_c]`` if only some supported members lie in ``E``;
* zero if no supported member lies in ``E``.

Conditioning normalizes those surviving masses.  Maximizing a pairwise expected
loss gap is therefore a box-constrained linear-fractional problem.  Its optimum
is a weighted average obtained by retaining every mandatory class and adding
optional classes in descending loss-gap order.  This gives an exact certificate
for every complete belief compatible with the original quotient masses and the
observed deterministic probe outcomes.

The implementation is deliberately finite and fail-closed.  It does not validate
probe models, hypothesis support, losses, quotient masses, costs, or transport to
a new physical system.  Exact dynamic programming is exponential in the number
of probes and intended for registered, modest candidate sets.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import cache
from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

ACTIVE_DECISION_ACQUISITION_VERSION: Final = 1
ACTIVE_DECISION_ACQUISITION_SEMANTICS: Final = (
    "exact-active-decision-certificate-over-conditioned-query-quotient-v1"
)
ACTIVE_DECISION_ACQUISITION_CLAIM_BOUNDARY: Final = (
    "Exact only for the supplied finite hypotheses, prior support, quotient "
    "masses, deterministic probe partitions, costs, loss matrix, and regret "
    "tolerance. It does not validate a provider or probe model, establish "
    "physical-state identification, calibrate probabilities, guarantee "
    "held-out transport, certify continuous actions, authorize deployment, or "
    "certify safety."
)

_PROBABILITY_ATOL: Final = 1e-12
_NUMERICAL_ATOL: Final = 1e-12


def _readonly_float(value: object, *, ndim: int, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = array.copy()
    result.setflags(write=False)
    return result


def _readonly_int(value: object, *, name: str, size: int | None = None) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer values")
    array = np.ascontiguousarray(raw, dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain exactly {size} entries")
    result = array.copy()
    result.setflags(write=False)
    return result


def _readonly_bool(value: object, *, name: str, size: int) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{name} must contain exactly {size} Boolean entries")
    result = array.copy()
    result.setflags(write=False)
    return result


def _probabilities(value: object, *, name: str, size: int | None = None) -> FloatArray:
    array = _readonly_float(value, ndim=1, name=name)
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain exactly {size} entries")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(array, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL):
        raise ValueError(f"{name} must sum to one")
    result = np.ascontiguousarray(array / total)
    result.setflags(write=False)
    return result


def _class_index(value: object, *, size: int) -> IntArray:
    array = _readonly_int(value, name="class_index", size=size)
    if np.any(array < 0):
        raise ValueError("class_index labels must be nonnegative")
    unique = np.unique(array)
    if not np.array_equal(unique, np.arange(int(unique[-1]) + 1, dtype=np.int64)):
        raise ValueError("class_index labels must be contiguous from zero")
    return array


def _losses(value: object, *, hypotheses: int) -> FloatArray:
    array = _readonly_float(value, ndim=2, name="loss_by_hypothesis_action")
    if array.shape[0] != hypotheses or array.shape[1] < 2:
        raise ValueError(
            "loss_by_hypothesis_action must have shape "
            "(hypothesis_count, action_count>=2)"
        )
    return array


def _loss_radii(
    value: object | None,
    *,
    hypotheses: int,
    actions: int,
) -> FloatArray:
    if value is None:
        radii = np.zeros((hypotheses, actions), dtype=np.float64)
        radii.setflags(write=False)
        return radii
    radii = _readonly_float(
        value,
        ndim=2,
        name="loss_radius_by_hypothesis_action",
    )
    if radii.shape != (hypotheses, actions):
        raise ValueError(
            "loss_radius_by_hypothesis_action must match "
            "loss_by_hypothesis_action"
        )
    if np.any(radii < 0.0):
        raise ValueError("loss_radius_by_hypothesis_action must be nonnegative")
    return radii


def _finite_nonnegative(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _array_digest(hasher: object, value: np.ndarray) -> None:
    if not hasattr(hasher, "update"):
        raise TypeError("hasher does not provide update")
    array = np.ascontiguousarray(value)
    shape = ",".join(str(item) for item in array.shape).encode("ascii")
    hasher.update(len(shape).to_bytes(4, "big"))  # type: ignore[attr-defined]
    hasher.update(shape)  # type: ignore[attr-defined]
    payload = array.tobytes(order="C")
    hasher.update(len(payload).to_bytes(8, "big"))  # type: ignore[attr-defined]
    hasher.update(payload)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class DeterministicDecisionProbeV1:
    """One finite-cost deterministic experiment over registered hypotheses."""

    probe_id: str
    outcome_index: IntArray
    cost: float
    probe_model_id: str = "registered-deterministic-partition-v1"
    probe_content_id: str = field(init=False)

    def __post_init__(self) -> None:
        probe_id = _identifier(self.probe_id, name="probe_id")
        model_id = _identifier(self.probe_model_id, name="probe_model_id")
        outcomes = _readonly_int(self.outcome_index, name="outcome_index")
        if np.any(outcomes < 0):
            raise ValueError("outcome_index labels must be nonnegative")
        cost = _finite_nonnegative(self.cost, name="cost", positive=True)
        hasher = hashlib.sha256()
        hasher.update(b"prob4d.active-decision-probe.v1\0")
        for text in (probe_id, model_id):
            payload = text.encode("utf-8")
            hasher.update(len(payload).to_bytes(4, "big"))
            hasher.update(payload)
        _array_digest(hasher, np.asarray(outcomes, dtype="<i8"))
        _array_digest(hasher, np.asarray([cost], dtype="<f8"))
        object.__setattr__(self, "probe_id", probe_id)
        object.__setattr__(self, "probe_model_id", model_id)
        object.__setattr__(self, "outcome_index", outcomes)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "probe_content_id", hasher.hexdigest())

    @property
    def hypothesis_count(self) -> int:
        return int(self.outcome_index.size)


class ConditionedDecisionCertificateV1(NamedTuple):
    """Exact decision certificate after a deterministic observation history."""

    consistent_hypothesis_mask: BoolArray
    class_mass_lower_before_normalization: FloatArray
    class_mass_upper_before_normalization: FloatArray
    pairwise_worst_case_loss_gap: FloatArray
    worst_case_regret: FloatArray
    minimax_action_index: int
    minimax_worst_case_regret: float
    regret_tolerance: float
    tolerance_admissible_action_mask: BoolArray
    robustly_optimal_action_mask: BoolArray
    loss_uncertainty_used: bool
    maximum_loss_radius: float

    @property
    def has_tolerance_admissible_action(self) -> bool:
        return bool(np.any(self.tolerance_admissible_action_mask))

    @property
    def action_count(self) -> int:
        return int(self.worst_case_regret.size)

    def summary(self) -> dict[str, object]:
        return {
            "version": ACTIVE_DECISION_ACQUISITION_VERSION,
            "semantics": ACTIVE_DECISION_ACQUISITION_SEMANTICS,
            "consistent_hypothesis_count": int(
                np.count_nonzero(self.consistent_hypothesis_mask)
            ),
            "minimax_action_index": self.minimax_action_index,
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "regret_tolerance": self.regret_tolerance,
            "has_tolerance_admissible_action": self.has_tolerance_admissible_action,
            "loss_uncertainty_used": self.loss_uncertainty_used,
            "maximum_loss_radius": self.maximum_loss_radius,
            "claim_boundary": ACTIVE_DECISION_ACQUISITION_CLAIM_BOUNDARY,
        }


def _maximum_box_weighted_average(
    values: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> float:
    """Maximize ``values @ x / sum(x)`` over ``lower <= x <= upper``.

    ``lower`` and ``upper`` are nonnegative class-mass bounds before event
    normalization.  The optimum includes all mandatory lower mass and then
    optional capacities in descending value order.  If there is no mandatory
    mass, any positive mass can be placed on a maximizing optional class.
    """

    active = upper > _NUMERICAL_ATOL
    if not np.any(active):
        raise ValueError("conditioning event has zero mass under every feasible belief")
    mandatory_mass = float(np.sum(lower, dtype=np.float64))
    if mandatory_mass <= _NUMERICAL_ATOL:
        return float(np.max(values[active]))

    numerator = float(np.dot(values, lower))
    denominator = mandatory_mass
    best = numerator / denominator
    capacity = upper - lower
    order = np.argsort(-values, kind="stable")
    for index in order:
        increment = float(capacity[index])
        if increment <= _NUMERICAL_ATOL:
            continue
        numerator += increment * float(values[index])
        denominator += increment
        best = max(best, numerator / denominator)
    return best


def conditioned_query_decision_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    *,
    consistent_hypothesis_mask: object | None = None,
    loss_radius_by_hypothesis_action: object | None = None,
    regret_tolerance: float = 0.0,
) -> ConditionedDecisionCertificateV1:
    """Certify a finite decision after deterministic probe outcomes.

    The original quotient masses remain fixed before conditioning.  Within each
    class, all probability allocations supported by the prior remain possible.
    The supplied mask is the intersection of all observed deterministic probe
    outcomes.  The returned loss gaps are exact suprema over every resulting
    conditional complete belief.  Optional elementwise loss radii provide a
    simultaneous upper certificate for unknown true losses inside the supplied
    intervals; validity of those radii remains caller-owned.
    """

    prior = _probabilities(prior_weights, name="prior_weights")
    classes = _class_index(class_index, size=prior.size)
    class_count = int(np.max(classes)) + 1
    quotient = _probabilities(
        quotient_weights,
        name="quotient_weights",
        size=class_count,
    )
    losses = _losses(loss_by_hypothesis_action, hypotheses=prior.size)
    radii = _loss_radii(
        loss_radius_by_hypothesis_action,
        hypotheses=prior.size,
        actions=losses.shape[1],
    )
    tolerance = _finite_nonnegative(regret_tolerance, name="regret_tolerance")
    if consistent_hypothesis_mask is None:
        consistent = _readonly_bool(
            np.ones(prior.size, dtype=np.bool_),
            name="consistent_hypothesis_mask",
            size=prior.size,
        )
    else:
        consistent = _readonly_bool(
            consistent_hypothesis_mask,
            name="consistent_hypothesis_mask",
            size=prior.size,
        )

    support = prior > 0.0
    prior_class_mass = np.bincount(
        classes,
        weights=prior,
        minlength=class_count,
    ).astype(np.float64, copy=False)
    unsupported = (quotient > _PROBABILITY_ATOL) & (
        prior_class_mass <= _PROBABILITY_ATOL
    )
    if np.any(unsupported):
        raise ValueError(
            "positive quotient mass has zero prior support for classes "
            f"{np.flatnonzero(unsupported).tolist()}"
        )

    lower = np.zeros(class_count, dtype=np.float64)
    upper = np.zeros(class_count, dtype=np.float64)
    class_consistent_members: list[np.ndarray] = []
    for class_id in range(class_count):
        supported = (classes == class_id) & support
        retained = supported & consistent
        class_consistent_members.append(retained)
        if not np.any(retained) or quotient[class_id] <= _PROBABILITY_ATOL:
            continue
        upper[class_id] = quotient[class_id]
        if np.array_equal(retained, supported):
            lower[class_id] = quotient[class_id]

    if float(np.sum(upper, dtype=np.float64)) <= _PROBABILITY_ATOL:
        raise ValueError("conditioning history is impossible under prior support")

    action_count = losses.shape[1]
    pairwise = np.zeros((action_count, action_count), dtype=np.float64)
    differences = (
        losses[:, :, None]
        - losses[:, None, :]
        + radii[:, :, None]
        + radii[:, None, :]
    )
    for action in range(action_count):
        for benchmark in range(action_count):
            if action == benchmark:
                continue
            class_max = np.zeros(class_count, dtype=np.float64)
            for class_id, retained in enumerate(class_consistent_members):
                if upper[class_id] <= _PROBABILITY_ATOL:
                    continue
                class_max[class_id] = float(
                    np.max(differences[retained, action, benchmark])
                )
            pairwise[action, benchmark] = _maximum_box_weighted_average(
                class_max,
                lower,
                upper,
            )

    worst_case = np.maximum(np.max(pairwise, axis=1), 0.0)
    minimum = float(np.min(worst_case))
    minimizers = np.flatnonzero(
        np.isclose(worst_case, minimum, rtol=0.0, atol=_NUMERICAL_ATOL)
    )
    minimax_action = int(minimizers[0])
    tolerance_mask = worst_case <= tolerance + _NUMERICAL_ATOL
    robust_mask = np.all(pairwise <= _NUMERICAL_ATOL, axis=1)

    return ConditionedDecisionCertificateV1(
        consistent_hypothesis_mask=consistent,
        class_mass_lower_before_normalization=_immutable_float64(lower),
        class_mass_upper_before_normalization=_immutable_float64(upper),
        pairwise_worst_case_loss_gap=_immutable_float64(pairwise),
        worst_case_regret=_immutable_float64(worst_case),
        minimax_action_index=minimax_action,
        minimax_worst_case_regret=minimum,
        regret_tolerance=tolerance,
        tolerance_admissible_action_mask=_immutable_bool(tolerance_mask),
        robustly_optimal_action_mask=_immutable_bool(robust_mask),
        loss_uncertainty_used=bool(np.any(radii > 0.0)),
        maximum_loss_radius=float(np.max(radii, initial=0.0)),
    )


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


class ProbeOutcomeCertificateV1(NamedTuple):
    outcome_index: int
    certificate: ConditionedDecisionCertificateV1


class ProbeEvaluationV1(NamedTuple):
    probe_id: str
    probe_content_id: str
    cost: float
    outcome_certificates: tuple[ProbeOutcomeCertificateV1, ...]
    worst_outcome_minimax_regret: float
    all_outcomes_tolerance_certified: bool


def evaluate_deterministic_probe(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    probe: DeterministicDecisionProbeV1,
    *,
    consistent_hypothesis_mask: object | None = None,
    loss_radius_by_hypothesis_action: object | None = None,
    regret_tolerance: float = 0.0,
) -> ProbeEvaluationV1:
    """Evaluate every feasible outcome of one deterministic probe."""

    prior = _probabilities(prior_weights, name="prior_weights")
    if probe.hypothesis_count != prior.size:
        raise ValueError("probe outcome roster does not match hypothesis count")
    if consistent_hypothesis_mask is None:
        current = np.ones(prior.size, dtype=np.bool_)
    else:
        current = np.ascontiguousarray(consistent_hypothesis_mask, dtype=np.bool_)
        if current.ndim != 1 or current.size != prior.size:
            raise ValueError("consistent_hypothesis_mask shape mismatch")
    classes = _class_index(class_index, size=prior.size)
    quotient = _probabilities(
        quotient_weights,
        name="quotient_weights",
        size=int(np.max(classes)) + 1,
    )
    feasible = current & (prior > 0.0) & (quotient[classes] > _PROBABILITY_ATOL)
    outcomes = np.unique(probe.outcome_index[feasible])
    if outcomes.size == 0:
        raise ValueError("current history has no feasible probe outcome")

    rows: list[ProbeOutcomeCertificateV1] = []
    for outcome in outcomes:
        child = current & (probe.outcome_index == outcome)
        certificate = conditioned_query_decision_certificate(
            prior,
            quotient,
            classes,
            loss_by_hypothesis_action,
            consistent_hypothesis_mask=child,
            loss_radius_by_hypothesis_action=loss_radius_by_hypothesis_action,
            regret_tolerance=regret_tolerance,
        )
        rows.append(
            ProbeOutcomeCertificateV1(
                outcome_index=int(outcome),
                certificate=certificate,
            )
        )
    worst = max(row.certificate.minimax_worst_case_regret for row in rows)
    return ProbeEvaluationV1(
        probe_id=probe.probe_id,
        probe_content_id=probe.probe_content_id,
        cost=probe.cost,
        outcome_certificates=tuple(rows),
        worst_outcome_minimax_regret=float(worst),
        all_outcomes_tolerance_certified=bool(
            all(row.certificate.has_tolerance_admissible_action for row in rows)
        ),
    )


class ActiveDecisionPolicyNodeV1(NamedTuple):
    state_id: str
    consistent_hypothesis_indices: tuple[int, ...]
    remaining_probe_ids: tuple[str, ...]
    minimax_action_index: int
    minimax_worst_case_regret: float
    certified: bool
    selected_probe_id: str | None
    outcome_children: tuple[tuple[int, str], ...]
    worst_case_remaining_cost: float


class ActiveDecisionPolicyV1(NamedTuple):
    root_state_id: str
    root_worst_case_cost: float
    feasible: bool
    nodes: tuple[ActiveDecisionPolicyNodeV1, ...]
    probe_content_ids: tuple[str, ...]
    regret_tolerance: float
    policy_id: str

    def summary(self) -> dict[str, object]:
        return {
            "version": ACTIVE_DECISION_ACQUISITION_VERSION,
            "semantics": ACTIVE_DECISION_ACQUISITION_SEMANTICS,
            "root_state_id": self.root_state_id,
            "root_worst_case_cost": self.root_worst_case_cost,
            "feasible": self.feasible,
            "node_count": len(self.nodes),
            "probe_count": len(self.probe_content_ids),
            "regret_tolerance": self.regret_tolerance,
            "policy_id": self.policy_id,
            "claim_boundary": ACTIVE_DECISION_ACQUISITION_CLAIM_BOUNDARY,
        }


def _mask_key(mask: np.ndarray) -> bytes:
    return np.packbits(mask, bitorder="little").tobytes()


def _state_id(mask: np.ndarray, remaining: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"prob4d.active-decision-state.v1\0")
    payload = _mask_key(mask)
    hasher.update(len(payload).to_bytes(4, "big"))
    hasher.update(payload)
    hasher.update(np.asarray(remaining, dtype="<i8").tobytes())
    return hasher.hexdigest()


def synthesize_minimax_active_decision_policy(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    probes: tuple[DeterministicDecisionProbeV1, ...],
    *,
    regret_tolerance: float = 0.0,
    loss_radius_by_hypothesis_action: object | None = None,
    initial_consistent_hypothesis_mask: object | None = None,
    maximum_probe_count: int = 18,
) -> ActiveDecisionPolicyV1:
    """Return the exact minimum worst-case-cost adaptive probe policy.

    Dynamic programming enumerates probe subsets and deterministic outcome
    histories.  The method is exponential and refuses candidate sets larger
    than ``maximum_probe_count``.
    """

    prior = _probabilities(prior_weights, name="prior_weights")
    classes = _class_index(class_index, size=prior.size)
    quotient = _probabilities(
        quotient_weights,
        name="quotient_weights",
        size=int(np.max(classes)) + 1,
    )
    losses = _losses(loss_by_hypothesis_action, hypotheses=prior.size)
    radii = _loss_radii(
        loss_radius_by_hypothesis_action,
        hypotheses=prior.size,
        actions=losses.shape[1],
    )
    tolerance = _finite_nonnegative(regret_tolerance, name="regret_tolerance")
    if not isinstance(maximum_probe_count, int) or maximum_probe_count < 0:
        raise ValueError("maximum_probe_count must be a nonnegative integer")
    if len(probes) > maximum_probe_count:
        raise ValueError(
            f"exact active policy supports at most {maximum_probe_count} probes"
        )
    identifiers = [probe.probe_id for probe in probes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("probe_id values must be unique")
    for probe in probes:
        if probe.hypothesis_count != prior.size:
            raise ValueError(f"probe {probe.probe_id!r} has the wrong hypothesis count")
    ordered = tuple(sorted(probes, key=lambda probe: probe.probe_id))
    if initial_consistent_hypothesis_mask is None:
        initial = np.ones(prior.size, dtype=np.bool_)
    else:
        initial = np.ascontiguousarray(
            initial_consistent_hypothesis_mask,
            dtype=np.bool_,
        )
        if initial.ndim != 1 or initial.size != prior.size:
            raise ValueError("initial_consistent_hypothesis_mask shape mismatch")
    initial.setflags(write=False)

    node_specs: dict[
        tuple[bytes, tuple[int, ...]],
        tuple[
            ConditionedDecisionCertificateV1,
            int | None,
            tuple[tuple[int, bytes, tuple[int, ...]], ...],
            float,
        ],
    ] = {}

    @cache
    def solve(mask_bytes: bytes, remaining: tuple[int, ...]) -> float:
        mask = np.unpackbits(
            np.frombuffer(mask_bytes, dtype=np.uint8),
            bitorder="little",
            count=prior.size,
        ).astype(np.bool_)
        certificate = conditioned_query_decision_certificate(
            prior,
            quotient,
            classes,
            losses,
            consistent_hypothesis_mask=mask,
            loss_radius_by_hypothesis_action=radii,
            regret_tolerance=tolerance,
        )
        state_key = (mask_bytes, remaining)
        if certificate.has_tolerance_admissible_action:
            node_specs[state_key] = (certificate, None, (), 0.0)
            return 0.0

        best_cost = math.inf
        best_probe: int | None = None
        best_children: tuple[tuple[int, bytes, tuple[int, ...]], ...] = ()
        feasible_mask = mask & (prior > 0.0) & (
            quotient[classes] > _PROBABILITY_ATOL
        )
        for probe_index in remaining:
            probe = ordered[probe_index]
            outcomes = np.unique(probe.outcome_index[feasible_mask])
            if outcomes.size <= 1:
                continue
            child_remaining = tuple(
                index for index in remaining if index != probe_index
            )
            child_specs: list[tuple[int, bytes, tuple[int, ...]]] = []
            child_costs: list[float] = []
            for outcome in outcomes:
                child_mask = mask & (probe.outcome_index == outcome)
                child_bytes = _mask_key(child_mask)
                child_specs.append((int(outcome), child_bytes, child_remaining))
                child_costs.append(solve(child_bytes, child_remaining))
            candidate = probe.cost + max(child_costs)
            if candidate < best_cost - _NUMERICAL_ATOL or (
                math.isclose(candidate, best_cost, abs_tol=_NUMERICAL_ATOL)
                and best_probe is not None
                and probe.probe_id < ordered[best_probe].probe_id
            ):
                best_cost = candidate
                best_probe = probe_index
                best_children = tuple(child_specs)
            elif best_probe is None and math.isfinite(candidate):
                best_cost = candidate
                best_probe = probe_index
                best_children = tuple(child_specs)
        node_specs[state_key] = (
            certificate,
            best_probe,
            best_children,
            float(best_cost),
        )
        return float(best_cost)

    root_remaining = tuple(range(len(ordered)))
    root_bytes = _mask_key(initial)
    root_cost = solve(root_bytes, root_remaining)

    nodes: list[ActiveDecisionPolicyNodeV1] = []
    for (mask_bytes, remaining), (
        certificate,
        selected_probe,
        children,
        cost,
    ) in sorted(node_specs.items(), key=lambda item: (item[0][0], item[0][1])):
        mask = np.unpackbits(
            np.frombuffer(mask_bytes, dtype=np.uint8),
            bitorder="little",
            count=prior.size,
        ).astype(np.bool_)
        nodes.append(
            ActiveDecisionPolicyNodeV1(
                state_id=_state_id(mask, remaining),
                consistent_hypothesis_indices=tuple(
                    int(index) for index in np.flatnonzero(mask)
                ),
                remaining_probe_ids=tuple(
                    ordered[index].probe_id for index in remaining
                ),
                minimax_action_index=certificate.minimax_action_index,
                minimax_worst_case_regret=certificate.minimax_worst_case_regret,
                certified=certificate.has_tolerance_admissible_action,
                selected_probe_id=(
                    None if selected_probe is None else ordered[selected_probe].probe_id
                ),
                outcome_children=tuple(
                    (
                        outcome,
                        _state_id(
                            np.unpackbits(
                                np.frombuffer(child_bytes, dtype=np.uint8),
                                bitorder="little",
                                count=prior.size,
                            ).astype(np.bool_),
                            child_remaining,
                        ),
                    )
                    for outcome, child_bytes, child_remaining in children
                ),
                worst_case_remaining_cost=cost,
            )
        )

    policy_payload = {
        "root_state_id": _state_id(initial, root_remaining),
        "root_worst_case_cost": root_cost if math.isfinite(root_cost) else "infinity",
        "regret_tolerance": tolerance,
        "probe_content_ids": [probe.probe_content_id for probe in ordered],
        "nodes": [
            {
                "state_id": node.state_id,
                "consistent_hypothesis_indices": node.consistent_hypothesis_indices,
                "remaining_probe_ids": node.remaining_probe_ids,
                "minimax_action_index": node.minimax_action_index,
                "minimax_worst_case_regret": node.minimax_worst_case_regret,
                "certified": node.certified,
                "selected_probe_id": node.selected_probe_id,
                "outcome_children": node.outcome_children,
                "worst_case_remaining_cost": (
                    node.worst_case_remaining_cost
                    if math.isfinite(node.worst_case_remaining_cost)
                    else "infinity"
                ),
            }
            for node in nodes
        ],
    }
    policy_id = hashlib.sha256(
        json.dumps(
            policy_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return ActiveDecisionPolicyV1(
        root_state_id=policy_payload["root_state_id"],
        root_worst_case_cost=root_cost,
        feasible=math.isfinite(root_cost),
        nodes=tuple(nodes),
        probe_content_ids=tuple(probe.probe_content_id for probe in ordered),
        regret_tolerance=tolerance,
        policy_id=policy_id,
    )


class GlobalProbeSetV1(NamedTuple):
    selected_probe_ids: tuple[str, ...]
    total_cost: float
    conflict_pair_count: int
    feasible: bool
    exact: bool


def minimum_cost_global_decision_identifying_probe_set(
    prior_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    probes: tuple[DeterministicDecisionProbeV1, ...],
    *,
    signature_tolerance: float = 0.0,
    maximum_probe_count: int = 24,
) -> GlobalProbeSetV1:
    """Solve the weighted set-cover form of global decision identification.

    A conflict pair consists of two prior-supported hypotheses currently in the
    same quotient class but with different action-loss-difference signatures.
    A probe covers the pair when it predicts different outcomes.  Covering all
    pairs is necessary and sufficient for the refined partition to preserve all
    expected action differences under arbitrary within-cell redistribution.
    """

    prior = _probabilities(prior_weights, name="prior_weights")
    classes = _class_index(class_index, size=prior.size)
    losses = _losses(loss_by_hypothesis_action, hypotheses=prior.size)
    tolerance = _finite_nonnegative(
        signature_tolerance,
        name="signature_tolerance",
    )
    if len(probes) > maximum_probe_count:
        raise ValueError(
            f"exact global set cover supports at most {maximum_probe_count} probes"
        )
    identifiers = [probe.probe_id for probe in probes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("probe_id values must be unique")
    ordered = tuple(sorted(probes, key=lambda probe: probe.probe_id))
    for probe in ordered:
        if probe.hypothesis_count != prior.size:
            raise ValueError(f"probe {probe.probe_id!r} has the wrong hypothesis count")

    signatures = losses[:, 1:] - losses[:, :1]
    support = prior > 0.0
    conflicts: list[tuple[int, int]] = []
    for left in range(prior.size):
        if not support[left]:
            continue
        for right in range(left + 1, prior.size):
            if not support[right] or classes[left] != classes[right]:
                continue
            if not np.allclose(
                signatures[left],
                signatures[right],
                rtol=0.0,
                atol=tolerance,
            ):
                conflicts.append((left, right))
    if not conflicts:
        return GlobalProbeSetV1((), 0.0, 0, True, True)

    full = (1 << len(conflicts)) - 1
    covers: list[int] = []
    for probe in ordered:
        mask = 0
        for pair_index, (left, right) in enumerate(conflicts):
            if probe.outcome_index[left] != probe.outcome_index[right]:
                mask |= 1 << pair_index
        covers.append(mask)
    uncovered = full
    for cover in covers:
        uncovered &= ~cover
    if uncovered:
        return GlobalProbeSetV1((), math.inf, len(conflicts), False, True)

    best_cost = math.inf
    best_selection: tuple[int, ...] = ()

    def search(
        index: int,
        covered: int,
        cost: float,
        selected: tuple[int, ...],
    ) -> None:
        nonlocal best_cost, best_selection
        if cost >= best_cost - _NUMERICAL_ATOL:
            return
        if covered == full:
            best_cost = cost
            best_selection = selected
            return
        if index >= len(ordered):
            return
        remaining_union = covered
        for candidate in range(index, len(ordered)):
            remaining_union |= covers[candidate]
        if remaining_union != full:
            return
        search(
            index + 1,
            covered | covers[index],
            cost + ordered[index].cost,
            (*selected, index),
        )
        search(index + 1, covered, cost, selected)

    search(0, 0, 0.0, ())
    return GlobalProbeSetV1(
        selected_probe_ids=tuple(ordered[index].probe_id for index in best_selection),
        total_cost=float(best_cost),
        conflict_pair_count=len(conflicts),
        feasible=math.isfinite(best_cost),
        exact=True,
    )


__all__ = [
    "ACTIVE_DECISION_ACQUISITION_CLAIM_BOUNDARY",
    "ACTIVE_DECISION_ACQUISITION_SEMANTICS",
    "ACTIVE_DECISION_ACQUISITION_VERSION",
    "ActiveDecisionPolicyNodeV1",
    "ActiveDecisionPolicyV1",
    "ConditionedDecisionCertificateV1",
    "DeterministicDecisionProbeV1",
    "GlobalProbeSetV1",
    "ProbeEvaluationV1",
    "ProbeOutcomeCertificateV1",
    "conditioned_query_decision_certificate",
    "evaluate_deterministic_probe",
    "minimum_cost_global_decision_identifying_probe_set",
    "synthesize_minimax_active_decision_policy",
]
