"""Exact finite-horizon certificates for stochastic adaptive sensing plans.

This standalone prototype extends a one-probe act--sense--fallback certificate
to finite-horizon adaptive policies.  A complete policy is chosen before any
probe outcome is observed.  It may then map each registered outcome to another
probe or to a terminal action.  Probe outcomes can be stochastic under each
finite physical hypothesis.

For each complete policy pi and physical hypothesis h, recursion computes the
expected total loss L(h, pi), including probe costs.  The policy is then treated
as one finite action by an exact quotient-regret certificate.  Consequently,
the returned plan is certified before sensing over every prior-supported
complete belief compatible with the supplied quotient masses.

The guarantee is conditional on the finite hypothesis support, quotient,
terminal losses, stochastic probe model, probe costs, finite horizon, and regret
tolerance.  It does not validate physical probe models, reset semantics,
exchangeability, conditional-independence validity, support completeness,
deployment, or safety.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final, Literal, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_VERSION: Final = 1
ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_SEMANTICS: Final = (
    "exact-preprobe-quotient-regret-over-finite-stochastic-policy-trees-v1"
)
ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied finite hypotheses, prior "
    "support, quotient masses, terminal losses, conditionally independent "
    "registered probe-outcome models, nonrepeatable probes, probe costs, finite "
    "horizon, retained policy class, and regret tolerance. It does not validate "
    "the physical support, quotient, sensor model, conditional independence, "
    "reset semantics, exchangeability, target transport, deployment, or safety."
)

_PROBABILITY_ATOL: Final = 1.0e-12
ATOL: Final = 1.0e-12


def _immutable_array(value: object, *, dtype: np.dtype) -> np.ndarray:
    """Return a C-contiguous array backed by immutable ``bytes`` storage."""

    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("certificate arrays must not contain Python objects")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


def _float_array(value: object, *, ndim: int, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return _immutable_array(result, dtype=np.dtype(np.float64))


def _probability_vector(value: object, *, size: int, name: str) -> FloatArray:
    result = _float_array(value, ndim=1, name=name)
    if result.size != size or np.any(result < 0.0):
        raise ValueError(f"{name} must contain {size} nonnegative entries")
    total = float(np.sum(result, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL):
        raise ValueError(f"{name} must sum to one")
    return _immutable_array(result / total, dtype=np.dtype(np.float64))


def _index_vector(value: object, *, size: int, upper: int, name: str) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    result = np.asarray(raw, dtype=np.int64)
    if result.ndim != 1 or result.size != size:
        raise ValueError(f"{name} must contain exactly {size} entries")
    if np.any(result < 0) or np.any(result >= upper):
        raise ValueError(f"{name} entries must lie in [0, {upper})")
    return _immutable_array(result, dtype=np.dtype(np.int64))


def _nonnegative_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class AdaptivePlanV1:
    """One complete direct or finite-horizon adaptive sensing policy."""

    mode: Literal["act", "sense"]
    action_index: int | None = None
    probe_index: int | None = None
    children_by_outcome: tuple["AdaptivePlanV1", ...] = ()

    def __post_init__(self) -> None:
        if self.mode == "act":
            if (
                isinstance(self.action_index, (bool, np.bool_))
                or not isinstance(self.action_index, Integral)
                or int(self.action_index) < 0
                or self.probe_index is not None
                or self.children_by_outcome
            ):
                raise ValueError(
                    "act plans require one nonnegative action index and no probe/children"
                )
            return
        if self.mode != "sense":
            raise ValueError("plan mode must be 'act' or 'sense'")
        if (
            self.action_index is not None
            or isinstance(self.probe_index, (bool, np.bool_))
            or not isinstance(self.probe_index, Integral)
            or int(self.probe_index) < 0
            or not self.children_by_outcome
            or not all(isinstance(child, AdaptivePlanV1) for child in self.children_by_outcome)
        ):
            raise ValueError(
                "sense plans require one nonnegative probe index, no action, and children"
            )

    @property
    def probe_depth(self) -> int:
        if self.mode == "act":
            return 0
        return 1 + max(child.probe_depth for child in self.children_by_outcome)

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children_by_outcome)

    def canonical_key(self) -> tuple:
        if self.mode == "act":
            return (0, int(self.action_index))
        return (
            1,
            int(self.probe_index),
            tuple(child.canonical_key() for child in self.children_by_outcome),
        )

    @property
    def probe_indices(self) -> tuple[int, ...]:
        """Return every probe index appearing in the complete policy tree."""

        if self.mode == "act":
            return ()
        if self.probe_index is None:
            raise RuntimeError("sensing plan is missing its probe index")
        descendants = tuple(
            probe
            for child in self.children_by_outcome
            for probe in child.probe_indices
        )
        return (self.probe_index, *descendants)

    def terminal_action(self, outcomes: Sequence[int]) -> int:
        """Execute the frozen tree for an observed outcome sequence."""

        plan = self
        offset = 0
        while plan.mode == "sense":
            if offset >= len(outcomes):
                raise ValueError("outcome sequence ended before the plan terminated")
            label = outcomes[offset]
            if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral):
                raise ValueError("outcomes must contain integers")
            index = int(label)
            if index < 0 or index >= len(plan.children_by_outcome):
                raise ValueError("outcome is outside the registered plan branch set")
            plan = plan.children_by_outcome[index]
            offset += 1
        if offset != len(outcomes):
            raise ValueError("outcome sequence continues after the plan terminated")
        if plan.action_index is None:
            raise RuntimeError("terminal plan is missing an action")
        return plan.action_index


@dataclass(frozen=True)
class _Candidate:
    plan: AdaptivePlanV1
    loss: FloatArray


@dataclass(frozen=True)
class AdaptiveActSenseFallbackCertificateV1:
    """Exact pre-probe certificate over all retained adaptive plans."""

    plans: tuple[AdaptivePlanV1, ...]
    loss_by_hypothesis_plan: FloatArray
    pairwise_worst_case_gap: FloatArray
    worst_case_regret: FloatArray
    minimizer_mask: BoolArray
    fallback_plan_index: int
    candidate_plan_index: int | None
    output_plan_index: int
    output_mode: Literal["act", "sense", "fallback"]
    used_fallback: bool
    regret_tolerance: float
    raw_plan_count: int
    retained_plan_count: int

    @property
    def output_plan(self) -> AdaptivePlanV1:
        return self.plans[self.output_plan_index]

    @property
    def candidate_plan(self) -> AdaptivePlanV1 | None:
        return (
            None
            if self.candidate_plan_index is None
            else self.plans[self.candidate_plan_index]
        )

    def summary(self) -> dict[str, object]:
        candidate_regret = (
            None
            if self.candidate_plan_index is None
            else float(self.worst_case_regret[self.candidate_plan_index])
        )
        return {
            "version": ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_VERSION,
            "semantics": ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_SEMANTICS,
            "raw_plan_count": self.raw_plan_count,
            "retained_plan_count": self.retained_plan_count,
            "candidate_plan_index": self.candidate_plan_index,
            "candidate_worst_case_regret": candidate_regret,
            "output_plan_index": self.output_plan_index,
            "output_mode": self.output_mode,
            "used_fallback": self.used_fallback,
            "regret_tolerance": self.regret_tolerance,
            "output_probe_depth": self.output_plan.probe_depth,
            "output_node_count": self.output_plan.node_count,
            "claim_boundary": ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY,
        }


class _Enumerator:
    def __init__(
        self,
        terminal_losses: FloatArray,
        probe_probabilities: tuple[FloatArray, ...],
        probe_costs: FloatArray,
        *,
        max_depth: int,
        max_frontier_count: int,
        fallback_action_index: int,
    ) -> None:
        self.terminal_losses = terminal_losses
        self.probe_probabilities = probe_probabilities
        self.probe_costs = probe_costs
        self.max_depth = max_depth
        self.max_frontier_count = max_frontier_count
        self.fallback_action_index = fallback_action_index
        self.raw_plan_count = 0
        self._memo: dict[
            tuple[tuple[int, ...], tuple[int, ...], int], tuple[_Candidate, ...]
        ] = {}

    @staticmethod
    def _loss_key(loss: FloatArray) -> bytes:
        canonical = np.asarray(loss, dtype=np.dtype("<f8"), order="C")
        return canonical.tobytes(order="C")

    @staticmethod
    def _prefer(left: _Candidate, right: _Candidate) -> _Candidate:
        left_key = (
            left.plan.probe_depth, left.plan.node_count, left.plan.canonical_key()
        )
        right_key = (
            right.plan.probe_depth,
            right.plan.node_count,
            right.plan.canonical_key(),
        )
        return left if left_key <= right_key else right

    def _deduplicate_and_prune(
        self,
        candidates: list[_Candidate],
        *,
        preserve_fallback: bool,
    ) -> tuple[_Candidate, ...]:
        unique: dict[bytes, _Candidate] = {}
        fallback_candidate: _Candidate | None = None
        for candidate in candidates:
            is_fallback = (
                candidate.plan.mode == "act"
                and candidate.plan.action_index == self.fallback_action_index
            )
            if preserve_fallback and is_fallback:
                # The caller-owned fallback is part of the operational contract.
                # Retain it even when another plan has an identical loss vector;
                # otherwise loss-vector deduplication could erase the exact
                # fallback identity and incorrectly turn a nonunique minimizer
                # into a unique executable candidate.
                fallback_candidate = candidate
                continue
            key = self._loss_key(candidate.loss)
            existing = unique.get(key)
            unique[key] = (
                candidate if existing is None else self._prefer(existing, candidate)
            )
        ordered = list(unique.values())
        if fallback_candidate is not None:
            ordered.append(fallback_candidate)
        ordered.sort(key=lambda item: item.plan.canonical_key())

        keep = np.ones(len(ordered), dtype=bool)
        for index, candidate in enumerate(ordered):
            if not keep[index]:
                continue
            is_fallback = (
                candidate.plan.mode == "act"
                and candidate.plan.action_index == self.fallback_action_index
            )
            if preserve_fallback and is_fallback:
                continue
            for other_index, other in enumerate(ordered):
                if index == other_index:
                    continue
                if np.all(other.loss <= candidate.loss + ATOL) and np.any(
                    other.loss < candidate.loss - ATOL
                ):
                    keep[index] = False
                    break
        retained = tuple(item for item, flag in zip(ordered, keep, strict=True) if flag)
        if len(retained) > self.max_frontier_count:
            raise ValueError(
                "adaptive policy frontier exceeds max_frontier_count: "
                f"{len(retained)} > {self.max_frontier_count}"
            )
        return retained

    def enumerate(
        self,
        support: tuple[int, ...],
        remaining_probes: tuple[int, ...],
        depth: int,
        *,
        preserve_fallback: bool = False,
    ) -> tuple[_Candidate, ...]:
        key = (support, remaining_probes, depth)
        cached = self._memo.get(key)
        if cached is not None and not preserve_fallback:
            return cached

        support_array = np.asarray(support, dtype=np.int64)
        candidates: list[_Candidate] = []
        for action_index in range(self.terminal_losses.shape[1]):
            candidates.append(
                _Candidate(
                    AdaptivePlanV1(mode="act", action_index=action_index),
                    _immutable_array(
                        self.terminal_losses[support_array, action_index],
                        dtype=np.dtype(np.float64),
                    ),
                )
            )
            self.raw_plan_count += 1

        if depth > 0:
            for probe_index in remaining_probes:
                probabilities = self.probe_probabilities[probe_index][support_array]
                child_probe_tuple = tuple(
                    item for item in remaining_probes if item != probe_index
                )
                child_frontiers: list[tuple[_Candidate, ...]] = []
                possible = True
                for outcome_index in range(probabilities.shape[1]):
                    active_local = np.flatnonzero(probabilities[:, outcome_index] > 0.0)
                    if active_local.size == 0:
                        # A globally impossible outcome still receives a deterministic
                        # placeholder branch; it contributes zero expected loss.
                        action = AdaptivePlanV1(
                            mode="act", action_index=self.fallback_action_index
                        )
                        child_frontiers.append(
                            (
                                _Candidate(
                                    action,
                                    _immutable_array(
                                        np.zeros(len(support)),
                                        dtype=np.dtype(np.float64),
                                    ),
                                ),
                            )
                        )
                        continue
                    child_support = tuple(support[int(i)] for i in active_local)
                    child_candidates = self.enumerate(
                        child_support,
                        child_probe_tuple,
                        depth - 1,
                    )
                    if not child_candidates:
                        possible = False
                        break
                    lifted: list[_Candidate] = []
                    for child in child_candidates:
                        full_loss = np.zeros(len(support), dtype=np.float64)
                        full_loss[active_local] = child.loss
                        lifted.append(
                            _Candidate(
                                child.plan,
                                _immutable_array(full_loss, dtype=np.dtype(np.float64)),
                            )
                        )
                    child_frontiers.append(tuple(lifted))
                if not possible:
                    continue

                projected = math.prod(len(frontier) for frontier in child_frontiers)
                if projected + self.raw_plan_count > self.max_frontier_count * 100:
                    raise ValueError(
                        "raw adaptive policy product exceeds safety cap before "
                        "pruning: "
                        f"{projected} branch combinations"
                    )
                for combination in itertools.product(*child_frontiers):
                    expected = np.full(
                        len(support),
                        float(self.probe_costs[probe_index]),
                        dtype=np.float64,
                    )
                    for outcome_index, child in enumerate(combination):
                        expected += probabilities[:, outcome_index] * child.loss
                    candidates.append(
                        _Candidate(
                            AdaptivePlanV1(
                                mode="sense",
                                probe_index=probe_index,
                                children_by_outcome=tuple(
                                    child.plan for child in combination
                                ),
                            ),
                            _immutable_array(expected, dtype=np.dtype(np.float64)),
                        )
                    )
                    self.raw_plan_count += 1

        retained = self._deduplicate_and_prune(
            candidates,
            preserve_fallback=preserve_fallback,
        )
        if not preserve_fallback:
            self._memo[key] = retained
        return retained


def _validate_probe_probabilities(
    value: Sequence[object],
    *,
    hypothesis_count: int,
) -> tuple[FloatArray, ...]:
    result: list[FloatArray] = []
    for probe_index, raw in enumerate(value):
        probabilities = _float_array(
            raw,
            ndim=2,
            name=f"probe_outcome_probability[{probe_index}]",
        )
        if probabilities.shape[0] != hypothesis_count or probabilities.shape[1] < 1:
            raise ValueError(
                f"probe {probe_index} probabilities must have shape "
                f"({hypothesis_count}, positive_outcome_count)"
            )
        if np.any(probabilities < 0.0):
            raise ValueError("probe probabilities must be nonnegative")
        row_sums = np.sum(probabilities, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "each hypothesis/probe outcome distribution must sum to one"
            )
        result.append(_immutable_array(probabilities, dtype=np.dtype(np.float64)))
    return tuple(result)


def _exact_quotient_regret(
    loss_matrix: FloatArray,
    prior_weights: FloatArray,
    quotient_weights: FloatArray,
    class_index: IntArray,
) -> tuple[FloatArray, FloatArray]:
    hypothesis_count, plan_count = loss_matrix.shape
    support = prior_weights > 0.0
    class_count = quotient_weights.size
    gaps = np.empty((plan_count, plan_count), dtype=np.float64)
    for candidate in range(plan_count):
        for comparator in range(plan_count):
            if candidate == comparator:
                gaps[candidate, comparator] = 0.0
                continue
            total = 0.0
            difference = loss_matrix[:, candidate] - loss_matrix[:, comparator]
            for class_id in range(class_count):
                members = support & (class_index == class_id)
                if quotient_weights[class_id] > ATOL and not np.any(members):
                    raise ValueError(
                        "positive quotient mass has no prior-supported hypothesis"
                    )
                if np.any(members):
                    total += float(quotient_weights[class_id]) * float(
                        np.max(difference[members])
                    )
            gaps[candidate, comparator] = total
    regrets = np.maximum(np.max(gaps, axis=1), 0.0)
    return (
        _immutable_array(gaps, dtype=np.dtype(np.float64)),
        _immutable_array(regrets, dtype=np.dtype(np.float64)),
    )


def adaptive_act_sense_fallback_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    terminal_loss_by_hypothesis_action: object,
    probe_outcome_probability: Sequence[object],
    probe_costs: object,
    *,
    fallback_action_index: int,
    max_depth: int,
    regret_tolerance: float,
    max_frontier_count: int = 20_000,
) -> AdaptiveActSenseFallbackCertificateV1:
    """Enumerate and certify finite-horizon stochastic sensing policies exactly."""

    losses = _float_array(
        terminal_loss_by_hypothesis_action,
        ndim=2,
        name="terminal_loss_by_hypothesis_action",
    )
    if losses.shape[0] < 1 or losses.shape[1] < 2:
        raise ValueError("terminal loss matrix must contain hypotheses and >=2 actions")
    hypothesis_count, action_count = losses.shape
    if (
        isinstance(fallback_action_index, (bool, np.bool_))
        or not isinstance(fallback_action_index, Integral)
        or int(fallback_action_index) < 0
        or int(fallback_action_index) >= action_count
    ):
        raise ValueError("fallback_action_index is outside the action roster")
    depth = _nonnegative_integer(max_depth, name="max_depth")
    frontier_cap = _positive_integer(max_frontier_count, name="max_frontier_count")
    tolerance = _nonnegative_real(regret_tolerance, name="regret_tolerance")
    probabilities = _validate_probe_probabilities(
        probe_outcome_probability,
        hypothesis_count=hypothesis_count,
    )
    costs = _float_array(probe_costs, ndim=1, name="probe_costs")
    if costs.size != len(probabilities) or np.any(costs < 0.0):
        raise ValueError("probe_costs must match probes and be nonnegative")
    if depth > len(probabilities):
        raise ValueError("max_depth cannot exceed the number of nonrepeatable probes")

    prior = _probability_vector(
        prior_weights,
        size=hypothesis_count,
        name="prior_weights",
    )
    classes_raw = np.asarray(class_index)
    if classes_raw.dtype.kind not in "iu":
        raise ValueError("class_index must contain integers")
    classes = np.asarray(classes_raw, dtype=np.int64)
    if classes.ndim != 1 or classes.size != hypothesis_count or np.any(classes < 0):
        raise ValueError("class_index must be a nonnegative hypothesis-length vector")
    class_count = int(np.max(classes)) + 1
    if not np.array_equal(np.unique(classes), np.arange(class_count)):
        raise ValueError("class_index labels must be contiguous from zero")
    quotient = _probability_vector(
        quotient_weights,
        size=class_count,
        name="quotient_weights",
    )
    classes = _immutable_array(classes, dtype=np.dtype(np.int64))

    enumerator = _Enumerator(
        losses,
        probabilities,
        costs,
        max_depth=depth,
        max_frontier_count=frontier_cap,
        fallback_action_index=int(fallback_action_index),
    )
    support = tuple(int(index) for index in np.flatnonzero(prior > 0.0))
    candidates = enumerator.enumerate(
        support,
        tuple(range(len(probabilities))),
        depth,
        preserve_fallback=True,
    )
    if not candidates:
        raise RuntimeError("adaptive policy enumeration returned no plans")

    plans = tuple(candidate.plan for candidate in candidates)
    support_loss = np.column_stack([candidate.loss for candidate in candidates])
    full_loss = np.full((hypothesis_count, len(candidates)), np.nan, dtype=np.float64)
    full_loss[np.asarray(support, dtype=np.int64)] = support_loss
    # Zero-prior rows never enter the certificate. Preserve finite storage by
    # copying their terminal fallback loss across the roster.
    zero_support = np.flatnonzero(prior <= 0.0)
    if zero_support.size:
        full_loss[zero_support] = losses[zero_support, int(fallback_action_index), None]
    loss_matrix = _immutable_array(full_loss, dtype=np.dtype(np.float64))
    gaps, regrets = _exact_quotient_regret(loss_matrix, prior, quotient, classes)

    fallback_matches = [
        index
        for index, plan in enumerate(plans)
        if plan.mode == "act" and plan.action_index == int(fallback_action_index)
    ]
    if len(fallback_matches) != 1:
        raise RuntimeError("exactly one direct fallback plan must be retained")
    fallback_plan_index = fallback_matches[0]

    minimum = float(np.min(regrets))
    minimizers = np.isclose(regrets, minimum, rtol=0.0, atol=ATOL)
    minimizer_indices = np.flatnonzero(minimizers)
    candidate_plan_index = (
        int(minimizer_indices[0]) if minimizer_indices.size == 1 else None
    )
    if (
        candidate_plan_index is not None
        and regrets[candidate_plan_index] <= tolerance + ATOL
    ):
        output_plan_index = candidate_plan_index
    else:
        output_plan_index = fallback_plan_index
    output_plan = plans[output_plan_index]
    if output_plan_index == fallback_plan_index:
        output_mode: Literal["act", "sense", "fallback"] = "fallback"
    else:
        output_mode = output_plan.mode

    return AdaptiveActSenseFallbackCertificateV1(
        plans=plans,
        loss_by_hypothesis_plan=loss_matrix,
        pairwise_worst_case_gap=gaps,
        worst_case_regret=regrets,
        minimizer_mask=_immutable_array(minimizers, dtype=np.dtype(np.bool_)),
        fallback_plan_index=fallback_plan_index,
        candidate_plan_index=candidate_plan_index,
        output_plan_index=output_plan_index,
        output_mode=output_mode,
        used_fallback=output_plan_index == fallback_plan_index,
        regret_tolerance=tolerance,
        raw_plan_count=enumerator.raw_plan_count,
        retained_plan_count=len(candidates),
    )


def _noisy_bit_probe(
    hypotheses: Sequence[tuple[int, ...]],
    bit_index: int,
    accuracy: float,
) -> FloatArray:
    probabilities = np.empty((len(hypotheses), 2), dtype=np.float64)
    for index, hypothesis in enumerate(hypotheses):
        bit = hypothesis[bit_index]
        probabilities[index, bit] = accuracy
        probabilities[index, 1 - bit] = 1.0 - accuracy
    return probabilities


def controlled_stochastic_xor_demo() -> dict[str, object]:
    """Return a reproducible case where two probes identify an action, not state."""

    hypotheses = tuple(itertools.product((0, 1), repeat=3))
    correct = np.asarray([x ^ y for x, y, _ in hypotheses], dtype=np.int64)
    losses = np.empty((len(hypotheses), 3), dtype=np.float64)
    losses[:, 0] = (correct != 0).astype(np.float64)
    losses[:, 1] = (correct != 1).astype(np.float64)
    losses[:, 2] = 0.45
    probes = (
        _noisy_bit_probe(hypotheses, 0, 0.95),
        _noisy_bit_probe(hypotheses, 1, 0.95),
        _noisy_bit_probe(hypotheses, 2, 0.99),
    )
    common = dict(
        prior_weights=np.full(8, 1.0 / 8.0),
        quotient_weights=[1.0],
        class_index=np.zeros(8, dtype=np.int64),
        terminal_loss_by_hypothesis_action=losses,
        probe_outcome_probability=probes,
        probe_costs=[0.05, 0.05, 0.01],
        fallback_action_index=2,
        regret_tolerance=0.25,
        max_frontier_count=20_000,
    )
    one = adaptive_act_sense_fallback_certificate(max_depth=1, **common)
    two = adaptive_act_sense_fallback_certificate(max_depth=2, **common)

    result = {
        "schema": "adaptive-act-sense-fallback-stochastic-xor-v1",
        "hypothesis_count": 8,
        "state_nuisance": "third latent bit is irrelevant to the action",
        "probe_accuracy": {"x": 0.95, "y": 0.95, "nuisance": 0.99},
        "probe_costs": {"x": 0.05, "y": 0.05, "nuisance": 0.01},
        "fallback_loss": 0.45,
        "regret_tolerance": 0.25,
        "one_probe": one.summary(),
        "two_probe": two.summary(),
        "two_probe_loss_range": [
            float(np.min(two.loss_by_hypothesis_plan[:, two.output_plan_index])),
            float(np.max(two.loss_by_hypothesis_plan[:, two.output_plan_index])),
        ],
        "two_probe_plan_key": two.output_plan.canonical_key(),
        "state_remains_unidentified": True,
        "interpretation": (
            "One noisy bit cannot resolve the XOR action, so the exact policy falls "
            "back. Two task-relevant probes support a certified plan while the "
            "nuisance bit and all physical hypotheses remain possible. The cheaper "
            "high-accuracy nuisance probe is not selected."
        ),
    }
    result["result_id"] = hashlib.sha256(json_bytes(result)).hexdigest()
    return result


def _best_fixed_two_probe_loss(
    terminal_losses: FloatArray,
    probe_probabilities: Sequence[FloatArray],
    probe_costs: Sequence[float],
) -> dict[str, object]:
    """Brute-force the best nonadaptive ordered two-probe policy.

    The same second probe is used for every first-probe outcome. Terminal
    actions may depend on both observed outcomes. This is a controlled-demo
    diagnostic, not part of the general certificate implementation.
    """

    hypothesis_count, action_count = terminal_losses.shape
    best_loss = math.inf
    best: tuple[int, int, tuple[int, ...]] | None = None
    for first, second in itertools.permutations(
        range(len(probe_probabilities)), 2
    ):
        first_probability = probe_probabilities[first]
        second_probability = probe_probabilities[second]
        outcome_count = first_probability.shape[1] * second_probability.shape[1]
        for mapping in itertools.product(range(action_count), repeat=outcome_count):
            loss = np.full(
                hypothesis_count,
                float(probe_costs[first]) + float(probe_costs[second]),
                dtype=np.float64,
            )
            for hypothesis in range(hypothesis_count):
                offset = 0
                for first_outcome in range(first_probability.shape[1]):
                    for second_outcome in range(second_probability.shape[1]):
                        action = mapping[offset]
                        probability = (
                            first_probability[hypothesis, first_outcome]
                            * second_probability[hypothesis, second_outcome]
                        )
                        loss[hypothesis] += (
                            probability * terminal_losses[hypothesis, action]
                        )
                        offset += 1
            worst = float(np.max(loss))
            candidate = (first, second, tuple(int(item) for item in mapping))
            if worst < best_loss - ATOL or (
                abs(worst - best_loss) <= ATOL
                and (best is None or candidate < best)
            ):
                best_loss = worst
                best = candidate
    if best is None:
        raise RuntimeError("fixed two-probe search returned no policy")
    return {
        "worst_case_loss": best_loss,
        "first_probe_index": best[0],
        "second_probe_index": best[1],
        "terminal_action_by_joint_outcome": best[2],
    }


def controlled_adaptive_router_demo() -> dict[str, object]:
    """Show why a complete adaptive policy tree is stronger than a probe list.

    The first noisy probe indicates which of two latent task bits controls the
    action. The second probe is therefore selected conditionally on the first
    outcome. A one-probe policy must fall back; a depth-two policy is certified
    while a fourth, cheaper and more accurate nuisance probe is ignored.
    """

    hypotheses = tuple(itertools.product((0, 1), repeat=4))
    correct = np.asarray(
        [x if router == 0 else y for router, x, y, _ in hypotheses],
        dtype=np.int64,
    )
    losses = np.empty((len(hypotheses), 3), dtype=np.float64)
    losses[:, 0] = (correct != 0).astype(np.float64)
    losses[:, 1] = (correct != 1).astype(np.float64)
    losses[:, 2] = 0.45
    probes = (
        _noisy_bit_probe(hypotheses, 0, 0.99),
        _noisy_bit_probe(hypotheses, 1, 0.95),
        _noisy_bit_probe(hypotheses, 2, 0.95),
        _noisy_bit_probe(hypotheses, 3, 0.999),
    )
    common = dict(
        prior_weights=np.full(16, 1.0 / 16.0),
        quotient_weights=[1.0],
        class_index=np.zeros(16, dtype=np.int64),
        terminal_loss_by_hypothesis_action=losses,
        probe_outcome_probability=probes,
        probe_costs=[0.02, 0.05, 0.05, 0.005],
        fallback_action_index=2,
        regret_tolerance=0.25,
        max_frontier_count=50_000,
    )
    direct = adaptive_act_sense_fallback_certificate(max_depth=0, **common)
    one = adaptive_act_sense_fallback_certificate(max_depth=1, **common)
    two = adaptive_act_sense_fallback_certificate(max_depth=2, **common)
    fixed_two = _best_fixed_two_probe_loss(losses, probes, common["probe_costs"])
    key = two.output_plan.canonical_key()
    expected_key_prefix = (1, 0)
    if key[:2] != expected_key_prefix:
        raise RuntimeError(
            "adaptive router plan no longer starts with the router probe"
        )
    second_probes = tuple(
        child.probe_index for child in two.output_plan.children_by_outcome
    )
    if second_probes != (1, 2):
        raise RuntimeError(
            "adaptive router plan no longer chooses the task bit by router outcome"
        )
    selected_loss = two.loss_by_hypothesis_plan[:, two.output_plan_index]
    result = {
        "schema": "adaptive-act-sense-fallback-stochastic-router-v1",
        "hypothesis_count": 16,
        "latent_state": ["router", "task_x", "task_y", "nuisance"],
        "correct_action": "task_x when router=0; task_y when router=1",
        "probe_accuracy": {
            "router": 0.99,
            "task_x": 0.95,
            "task_y": 0.95,
            "nuisance": 0.999,
        },
        "probe_costs": {
            "router": 0.02,
            "task_x": 0.05,
            "task_y": 0.05,
            "nuisance": 0.005,
        },
        "fallback_loss": 0.45,
        "regret_tolerance": 0.25,
        "direct_only": direct.summary(),
        "one_probe": one.summary(),
        "two_probe": two.summary(),
        "best_nonadaptive_fixed_two_probe": fixed_two,
        "adaptive_worst_case_improvement_over_best_fixed_two_probe": (
            float(fixed_two["worst_case_loss"])
            - float(two.worst_case_regret[two.output_plan_index])
        ),
        "two_probe_loss_range": [
            float(np.min(selected_loss)),
            float(np.max(selected_loss)),
        ],
        "selected_policy_tree": key,
        "selected_second_probe_by_router_outcome": second_probes,
        "state_remains_unidentified": True,
        "nuisance_probe_selected": 3 in two.output_plan.probe_indices,
        "interpretation": (
            "The exact pre-probe policy first measures the routing variable, then "
            "measures task_x or task_y according to the observed route. No direct "
            "or one-probe policy meets the regret tolerance. The cheaper, more "
            "accurate nuisance probe is ignored, and stochastic outcomes leave the "
            "complete physical state unidentified."
        ),
    }
    result["result_id"] = hashlib.sha256(json_bytes(result)).hexdigest()
    return result


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY",
    "ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_SEMANTICS",
    "ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_VERSION",
    "AdaptiveActSenseFallbackCertificateV1",
    "AdaptivePlanV1",
    "adaptive_act_sense_fallback_certificate",
    "controlled_adaptive_router_demo",
    "controlled_stochastic_xor_demo",
]


if __name__ == "__main__":
    print(json.dumps(controlled_adaptive_router_demo(), indent=2, sort_keys=True))
