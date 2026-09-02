"""Trajectory-conformal support enlargement for finite-action certificates.

The finite-support certificate in :mod:`query_decision_certificate_v1` is exact
for its registered physical hypotheses. A new physical trajectory may lie
outside that support. This module calibrates the *regret excess* beyond the
finite-support bound at the complete-trajectory level and uses the resulting
split-conformal radius to gate departures from a caller-owned fallback.

For calibration trajectory ``j``, let

    S_j = max_d [regret_jd(a_jd) - B_jd(a_jd)]_+

where the maximum is over decisions at which a fixed base policy selects a
nonfallback action and ``B`` is that action's registered finite-support regret
bound. With ``n`` exchangeable calibration trajectories, the split-conformal
radius is the ``ceil((n + 1) * (1 - alpha))`` order statistic, or infinity when
that order statistic is unavailable. For one new exchangeable trajectory,

    P(S_new <= radius) >= 1 - alpha.

Consequently, executing a base nonfallback action only when
``B + radius <= epsilon`` controls the probability that *any* executed action
on the new trajectory has realized regret above ``epsilon`` by ``alpha``.
This is a marginal trajectory-level guarantee for the fixed base policy. It is
not pointwise conditional validity and it does not cover arbitrary domain shift.
"""

from __future__ import annotations

from math import ceil
from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

SUPPORT_ROBUST_DECISION_CERTIFICATE_VERSION: Final = 1
SUPPORT_ROBUST_DECISION_CERTIFICATE_SEMANTICS: Final = (
    "split-conformal-complete-trajectory-regret-excess-envelope-v1"
)
SUPPORT_ROBUST_DECISION_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The envelope gives a marginal finite-sample guarantee for one new complete "
    "trajectory exchangeable with the calibration trajectories and for the fixed "
    "base selection policy used to define the calibration scores. It does not "
    "provide individual conditional validity, validate the physical support or "
    "loss model, cover arbitrary object/action/domain shift, calibrate a predictive "
    "distribution, certify deployment safety, or authorize robot execution."
)
CALIBRATION_UNIT: Final = "complete_trajectory"
_NUMERICAL_ATOL: Final = 1e-12


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_int64(value: object) -> IntArray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    array.setflags(write=False)
    return array


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


def _unit_interval_open(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number strictly between zero and one")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be a real number strictly between zero and one")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _nonnegative_radius(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("conformal_radius must be a nonnegative real number or +infinity")
    result = float(value)
    if np.isnan(result) or result < 0.0 or result == -np.inf:
        raise ValueError("conformal_radius must be a nonnegative real number or +infinity")
    return result


def _action_index(value: object, *, name: str, action_count: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer action index")
    result = int(raw)
    if result < 0 or result >= action_count:
        raise ValueError(f"{name} must lie in [0, {action_count})")
    return result


class SplitConformalTrajectoryEnvelopeV1(NamedTuple):
    """Split-conformal radius for complete-trajectory regret excess."""

    calibration_scores: FloatArray
    miscoverage: float
    order_statistic: int
    radius: float
    finite: bool

    @property
    def calibration_trajectory_count(self) -> int:
        return int(self.calibration_scores.size)

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.miscoverage

    def summary(self) -> dict[str, object]:
        return {
            "version": SUPPORT_ROBUST_DECISION_CERTIFICATE_VERSION,
            "semantics": SUPPORT_ROBUST_DECISION_CERTIFICATE_SEMANTICS,
            "calibration_unit": CALIBRATION_UNIT,
            "calibration_trajectory_count": self.calibration_trajectory_count,
            "miscoverage": self.miscoverage,
            "nominal_coverage": self.nominal_coverage,
            "order_statistic": self.order_statistic,
            "radius": self.radius,
            "finite": self.finite,
            "claim_boundary": SUPPORT_ROBUST_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
        }


class TrajectoryPolicyRegretExcessV1(NamedTuple):
    """One complete trajectory's nonconformity score for a fixed base policy."""

    realized_regret: FloatArray
    finite_support_regret_bound: FloatArray
    regret_excess: FloatArray
    nonfallback_mask: BoolArray
    score: float

    @property
    def decision_count(self) -> int:
        return int(self.realized_regret.size)

    @property
    def nonfallback_count(self) -> int:
        return int(np.count_nonzero(self.nonfallback_mask))


class SupportRobustActionDecisionV1(NamedTuple):
    """Fail-closed wrapper around one base finite-support action decision."""

    base_selected_action_index: int
    fallback_action_index: int
    finite_support_regret_bound: float
    conformal_radius: float
    support_robust_regret_bound: float
    operational_regret_tolerance: float
    execute_base_nonfallback: bool
    returned_action_index: int
    reason: str

    def summary(self) -> dict[str, object]:
        return {
            "version": SUPPORT_ROBUST_DECISION_CERTIFICATE_VERSION,
            "semantics": SUPPORT_ROBUST_DECISION_CERTIFICATE_SEMANTICS,
            "base_selected_action_index": self.base_selected_action_index,
            "fallback_action_index": self.fallback_action_index,
            "finite_support_regret_bound": self.finite_support_regret_bound,
            "conformal_radius": self.conformal_radius,
            "support_robust_regret_bound": self.support_robust_regret_bound,
            "operational_regret_tolerance": self.operational_regret_tolerance,
            "execute_base_nonfallback": self.execute_base_nonfallback,
            "returned_action_index": self.returned_action_index,
            "reason": self.reason,
            "claim_boundary": SUPPORT_ROBUST_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
        }


def split_conformal_trajectory_envelope(
    calibration_scores: object,
    *,
    miscoverage: float,
) -> SplitConformalTrajectoryEnvelopeV1:
    """Calibrate a conservative complete-trajectory regret-excess radius.

    The returned ``order_statistic`` is one-indexed. If the requested finite
    order statistic is larger than the calibration sample, ``radius`` is
    ``+inf`` and every nonfallback action must be rejected by the operational
    wrapper. This is the finite-sample-correct behavior for small calibration
    sets; the function never silently clips the rank to the largest observation.
    """

    raw = np.asarray(calibration_scores)
    if raw.dtype.kind not in "iuf":
        raise ValueError("calibration_scores must contain real numeric values")
    scores = np.ascontiguousarray(raw, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("calibration_scores must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration_scores must be finite")
    if np.any(scores < 0.0):
        raise ValueError("calibration_scores must be nonnegative")
    alpha = _unit_interval_open(miscoverage, name="miscoverage")

    order = int(ceil((scores.size + 1) * (1.0 - alpha)))
    if order > scores.size:
        radius = float("inf")
        finite = False
    else:
        radius = float(np.partition(scores, order - 1)[order - 1])
        finite = True

    return SplitConformalTrajectoryEnvelopeV1(
        calibration_scores=_immutable_float64(scores),
        miscoverage=alpha,
        order_statistic=order,
        radius=radius,
        finite=finite,
    )


def trajectory_policy_regret_excess(
    realized_loss_by_decision_action: object,
    finite_support_regret_by_decision_action: object,
    selected_action_index: object,
    *,
    fallback_action_index: int,
) -> TrajectoryPolicyRegretExcessV1:
    """Return one trajectory's maximum regret excess for a fixed base policy.

    Decisions at which the base policy returns the fallback do not contribute
    to the score. If every decision falls back, the trajectory score is zero.
    This definition makes the conformal guarantee policy-specific: changing the
    base selection rule requires recalibration or a stronger all-action score.
    """

    raw_losses = np.asarray(realized_loss_by_decision_action)
    raw_bounds = np.asarray(finite_support_regret_by_decision_action)
    if raw_losses.dtype.kind not in "iuf" or raw_bounds.dtype.kind not in "iuf":
        raise ValueError("losses and finite-support regret bounds must be real numeric arrays")
    losses = np.ascontiguousarray(raw_losses, dtype=np.float64)
    bounds = np.ascontiguousarray(raw_bounds, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] < 2:
        raise ValueError(
            "realized_loss_by_decision_action must have shape (decision_count, action_count>=2)"
        )
    if bounds.shape != losses.shape:
        raise ValueError(
            "finite_support_regret_by_decision_action must match the realized loss shape"
        )
    if not np.all(np.isfinite(losses)):
        raise ValueError("realized losses must be finite")
    if not np.all(np.isfinite(bounds)) or np.any(bounds < 0.0):
        raise ValueError("finite-support regret bounds must be finite and nonnegative")

    raw_actions = np.asarray(selected_action_index)
    if raw_actions.dtype.kind not in "iu":
        raise ValueError("selected_action_index must contain integer action indices")
    actions = np.ascontiguousarray(raw_actions, dtype=np.int64)
    if actions.ndim != 1 or actions.shape[0] != losses.shape[0]:
        raise ValueError("selected_action_index must have one entry per decision")
    if np.any(actions < 0) or np.any(actions >= losses.shape[1]):
        raise ValueError("selected_action_index contains an out-of-range action")
    fallback = _action_index(
        fallback_action_index,
        name="fallback_action_index",
        action_count=losses.shape[1],
    )

    decision_index = np.arange(losses.shape[0], dtype=np.int64)
    selected_loss = losses[decision_index, actions]
    realized_regret = selected_loss - np.min(losses, axis=1)
    realized_regret = np.maximum(realized_regret, 0.0)
    selected_bound = bounds[decision_index, actions]
    nonfallback = actions != fallback
    excess = np.zeros(losses.shape[0], dtype=np.float64)
    excess[nonfallback] = np.maximum(
        realized_regret[nonfallback] - selected_bound[nonfallback],
        0.0,
    )
    score = float(np.max(excess, initial=0.0))

    return TrajectoryPolicyRegretExcessV1(
        realized_regret=_immutable_float64(realized_regret),
        finite_support_regret_bound=_immutable_float64(selected_bound),
        regret_excess=_immutable_float64(excess),
        nonfallback_mask=_immutable_bool(nonfallback),
        score=score,
    )


def support_robust_action_decision(
    *,
    base_selected_action_index: int,
    fallback_action_index: int,
    action_count: int,
    finite_support_regret_bound: float,
    conformal_radius: float,
    operational_regret_tolerance: float,
) -> SupportRobustActionDecisionV1:
    """Apply a calibrated open-world regret enlargement and exact fallback.

    The wrapper gates a *fixed* base action. It does not search for a different
    action after seeing the conformal radius, because policy-specific calibration
    would not justify such post-calibration reselection.
    """

    if isinstance(action_count, (bool, np.bool_)) or not isinstance(
        action_count, (int, np.integer)
    ):
        raise ValueError("action_count must be an integer of at least two")
    count = int(action_count)
    if count < 2:
        raise ValueError("action_count must be an integer of at least two")
    selected = _action_index(
        base_selected_action_index,
        name="base_selected_action_index",
        action_count=count,
    )
    fallback = _action_index(
        fallback_action_index,
        name="fallback_action_index",
        action_count=count,
    )
    support_bound = _finite_nonnegative(
        finite_support_regret_bound,
        name="finite_support_regret_bound",
    )
    radius = _nonnegative_radius(conformal_radius)
    tolerance = _finite_nonnegative(
        operational_regret_tolerance,
        name="operational_regret_tolerance",
    )
    robust_bound = support_bound + radius

    if selected == fallback:
        execute = False
        returned = fallback
        reason = "base_policy_fallback"
    elif robust_bound <= tolerance + _NUMERICAL_ATOL:
        execute = True
        returned = selected
        reason = "support_robust_regret_within_tolerance"
    else:
        execute = False
        returned = fallback
        reason = (
            "conformal_radius_unavailable"
            if not np.isfinite(radius)
            else "support_robust_regret_exceeds_tolerance"
        )

    return SupportRobustActionDecisionV1(
        base_selected_action_index=selected,
        fallback_action_index=fallback,
        finite_support_regret_bound=support_bound,
        conformal_radius=radius,
        support_robust_regret_bound=robust_bound,
        operational_regret_tolerance=tolerance,
        execute_base_nonfallback=execute,
        returned_action_index=returned,
        reason=reason,
    )


__all__ = [
    "CALIBRATION_UNIT",
    "SUPPORT_ROBUST_DECISION_CERTIFICATE_CLAIM_BOUNDARY",
    "SUPPORT_ROBUST_DECISION_CERTIFICATE_SEMANTICS",
    "SUPPORT_ROBUST_DECISION_CERTIFICATE_VERSION",
    "SplitConformalTrajectoryEnvelopeV1",
    "SupportRobustActionDecisionV1",
    "TrajectoryPolicyRegretExcessV1",
    "split_conformal_trajectory_envelope",
    "support_robust_action_decision",
    "trajectory_policy_regret_excess",
]
