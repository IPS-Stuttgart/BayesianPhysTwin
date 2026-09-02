"""Trajectory-level conformal envelopes for finite-action regret certificates.

The registered finite-support certificate supplies one upper regret bound per
action.  A calibration trajectory also reveals the realized regret of every
registered action.  For each calibration trajectory, this module records the
largest excess of realized regret over the registered bound across all of its
decisions and candidate actions.  A split-conformal order statistic then gives
a single nonnegative inflation radius.

Under exchangeability of complete calibration and future trajectories, the
inflated bound covers every candidate action at every decision of one future
trajectory with marginal probability at least ``1 - miscoverage``.  Therefore,
selecting an action only when its inflated bound is below a declared regret
budget controls the joint event "depart from fallback and exceed the budget".

The guarantee is trajectory-marginal and conditional on the declared action,
loss, trajectory, and exchangeability semantics.  It is not pointwise
conditional validity, an unseen-object guarantee, or deployment safety.
"""

from __future__ import annotations

from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

CONFORMAL_REGRET_ENVELOPE_VERSION: Final = 1
CONFORMAL_REGRET_ENVELOPE_SEMANTICS: Final = (
    "trajectory-split-conformal-simultaneous-action-regret-inflation-v1"
)
CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY: Final = (
    "Coverage is marginal over an exchangeable future complete trajectory and "
    "simultaneous only for the registered candidate actions and decisions on "
    "that trajectory. It does not provide pointwise conditional validity, "
    "validate exchangeability, establish unseen-object transport, justify the "
    "loss or regret budget, calibrate probabilities, authorize deployment, or "
    "certify safety."
)

_NUMERICAL_ATOL: Final = 1e-12


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


def _finite_unit_interval(value: object, *, name: str) -> float:
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


def _trajectory_tensor(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError(
            f"{name} must have shape (trajectory_count, decision_count, action_count)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return _immutable_float64(array)


def _mask(value: object | None, *, action_count: int, name: str) -> BoolArray:
    if value is None:
        return _immutable_bool(np.ones(action_count, dtype=np.bool_))
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean values")
    array = np.ascontiguousarray(raw, dtype=np.bool_)
    if array.shape != (action_count,) or not np.any(array):
        raise ValueError(f"{name} must select at least one of {action_count} actions")
    return _immutable_bool(array)


class TrajectoryConformalRegretEnvelopeV1(NamedTuple):
    """One-sided simultaneous regret inflation calibrated by trajectories."""

    miscoverage: float
    calibration_trajectory_count: int
    decision_count_per_trajectory: int
    action_count: int
    candidate_action_mask: BoolArray
    trajectory_nonconformity_scores: FloatArray
    finite_sample_rank: int
    radius: float

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.miscoverage

    @property
    def has_finite_radius(self) -> bool:
        return bool(np.isfinite(self.radius))

    def summary(self) -> dict[str, object]:
        return {
            "version": CONFORMAL_REGRET_ENVELOPE_VERSION,
            "semantics": CONFORMAL_REGRET_ENVELOPE_SEMANTICS,
            "miscoverage": self.miscoverage,
            "nominal_coverage": self.nominal_coverage,
            "calibration_trajectory_count": self.calibration_trajectory_count,
            "decision_count_per_trajectory": self.decision_count_per_trajectory,
            "action_count": self.action_count,
            "finite_sample_rank": self.finite_sample_rank,
            "radius": self.radius,
            "has_finite_radius": self.has_finite_radius,
            "claim_boundary": CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY,
        }


class SupportRobustDecisionV1(NamedTuple):
    """Fail-closed action choice from conformally inflated regret bounds."""

    registered_worst_case_regret: FloatArray
    conformal_radius: float
    inflated_regret_upper_bound: FloatArray
    regret_tolerance: float
    candidate_action_mask: BoolArray
    tolerance_admissible_action_mask: BoolArray
    selected_action_index: int
    fallback_action_index: int
    used_fallback: bool

    def summary(self) -> dict[str, object]:
        return {
            "version": CONFORMAL_REGRET_ENVELOPE_VERSION,
            "semantics": CONFORMAL_REGRET_ENVELOPE_SEMANTICS,
            "selected_action_index": self.selected_action_index,
            "fallback_action_index": self.fallback_action_index,
            "used_fallback": self.used_fallback,
            "conformal_radius": self.conformal_radius,
            "regret_tolerance": self.regret_tolerance,
            "admissible_action_count": int(
                np.count_nonzero(self.tolerance_admissible_action_mask)
            ),
            "claim_boundary": CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY,
        }


def trajectory_conformal_regret_envelope(
    realized_regret_by_trajectory: object,
    registered_upper_bound_by_trajectory: object,
    *,
    miscoverage: float,
    candidate_action_mask: object | None = None,
) -> TrajectoryConformalRegretEnvelopeV1:
    """Calibrate a nonnegative trajectory-level simultaneous inflation radius.

    Inputs have shape ``(trajectory, decision, action)``.  The nonconformity
    score for one trajectory is the maximum of
    ``realized_regret - registered_upper_bound`` over every decision and every
    selected candidate action.  The radius is the
    ``ceil((n + 1) * (1 - alpha))`` order statistic.  If that rank exceeds the
    number of calibration trajectories, the radius is infinite and every
    support-robust policy must fall back.
    """

    realized = _trajectory_tensor(
        realized_regret_by_trajectory,
        name="realized_regret_by_trajectory",
    )
    registered = _trajectory_tensor(
        registered_upper_bound_by_trajectory,
        name="registered_upper_bound_by_trajectory",
    )
    if realized.shape != registered.shape:
        raise ValueError(
            "realized and registered trajectory tensors must have equal shape"
        )
    alpha = _finite_unit_interval(miscoverage, name="miscoverage")
    mask = _mask(
        candidate_action_mask,
        action_count=realized.shape[2],
        name="candidate_action_mask",
    )

    excess = realized[:, :, mask] - registered[:, :, mask]
    scores = np.max(excess, axis=(1, 2))
    rank = int(np.ceil((realized.shape[0] + 1) * (1.0 - alpha)))
    if rank > realized.shape[0]:
        radius = float("inf")
    else:
        radius = max(0.0, float(np.partition(scores, rank - 1)[rank - 1]))

    return TrajectoryConformalRegretEnvelopeV1(
        miscoverage=alpha,
        calibration_trajectory_count=int(realized.shape[0]),
        decision_count_per_trajectory=int(realized.shape[1]),
        action_count=int(realized.shape[2]),
        candidate_action_mask=mask,
        trajectory_nonconformity_scores=_immutable_float64(scores),
        finite_sample_rank=rank,
        radius=radius,
    )


def support_robust_decision(
    registered_worst_case_regret: object,
    *,
    conformal_radius: float,
    regret_tolerance: float,
    fallback_action_index: int = 0,
    candidate_action_mask: object | None = None,
) -> SupportRobustDecisionV1:
    """Choose the unique lowest inflated-regret action or exact fallback.

    The fallback remains part of the registered action roster.  A nonfallback
    action is returned only when it is the unique minimum-regret action and its
    inflated bound is below ``regret_tolerance``.  If fallback minimizes the
    bound, the minimum is ambiguous, or the radius is infinite, the procedure
    returns the exact fallback.
    """

    raw = np.asarray(registered_worst_case_regret)
    if raw.dtype.kind not in "iuf":
        raise ValueError(
            "registered_worst_case_regret must contain real numeric values"
        )
    registered = np.ascontiguousarray(raw, dtype=np.float64)
    if registered.ndim != 1 or registered.size < 2:
        raise ValueError(
            "registered_worst_case_regret must contain at least two actions"
        )
    if not np.all(np.isfinite(registered)) or np.any(registered < 0.0):
        raise ValueError("registered_worst_case_regret must be finite and nonnegative")
    if isinstance(fallback_action_index, (bool, np.bool_)) or not isinstance(
        fallback_action_index, (int, np.integer)
    ):
        raise ValueError("fallback_action_index must be an integer")
    fallback = int(fallback_action_index)
    if not 0 <= fallback < registered.size:
        raise ValueError("fallback_action_index is out of range")
    if isinstance(conformal_radius, (bool, np.bool_)) or not isinstance(
        conformal_radius, Real
    ):
        raise ValueError(
            "conformal_radius must be a nonnegative real or positive infinity"
        )
    radius = float(conformal_radius)
    if np.isnan(radius) or radius < 0.0:
        raise ValueError(
            "conformal_radius must be a nonnegative real or positive infinity"
        )
    tolerance = _finite_nonnegative(regret_tolerance, name="regret_tolerance")
    candidates = np.asarray(
        _mask(
            candidate_action_mask,
            action_count=registered.size,
            name="candidate_action_mask",
        ),
        dtype=np.bool_,
    ).copy()
    if not candidates[fallback]:
        raise ValueError("candidate_action_mask must include fallback_action_index")

    inflated = registered + radius
    minimum = float(np.min(inflated[candidates]))
    minimizers = np.flatnonzero(
        candidates
        & np.isclose(inflated, minimum, rtol=0.0, atol=_NUMERICAL_ATOL)
    )
    admissible = np.zeros(registered.size, dtype=np.bool_)
    if np.isfinite(radius):
        admissible = candidates & (inflated <= tolerance + _NUMERICAL_ATOL)
    selected = fallback
    used_fallback = True
    if minimizers.size == 1:
        unique = int(minimizers[0])
        if unique != fallback and admissible[unique]:
            selected = unique
            used_fallback = False

    return SupportRobustDecisionV1(
        registered_worst_case_regret=_immutable_float64(registered),
        conformal_radius=radius,
        inflated_regret_upper_bound=_immutable_float64(inflated),
        regret_tolerance=tolerance,
        candidate_action_mask=_immutable_bool(candidates),
        tolerance_admissible_action_mask=_immutable_bool(admissible),
        selected_action_index=selected,
        fallback_action_index=fallback,
        used_fallback=used_fallback,
    )


__all__ = [
    "CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY",
    "CONFORMAL_REGRET_ENVELOPE_SEMANTICS",
    "CONFORMAL_REGRET_ENVELOPE_VERSION",
    "SupportRobustDecisionV1",
    "TrajectoryConformalRegretEnvelopeV1",
    "support_robust_decision",
    "trajectory_conformal_regret_envelope",
]
