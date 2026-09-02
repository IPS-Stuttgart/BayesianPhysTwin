"""Split-conformal regret envelopes for complete act--sense plans.

A contingent sensing plan is selected *before* the probe and contains a frozen
map from every registered probe outcome to a terminal physical action.  This
module calibrates those complete plans rather than recalibrating a terminal
action after observing the probe.

For calibration trajectory ``j``, registered decision ``d``, and complete plan
``p``, let ``R[j,d,p]`` be realized regret and ``B[j,d,p]`` a pre-outcome regret
upper bound.  Given positive, pre-registered plan scales ``s[p]``, define

    S[j] = max_{d,p in C} max(0, R[j,d,p] - B[j,d,p]) / s[p].

The split-conformal order statistic ``q`` gives simultaneous plan-wise bounds

    R[new,d,p] <= B[new,d,p] + q * s[p]

for every registered decision and candidate plan on one exchangeable future
trajectory, with marginal coverage at least ``1 - alpha``.  Plan-dependent
scales preserve the guarantee while allowing fragile and stable probes to
receive different inflation.  The scales, plan roster, candidate mask, and all
model choices must be fixed before calibration outcomes are used.

The guarantee is trajectory-marginal.  It does not validate exchangeability,
probe physics, reset semantics, support-miss assumptions, loss construction,
plan scales, calibration-group independence, deployment, or safety.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    ActSenseFallbackCertificateV1,
    ContingentPlanV1,
)
from bayesian_phystwin.support_robust_act_sense_fallback_certificate_v1 import (
    SupportRobustActSenseFallbackCertificateV1,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
BaseCertificate: TypeAlias = (
    ActSenseFallbackCertificateV1 | SupportRobustActSenseFallbackCertificateV1
)

CONFORMAL_COMPLETE_PLAN_CERTIFICATE_VERSION: Final = 1
CONFORMAL_COMPLETE_PLAN_CERTIFICATE_SEMANTICS: Final = (
    "trajectory-split-conformal-simultaneous-complete-plan-regret-v1"
)
CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The split-conformal statement is marginal over one exchangeable future "
    "trajectory and conditional on a plan roster, candidate mask, plan scales, "
    "registered regret bounds, probe-outcome semantics, and all tuning choices "
    "fixed before calibration outcomes. It does not validate exchangeability, "
    "independence of calibration groups, the represented or unknown physical "
    "support, probe physics, reset semantics, action losses, support-miss bounds, "
    "target transport, deployment, or safety."
)

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


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _open_unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in (0, 1)")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0 or result >= 1.0:
        raise ValueError(f"{name} must be a finite real number in (0, 1)")
    return result


def _finite_loss_tensor(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != 3 or any(size <= 0 for size in array.shape):
        raise ValueError(
            f"{name} must have shape "
            "(positive_trajectory_count, positive_decision_count, positive_count)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return _immutable_float64(array)


def _nonnegative_tensor(value: object, *, name: str) -> FloatArray:
    array = _finite_loss_tensor(value, name=name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain nonnegative values")
    return array


def _probe_outcome_tensor(
    value: object,
    *,
    trajectory_count: int,
    decision_count: int,
    probe_count: int,
) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(
            "probe_outcome_index_by_trajectory_decision_probe must contain integers"
        )
    outcomes = np.ascontiguousarray(raw, dtype=np.int64)
    expected = (trajectory_count, decision_count, probe_count)
    if outcomes.shape != expected:
        raise ValueError(
            "probe_outcome_index_by_trajectory_decision_probe must have shape "
            f"{expected}"
        )
    if np.any(outcomes < 0):
        raise ValueError("probe outcome indices must be nonnegative")
    return _immutable_int64(outcomes)


def _nonnegative_vector(value: object, *, size: int, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    vector = np.ascontiguousarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size != size:
        raise ValueError(f"{name} must contain exactly {size} entries")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return _immutable_float64(vector)


def _positive_vector(value: object, *, size: int, name: str) -> FloatArray:
    vector = _nonnegative_vector(value, size=size, name=name)
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain strictly positive values")
    return vector


def _candidate_mask(value: object | None, *, size: int) -> BoolArray:
    if value is None:
        return _immutable_bool(np.ones(size, dtype=np.bool_))
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError("candidate_plan_mask must contain booleans")
    mask = np.ascontiguousarray(raw, dtype=np.bool_)
    if mask.ndim != 1 or mask.size != size:
        raise ValueError(f"candidate_plan_mask must contain exactly {size} entries")
    if not np.any(mask):
        raise ValueError("candidate_plan_mask must select at least one plan")
    return _immutable_bool(mask)


def _certificate_parts(
    certificate: BaseCertificate,
) -> tuple[ActSenseFallbackCertificateV1, FloatArray, int, float]:
    if isinstance(certificate, SupportRobustActSenseFallbackCertificateV1):
        return (
            certificate.represented_certificate,
            certificate.support_robust_worst_case_regret,
            certificate.fallback_plan_index,
            certificate.regret_tolerance,
        )
    if isinstance(certificate, ActSenseFallbackCertificateV1):
        return (
            certificate,
            certificate.plan_certificate.worst_case_regret,
            certificate.fallback_plan_index,
            certificate.plan_certificate.regret_tolerance,
        )
    raise ValueError(
        "certificate must be an act-sense-fallback or support-robust certificate"
    )


class CompletePlanRegretTensorV1(NamedTuple):
    """Observed losses and regrets for every frozen complete plan."""

    represented_certificate: ActSenseFallbackCertificateV1
    probe_costs: FloatArray
    plan_loss_by_trajectory_decision_plan: FloatArray
    best_plan_loss_by_trajectory_decision: FloatArray
    realized_regret_by_trajectory_decision_plan: FloatArray

    @property
    def trajectory_count(self) -> int:
        return int(self.plan_loss_by_trajectory_decision_plan.shape[0])

    @property
    def decision_count(self) -> int:
        return int(self.plan_loss_by_trajectory_decision_plan.shape[1])

    @property
    def plan_count(self) -> int:
        return int(self.plan_loss_by_trajectory_decision_plan.shape[2])

    def summary(self) -> dict[str, object]:
        return {
            "version": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_VERSION,
            "trajectory_count": self.trajectory_count,
            "decision_count": self.decision_count,
            "plan_count": self.plan_count,
            "terminal_action_count": self.represented_certificate.direct_plan_count,
            "probe_count": self.represented_certificate.probe_count,
            "claim_boundary": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY,
        }


class ScaledTrajectoryConformalPlanEnvelopeV1(NamedTuple):
    """One split-conformal multiplier and plan-dependent inflation vector."""

    miscoverage: float
    candidate_plan_mask: BoolArray
    plan_scales: FloatArray
    trajectory_scores: FloatArray
    order_statistic_rank: int
    score_quantile: float
    inflation_by_plan: FloatArray
    finite_sample_coverage_lower_bound: float
    trajectory_count: int
    decision_count: int
    plan_count: int

    @property
    def has_finite_quantile(self) -> bool:
        return bool(np.isfinite(self.score_quantile))

    def summary(self) -> dict[str, object]:
        return {
            "version": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_VERSION,
            "semantics": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_SEMANTICS,
            "miscoverage": self.miscoverage,
            "trajectory_count": self.trajectory_count,
            "decision_count": self.decision_count,
            "plan_count": self.plan_count,
            "candidate_plan_count": int(np.count_nonzero(self.candidate_plan_mask)),
            "order_statistic_rank": self.order_statistic_rank,
            "score_quantile": self.score_quantile,
            "has_finite_quantile": self.has_finite_quantile,
            "finite_sample_coverage_lower_bound": (
                self.finite_sample_coverage_lower_bound
            ),
            "claim_boundary": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY,
        }


class ConformalActSenseFallbackDecisionV1(NamedTuple):
    """Calibrated complete-plan decision with exact caller-owned fallback."""

    certificate: BaseCertificate
    envelope: ScaledTrajectoryConformalPlanEnvelopeV1
    registered_worst_case_regret: FloatArray
    calibrated_regret_upper_by_plan: FloatArray
    regret_tolerance: float
    tolerance_admissible_plan_mask: BoolArray
    minimizer_plan_mask: BoolArray
    minimizer_count: int
    candidate_plan_index: int | None
    output_plan_index: int
    output_mode: str
    used_fallback: bool
    fallback_reason: str | None

    @property
    def plans(self) -> tuple[ContingentPlanV1, ...]:
        represented, _, _, _ = _certificate_parts(self.certificate)
        return represented.plans

    @property
    def output_plan(self) -> ContingentPlanV1:
        return self.plans[self.output_plan_index]

    @property
    def selected_probe_index(self) -> int | None:
        return self.output_plan.probe_index if self.output_mode == "sense" else None

    def terminal_action(self, outcome_index: int | None = None) -> int:
        """Resolve the frozen direct or outcome-contingent terminal action."""

        return self.output_plan.terminal_action(outcome_index)

    def summary(self) -> dict[str, object]:
        return {
            "version": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_VERSION,
            "semantics": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_SEMANTICS,
            "plan_count": len(self.plans),
            "regret_tolerance": self.regret_tolerance,
            "candidate_plan_index": self.candidate_plan_index,
            "minimizer_count": self.minimizer_count,
            "output_plan_index": self.output_plan_index,
            "output_mode": self.output_mode,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "selected_probe_index": self.selected_probe_index,
            "conformal_score_quantile": self.envelope.score_quantile,
            "finite_sample_coverage_lower_bound": (
                self.envelope.finite_sample_coverage_lower_bound
            ),
            "claim_boundary": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY,
        }


def complete_plan_regret_tensor(
    certificate: BaseCertificate,
    terminal_loss_by_trajectory_decision_action: object,
    probe_outcome_index_by_trajectory_decision_probe: object,
    probe_costs: object,
) -> CompletePlanRegretTensorV1:
    """Evaluate every complete direct or probe-contingent plan on logged data.

    The same frozen plan roster used before sensing is evaluated on every logged
    trajectory and decision.  A sensing plan pays its registered probe cost and
    then applies only its precommitted outcome-to-action map.  No post-outcome
    optimization is performed.
    """

    represented, _, _, _ = _certificate_parts(certificate)
    terminal = _finite_loss_tensor(
        terminal_loss_by_trajectory_decision_action,
        name="terminal_loss_by_trajectory_decision_action",
    )
    trajectory_count, decision_count, action_count = terminal.shape
    if action_count != represented.direct_plan_count:
        raise ValueError("terminal loss action count does not match the certificate")
    outcomes = _probe_outcome_tensor(
        probe_outcome_index_by_trajectory_decision_probe,
        trajectory_count=trajectory_count,
        decision_count=decision_count,
        probe_count=represented.probe_count,
    )
    costs = _nonnegative_vector(
        probe_costs,
        size=represented.probe_count,
        name="probe_costs",
    )

    plan_losses = np.empty(
        (trajectory_count, decision_count, represented.plan_count),
        dtype=np.float64,
    )
    for plan_index, plan in enumerate(represented.plans):
        if plan.mode == "act":
            if plan.direct_action_index is None:
                raise RuntimeError("direct plan is missing its terminal action")
            plan_losses[:, :, plan_index] = terminal[:, :, plan.direct_action_index]
            continue
        if plan.mode != "sense" or plan.probe_index is None:
            raise RuntimeError("certificate contains an invalid complete plan")
        mapping = plan.terminal_action_by_outcome
        if mapping.size == 0:
            raise RuntimeError("sensing plan contains no terminal action map")
        labels = outcomes[:, :, plan.probe_index]
        if np.any(labels >= mapping.size):
            raise ValueError(
                f"probe {plan.probe_index} produced an unregistered outcome index"
            )
        terminal_actions = mapping[labels]
        if np.any(terminal_actions < 0) or np.any(terminal_actions >= action_count):
            raise RuntimeError("sensing plan maps to an invalid terminal action")
        selected = np.take_along_axis(
            terminal,
            terminal_actions[:, :, None],
            axis=2,
        )[:, :, 0]
        plan_losses[:, :, plan_index] = costs[plan.probe_index] + selected

    best = np.min(plan_losses, axis=2)
    realized_regret = np.maximum(plan_losses - best[:, :, None], 0.0)
    return CompletePlanRegretTensorV1(
        represented_certificate=represented,
        probe_costs=costs,
        plan_loss_by_trajectory_decision_plan=_immutable_float64(plan_losses),
        best_plan_loss_by_trajectory_decision=_immutable_float64(best),
        realized_regret_by_trajectory_decision_plan=_immutable_float64(realized_regret),
    )


def scaled_trajectory_conformal_plan_envelope(
    realized_regret_by_trajectory_decision_plan: object,
    registered_regret_upper_by_trajectory_decision_plan: object,
    plan_scales: object,
    *,
    miscoverage: float,
    candidate_plan_mask: object | None = None,
) -> ScaledTrajectoryConformalPlanEnvelopeV1:
    """Calibrate simultaneous complete-plan regret inflation by trajectory.

    The calibration unit is a complete trajectory.  Within-trajectory decisions
    and plans are protected simultaneously by the trajectory-wise maximum score.
    If the finite-sample conformal rank exceeds the number of calibration
    trajectories, the returned quantile and all plan inflations are infinite.
    """

    realized = _nonnegative_tensor(
        realized_regret_by_trajectory_decision_plan,
        name="realized_regret_by_trajectory_decision_plan",
    )
    registered = _nonnegative_tensor(
        registered_regret_upper_by_trajectory_decision_plan,
        name="registered_regret_upper_by_trajectory_decision_plan",
    )
    if realized.shape != registered.shape:
        raise ValueError("realized and registered regret tensors must have equal shape")
    trajectory_count, decision_count, plan_count = realized.shape
    scales = _positive_vector(
        plan_scales,
        size=plan_count,
        name="plan_scales",
    )
    candidates = _candidate_mask(candidate_plan_mask, size=plan_count)
    alpha = _open_unit_interval(miscoverage, name="miscoverage")

    standardized_excess = (realized - registered) / scales[None, None, :]
    scores = np.max(
        standardized_excess[:, :, candidates],
        axis=(1, 2),
    )
    scores = np.maximum(scores, 0.0)
    rank = int(math.ceil((trajectory_count + 1) * (1.0 - alpha)))
    if rank > trajectory_count:
        quantile = math.inf
        inflation = np.full(plan_count, math.inf, dtype=np.float64)
        coverage = 1.0
    else:
        quantile = float(np.sort(scores)[rank - 1])
        inflation = quantile * scales
        coverage = rank / (trajectory_count + 1)

    return ScaledTrajectoryConformalPlanEnvelopeV1(
        miscoverage=alpha,
        candidate_plan_mask=candidates,
        plan_scales=scales,
        trajectory_scores=_immutable_float64(scores),
        order_statistic_rank=rank,
        score_quantile=quantile,
        inflation_by_plan=_immutable_float64(inflation),
        finite_sample_coverage_lower_bound=float(coverage),
        trajectory_count=trajectory_count,
        decision_count=decision_count,
        plan_count=plan_count,
    )


def support_robust_plan_width_scales(
    certificate: SupportRobustActSenseFallbackCertificateV1,
    *,
    minimum_scale: float,
) -> FloatArray:
    """Use the registered unknown-plan loss widths as pre-outcome scales."""

    floor = _finite_nonnegative(minimum_scale, name="minimum_scale")
    if floor <= 0.0:
        raise ValueError("minimum_scale must be strictly positive")
    width = certificate.unknown_plan_loss_upper - certificate.unknown_plan_loss_lower
    return _immutable_float64(np.maximum(width, floor))


def conformal_act_sense_fallback_decision(
    certificate: BaseCertificate,
    envelope: ScaledTrajectoryConformalPlanEnvelopeV1,
    *,
    regret_tolerance: float | None = None,
) -> ConformalActSenseFallbackDecisionV1:
    """Choose one uniquely calibrated plan or reproduce fallback exactly.

    The selected plan minimizes its calibrated regret upper bound among the
    envelope's candidate plans.  A nonfallback plan is returned only when that
    minimizer is unique and its calibrated upper bound is within the registered
    tolerance.  Otherwise the caller-owned fallback plan is reproduced exactly.
    """

    represented, registered_value, fallback, default_tolerance = _certificate_parts(
        certificate
    )
    registered = _nonnegative_vector(
        registered_value,
        size=represented.plan_count,
        name="registered_worst_case_regret",
    )
    if envelope.plan_count != represented.plan_count:
        raise ValueError("conformal envelope plan count does not match certificate")
    tolerance = (
        default_tolerance
        if regret_tolerance is None
        else _finite_nonnegative(regret_tolerance, name="regret_tolerance")
    )
    calibrated = registered + envelope.inflation_by_plan
    candidates = envelope.candidate_plan_mask
    finite_candidates = candidates & np.isfinite(calibrated)
    minimizer_mask = np.zeros(represented.plan_count, dtype=np.bool_)
    candidate_plan_index: int | None
    if np.any(finite_candidates):
        minimum = float(np.min(calibrated[finite_candidates]))
        minimizer_mask = finite_candidates & np.isclose(
            calibrated,
            minimum,
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        )
        candidate_plan_index = int(np.flatnonzero(minimizer_mask)[0])
    else:
        candidate_plan_index = None
    minimizer_count = int(np.count_nonzero(minimizer_mask))
    admissible = candidates & (calibrated <= tolerance + _NUMERICAL_ATOL)

    if (
        candidate_plan_index is not None
        and minimizer_count == 1
        and admissible[candidate_plan_index]
    ):
        output_plan_index = candidate_plan_index
        fallback_reason = None
    else:
        output_plan_index = fallback
        if candidate_plan_index is None:
            fallback_reason = "infinite-conformal-envelope"
        elif minimizer_count != 1:
            fallback_reason = "nonunique-calibrated-minimax-plan"
        else:
            fallback_reason = "calibrated-regret-exceeds-tolerance"

    output_plan = represented.plans[output_plan_index]
    if output_plan_index == fallback:
        output_mode = "fallback"
        used_fallback = True
        if fallback_reason is None:
            fallback_reason = "registered-fallback-is-calibrated-minimax-plan"
    elif output_plan.mode == "sense":
        output_mode = "sense"
        used_fallback = False
    elif output_plan.mode == "act":
        output_mode = "act"
        used_fallback = False
    else:
        raise RuntimeError("certificate returned an unsupported plan mode")

    if (
        output_mode != "fallback"
        and calibrated[output_plan_index] > tolerance + _NUMERICAL_ATOL
    ):
        raise RuntimeError("nonfallback plan exceeds calibrated regret tolerance")
    if output_mode == "fallback" and output_plan_index != fallback:
        raise RuntimeError("fallback output does not reproduce the caller-owned plan")

    return ConformalActSenseFallbackDecisionV1(
        certificate=certificate,
        envelope=envelope,
        registered_worst_case_regret=registered,
        calibrated_regret_upper_by_plan=_immutable_float64(calibrated),
        regret_tolerance=tolerance,
        tolerance_admissible_plan_mask=_immutable_bool(admissible),
        minimizer_plan_mask=_immutable_bool(minimizer_mask),
        minimizer_count=minimizer_count,
        candidate_plan_index=candidate_plan_index,
        output_plan_index=output_plan_index,
        output_mode=output_mode,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
    )


__all__ = [
    "CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY",
    "CONFORMAL_COMPLETE_PLAN_CERTIFICATE_SEMANTICS",
    "CONFORMAL_COMPLETE_PLAN_CERTIFICATE_VERSION",
    "CompletePlanRegretTensorV1",
    "ConformalActSenseFallbackDecisionV1",
    "ScaledTrajectoryConformalPlanEnvelopeV1",
    "complete_plan_regret_tensor",
    "conformal_act_sense_fallback_decision",
    "scaled_trajectory_conformal_plan_envelope",
    "support_robust_plan_width_scales",
]
