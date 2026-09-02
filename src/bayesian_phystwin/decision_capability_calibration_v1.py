"""Finite-group calibration for affine decision-capability atlases.

A decision-capability atlas is exact relative to a registered finite physical
support. This module adds a separate finite-sample correction learned from one
nonconformity score per independent calibration group. The correction shifts
all pairwise capability half-spaces inward while preserving their exact affine
geometry and witness metadata.

The statistical statement is conditional on exchangeability of complete
calibration and target groups, a fixed pre-calibration model, quotient, task
domain, action set, loss family, and scoring rule. It is marginal over a future
group; it is not a pointwise safety guarantee and does not establish validity
under distribution shift.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, comb
from numbers import Integral, Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from .decision_capability_atlas_v1 import AffineCapabilityHalfspacesV1

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

DECISION_CAPABILITY_CALIBRATION_VERSION: Final = 1
DECISION_CAPABILITY_CALIBRATION_SEMANTICS: Final = (
    "finite-group-conformal-correction-for-affine-capability-halfspaces-v1"
)
DECISION_CAPABILITY_CALIBRATION_CLAIM_BOUNDARY: Final = (
    "The correction is finite-sample marginal only under exchangeability of "
    "complete calibration and target groups and a fixed pre-calibration model, "
    "quotient, task domain, action set, loss family, and score. It does not "
    "validate those ingredients, provide conditional or pointwise safety, cover "
    "distribution shift, or authorize deployment."
)
_NUMERICAL_ATOL: Final = 1e-10


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_int64(value: object) -> IntArray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    array.setflags(write=False)
    return array


def _finite_float_array(value: object, *, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != ndim or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return _immutable_float64(array)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    return result


def _validated_halfspaces(value: object) -> AffineCapabilityHalfspacesV1:
    if not isinstance(value, AffineCapabilityHalfspacesV1):
        raise TypeError("halfspaces must be AffineCapabilityHalfspacesV1")
    normal = np.asarray(value.normal)
    offset = np.asarray(value.offset)
    benchmark = np.asarray(value.benchmark_action_index)
    witness = np.asarray(value.witness_hypothesis_index)
    if normal.ndim != 2 or normal.shape[1] < 1:
        raise ValueError("halfspaces must have a positive task dimension")
    if offset.shape != (normal.shape[0],):
        raise ValueError("halfspaces contain inconsistent normal and offset arrays")
    if benchmark.shape != (normal.shape[0],):
        raise ValueError("halfspaces contain inconsistent benchmark metadata")
    if witness.ndim != 2 or witness.shape[0] != normal.shape[0]:
        raise ValueError("halfspaces contain inconsistent witness metadata")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(offset)):
        raise ValueError("halfspaces must be finite")
    if isinstance(value.action_index, bool) or not isinstance(
        value.action_index, Integral
    ):
        raise ValueError("halfspaces action_index must be an integer")
    tolerance = float(value.regret_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("halfspaces regret_tolerance must be nonnegative and finite")
    return value


class AffineEnvelopeMaximumV1(NamedTuple):
    """Maximum of a finite affine lower envelope over a task box."""

    maximum_value: float
    task_parameter: FloatArray
    active_affine_index: IntArray
    active_set_count: int


class PairwiseAtlasUndercoverageV1(NamedTuple):
    """Exact task-box undercoverage score for one proposed action."""

    action_index: int
    benchmark_action_index: IntArray
    raw_score_by_benchmark: FloatArray
    nonnegative_score: float
    critical_benchmark_action_index: int
    critical_task_parameter: FloatArray


class FiniteGroupAtlasCalibrationV1(NamedTuple):
    """Split-conformal correction from one score per calibration group."""

    alpha: float
    group_scores: FloatArray
    quantile_rank: int
    quantile_value: float
    correction: float
    guaranteed_marginal_coverage: float

    @property
    def calibration_group_count(self) -> int:
        return int(self.group_scores.size)

    def summary(self) -> dict[str, object]:
        return {
            "version": DECISION_CAPABILITY_CALIBRATION_VERSION,
            "semantics": DECISION_CAPABILITY_CALIBRATION_SEMANTICS,
            "alpha": self.alpha,
            "calibration_group_count": self.calibration_group_count,
            "quantile_rank": self.quantile_rank,
            "quantile_value": self.quantile_value,
            "correction": self.correction,
            "guaranteed_marginal_coverage": self.guaranteed_marginal_coverage,
            "claim_boundary": DECISION_CAPABILITY_CALIBRATION_CLAIM_BOUNDARY,
        }


def finite_group_atlas_calibration(
    group_scores: object,
    *,
    alpha: float,
    clip_below_zero: bool = True,
) -> FiniteGroupAtlasCalibrationV1:
    """Return the finite-sample order-statistic atlas correction.

    Supply exactly one fixed nonconformity score per independent calibration
    group. The function fails closed when ``alpha`` is too small for a finite
    order statistic with the available number of groups.
    """

    scores = _finite_float_array(group_scores, name="group_scores", ndim=1)
    if not isinstance(alpha, Real) or isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be a finite scalar strictly between zero and one")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be a finite scalar strictly between zero and one")
    if not isinstance(clip_below_zero, (bool, np.bool_)):
        raise ValueError("clip_below_zero must be boolean")

    group_count = int(scores.size)
    rank = int(ceil((group_count + 1) * (1.0 - alpha_value)))
    if rank > group_count:
        raise ValueError(
            "alpha is too small for a finite split-conformal correction with "
            f"{group_count} calibration groups"
        )
    quantile = float(np.sort(scores)[rank - 1])
    correction = max(0.0, quantile) if bool(clip_below_zero) else quantile
    return FiniteGroupAtlasCalibrationV1(
        alpha=alpha_value,
        group_scores=scores,
        quantile_rank=rank,
        quantile_value=quantile,
        correction=correction,
        guaranteed_marginal_coverage=rank / (group_count + 1),
    )


def statistically_corrected_halfspaces(
    halfspaces: object,
    correction: object,
) -> AffineCapabilityHalfspacesV1:
    """Shift an action region inward by one nonnegative statistical correction."""

    region = _validated_halfspaces(halfspaces)
    delta = _nonnegative_float(correction, name="correction")
    return AffineCapabilityHalfspacesV1(
        action_index=int(region.action_index),
        regret_tolerance=float(region.regret_tolerance),
        active_class_index=_immutable_int64(region.active_class_index),
        normal=_immutable_float64(region.normal),
        offset=_immutable_float64(np.asarray(region.offset) - delta),
        benchmark_action_index=_immutable_int64(region.benchmark_action_index),
        witness_hypothesis_index=_immutable_int64(region.witness_hypothesis_index),
    )


def maximize_affine_lower_envelope_on_box(
    affine_intercepts: object,
    affine_coefficients: object,
    task_bounds: object,
    *,
    maximum_active_sets: int = 100_000,
) -> AffineEnvelopeMaximumV1:
    """Exactly maximize ``min_r(intercept[r] + coefficient[r] @ theta)``.

    The task domain is an axis-aligned box. The routine enumerates vertices of
    the equivalent linear program in ``task_dimension + 1`` variables and fails
    closed before a caller-supplied active-set limit is exceeded. It is intended
    for low-dimensional registered task families.
    """

    intercept = _finite_float_array(
        affine_intercepts,
        name="affine_intercepts",
        ndim=1,
    )
    coefficient = _finite_float_array(
        affine_coefficients,
        name="affine_coefficients",
        ndim=2,
    )
    bounds = _finite_float_array(task_bounds, name="task_bounds", ndim=2)
    if coefficient.shape[0] != intercept.size:
        raise ValueError("affine coefficients and intercepts have inconsistent rows")
    dimension = int(coefficient.shape[1])
    if dimension < 1 or bounds.shape != (dimension, 2):
        raise ValueError("task_bounds must have shape (task_dimension, 2)")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each task bound must have lower < upper")
    active_set_limit = _positive_integer(
        maximum_active_sets,
        name="maximum_active_sets",
    )

    variable_count = dimension + 1
    constraint_count = int(intercept.size + 2 * dimension)
    active_set_count = comb(constraint_count, variable_count)
    if active_set_count > active_set_limit:
        raise ValueError(
            "exact affine-envelope maximization requires "
            f"{active_set_count} active sets, exceeding maximum_active_sets="
            f"{active_set_limit}"
        )

    matrix: FloatArray = np.zeros((constraint_count, variable_count), dtype=np.float64)
    rhs: FloatArray = np.empty(constraint_count, dtype=np.float64)
    matrix[: intercept.size, :dimension] = -coefficient
    matrix[: intercept.size, dimension] = 1.0
    rhs[: intercept.size] = intercept
    row = int(intercept.size)
    for axis in range(dimension):
        matrix[row, axis] = 1.0
        rhs[row] = bounds[axis, 1]
        row += 1
        matrix[row, axis] = -1.0
        rhs[row] = -bounds[axis, 0]
        row += 1

    best_value = -np.inf
    best_task: np.ndarray | None = None
    for selected in combinations(range(constraint_count), variable_count):
        selected_index = np.asarray(selected, dtype=np.int64)
        active_matrix = matrix[selected_index]
        if np.linalg.matrix_rank(active_matrix, tol=_NUMERICAL_ATOL) < variable_count:
            continue
        candidate = np.linalg.solve(active_matrix, rhs[selected_index])
        if np.all(matrix @ candidate <= rhs + _NUMERICAL_ATOL):
            value = float(candidate[-1])
            if value > best_value + _NUMERICAL_ATOL:
                best_value = value
                best_task = candidate[:dimension].copy()

    if best_task is None:
        raise RuntimeError("failed to locate a feasible affine-envelope optimum")
    affine_value = intercept + coefficient @ best_task
    active_affine = np.flatnonzero(
        np.abs(affine_value - best_value) <= 10.0 * _NUMERICAL_ATOL
    ).astype(np.int64, copy=False)
    if active_affine.size == 0:
        active_affine = np.array([int(np.argmin(affine_value))], dtype=np.int64)
    return AffineEnvelopeMaximumV1(
        maximum_value=best_value,
        task_parameter=_immutable_float64(best_task),
        active_affine_index=_immutable_int64(active_affine),
        active_set_count=active_set_count,
    )


def affine_box_pairwise_undercoverage_score(
    halfspaces: object,
    realized_gap_intercepts: object,
    realized_gap_coefficients: object,
    task_bounds: object,
    *,
    maximum_active_sets: int = 100_000,
) -> PairwiseAtlasUndercoverageV1:
    """Score real pairwise gaps against one action's model envelope exactly.

    ``realized_gap_intercepts[b] + realized_gap_coefficients[b] @ theta`` is the
    realized loss gap of the proposed action relative to benchmark action ``b``.
    The returned nonconformity score is the positive part of the largest gap by
    which reality exceeds the model envelope anywhere in the registered task
    box and against any represented benchmark action.
    """

    region = _validated_halfspaces(halfspaces)
    intercept = _finite_float_array(
        realized_gap_intercepts,
        name="realized_gap_intercepts",
        ndim=1,
    )
    coefficient = _finite_float_array(
        realized_gap_coefficients,
        name="realized_gap_coefficients",
        ndim=2,
    )
    if coefficient.shape != (intercept.size, region.task_dimension):
        raise ValueError(
            "realized_gap_coefficients must have shape (action_count, task_dimension)"
        )
    benchmark = np.asarray(region.benchmark_action_index, dtype=np.int64)
    unique_benchmark = np.unique(benchmark)
    if unique_benchmark.size < 1:
        raise ValueError("halfspaces must represent at least one benchmark action")
    if np.any(unique_benchmark < 0) or np.any(unique_benchmark >= intercept.size):
        raise ValueError("benchmark action index is outside the realized gap roster")
    if int(region.action_index) >= intercept.size:
        raise ValueError("proposed action index is outside the realized gap roster")

    raw_scores = np.empty(unique_benchmark.size, dtype=np.float64)
    witnesses = np.empty(
        (unique_benchmark.size, region.task_dimension),
        dtype=np.float64,
    )
    for position, benchmark_action in enumerate(unique_benchmark):
        rows = benchmark == benchmark_action
        model_intercept = (
            float(region.regret_tolerance) - np.asarray(region.offset)[rows]
        )
        difference_intercept = intercept[benchmark_action] - model_intercept
        difference_coefficient = (
            coefficient[benchmark_action][None, :] - np.asarray(region.normal)[rows]
        )
        optimum = maximize_affine_lower_envelope_on_box(
            difference_intercept,
            difference_coefficient,
            task_bounds,
            maximum_active_sets=maximum_active_sets,
        )
        raw_scores[position] = optimum.maximum_value
        witnesses[position] = optimum.task_parameter

    critical_position = int(np.argmax(raw_scores))
    score = max(0.0, float(raw_scores[critical_position]))
    return PairwiseAtlasUndercoverageV1(
        action_index=int(region.action_index),
        benchmark_action_index=_immutable_int64(unique_benchmark),
        raw_score_by_benchmark=_immutable_float64(raw_scores),
        nonnegative_score=score,
        critical_benchmark_action_index=int(unique_benchmark[critical_position]),
        critical_task_parameter=_immutable_float64(witnesses[critical_position]),
    )


__all__ = [
    "DECISION_CAPABILITY_CALIBRATION_CLAIM_BOUNDARY",
    "DECISION_CAPABILITY_CALIBRATION_SEMANTICS",
    "DECISION_CAPABILITY_CALIBRATION_VERSION",
    "AffineEnvelopeMaximumV1",
    "FiniteGroupAtlasCalibrationV1",
    "PairwiseAtlasUndercoverageV1",
    "affine_box_pairwise_undercoverage_score",
    "finite_group_atlas_calibration",
    "maximize_affine_lower_envelope_on_box",
    "statistically_corrected_halfspaces",
]
