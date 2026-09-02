"""Finite-sample calibration for exact affine decision-capability atlases.

The model-side atlas in :mod:`decision_capability_atlas_v1` is exact relative to
its registered finite physical support. This module adds a nonnegative split-
conformal correction computed from one scalar score per independent physical
object or trajectory. Subtracting that correction from every capability
half-space offset preserves the polyhedral atlas while accounting for observed
model undercoverage.

The statistical statement is marginal over a future exchangeable calibration
unit. It is not conditional validity, robustness to arbitrary distribution
shift, validation of the registered loss, a physical-safety certificate, or an
authorization for deployment.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import ceil, comb
from numbers import Integral, Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from .decision_capability_atlas_v1 import AffineCapabilityHalfspacesV1
from .decision_capability_task_uncertainty_v1 import (
    box_robust_center_halfspaces,
    ellipsoid_robust_center_halfspaces,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

STATISTICALLY_VALID_CAPABILITY_ATLAS_VERSION: Final = 1
STATISTICALLY_VALID_CAPABILITY_ATLAS_SEMANTICS: Final = (
    "object-level-split-conformal-correction-of-exact-affine-capability-atlas-v1"
)
STATISTICALLY_VALID_CAPABILITY_ATLAS_CLAIM_BOUNDARY: Final = (
    "Finite-sample marginal validity requires that complete calibration and "
    "target physical units are exchangeable and that the routed score, action "
    "set, task family, loss, and calibration protocol are fixed independently "
    "of target outcomes. The result is not conditional validity, arbitrary-shift "
    "robustness, validation of the physical support or task objective, a safety "
    "certificate, or authorization for deployment."
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


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


def _finite_vector(value: object, *, name: str, size: int | None = None) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if size is not None and array.size != size:
        raise ValueError(f"{name} has the wrong dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return _immutable_float64(array)


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _region(value: object) -> AffineCapabilityHalfspacesV1:
    if not isinstance(value, AffineCapabilityHalfspacesV1):
        raise TypeError("halfspaces must be AffineCapabilityHalfspacesV1")
    normal = np.asarray(value.normal)
    offset = np.asarray(value.offset)
    benchmark = np.asarray(value.benchmark_action_index)
    witness = np.asarray(value.witness_hypothesis_index)
    if (
        normal.ndim != 2
        or offset.ndim != 1
        or benchmark.ndim != 1
        or normal.shape[0] != offset.size
        or benchmark.size != offset.size
        or witness.ndim != 2
        or witness.shape[0] != offset.size
    ):
        raise ValueError("halfspaces contain inconsistent arrays")
    if normal.shape[1] < 1:
        raise ValueError("halfspaces must have a positive task dimension")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(offset)):
        raise ValueError("halfspaces must be finite")
    return value


class SplitConformalCapabilityCorrectionV1(NamedTuple):
    """One-sided split-conformal correction from independent unit scores."""

    miscoverage: float
    calibration_unit_count: int
    order_statistic_rank: int
    raw_quantile: float
    nonnegative_correction: float
    calibration_scores: FloatArray
    sorted_scores: FloatArray

    def summary(self) -> dict[str, object]:
        return {
            "version": STATISTICALLY_VALID_CAPABILITY_ATLAS_VERSION,
            "semantics": STATISTICALLY_VALID_CAPABILITY_ATLAS_SEMANTICS,
            "miscoverage": self.miscoverage,
            "calibration_unit_count": self.calibration_unit_count,
            "order_statistic_rank": self.order_statistic_rank,
            "raw_quantile": self.raw_quantile,
            "nonnegative_correction": self.nonnegative_correction,
            "claim_boundary": STATISTICALLY_VALID_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
        }


class ContinuousUndercoverageWitnessV1(NamedTuple):
    """Exact maximum realized-minus-model gap over a rectangular task domain."""

    score: float
    task_parameter: FloatArray
    realized_pairwise_gap: float
    model_pairwise_gap: float
    benchmark_action_index: int
    active_model_halfspace_index: int
    enumerated_candidate_vertex_count: int

    def summary(self) -> dict[str, object]:
        return {
            "version": STATISTICALLY_VALID_CAPABILITY_ATLAS_VERSION,
            "score": self.score,
            "task_parameter": self.task_parameter.tolist(),
            "realized_pairwise_gap": self.realized_pairwise_gap,
            "model_pairwise_gap": self.model_pairwise_gap,
            "benchmark_action_index": self.benchmark_action_index,
            "active_model_halfspace_index": self.active_model_halfspace_index,
            "enumerated_candidate_vertex_count": self.enumerated_candidate_vertex_count,
            "claim_boundary": STATISTICALLY_VALID_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
        }


class CalibrationUnitMaximumV1(NamedTuple):
    """Maximum routed undercoverage score for one complete physical unit."""

    score: float
    selected_case_index: int
    witness: ContinuousUndercoverageWitnessV1
    case_count: int


class CalibratedTaskAtlasV1(NamedTuple):
    """Task-point action mask after one common conformal correction."""

    task_parameters: FloatArray
    nonnegative_correction: float
    action_capability_mask: BoolArray

    @property
    def capability_mask(self) -> BoolArray:
        return _immutable_bool(np.any(self.action_capability_mask, axis=1))

    @property
    def unique_action_mask(self) -> BoolArray:
        return _immutable_bool(
            np.count_nonzero(self.action_capability_mask, axis=1) == 1
        )

    @property
    def overlap_mask(self) -> BoolArray:
        return _immutable_bool(
            np.count_nonzero(self.action_capability_mask, axis=1) > 1
        )

    def summary(self) -> dict[str, object]:
        return {
            "version": STATISTICALLY_VALID_CAPABILITY_ATLAS_VERSION,
            "semantics": STATISTICALLY_VALID_CAPABILITY_ATLAS_SEMANTICS,
            "task_count": int(self.task_parameters.shape[0]),
            "task_dimension": int(self.task_parameters.shape[1]),
            "action_count": int(self.action_capability_mask.shape[1]),
            "nonnegative_correction": self.nonnegative_correction,
            "capable_task_count": int(np.count_nonzero(self.capability_mask)),
            "unique_task_count": int(np.count_nonzero(self.unique_action_mask)),
            "overlap_task_count": int(np.count_nonzero(self.overlap_mask)),
            "claim_boundary": STATISTICALLY_VALID_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
        }


def split_conformal_capability_correction(
    calibration_unit_scores: object,
    *,
    miscoverage: float,
) -> SplitConformalCapabilityCorrectionV1:
    """Return the standard one-sided split-conformal order statistic.

    Every input value must already be one scalar maximum score for a complete
    exchangeability unit. Correlated windows, actions, benchmarks, and tasks
    must be aggregated inside that score rather than supplied as separate rows.
    The correction is clipped below at zero so target calibration never enlarges
    the model-side capability atlas.
    """

    scores = _finite_vector(calibration_unit_scores, name="calibration_unit_scores")
    alpha = _finite_scalar(miscoverage, name="miscoverage")
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    rank = int(ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        raise ValueError(
            "calibration sample is too small for the requested finite-sample "
            "miscoverage; the valid fail-closed correction is infinite"
        )
    sorted_scores = np.sort(scores)
    raw = float(sorted_scores[rank - 1])
    correction = max(raw, 0.0)
    return SplitConformalCapabilityCorrectionV1(
        miscoverage=alpha,
        calibration_unit_count=int(scores.size),
        order_statistic_rank=rank,
        raw_quantile=raw,
        nonnegative_correction=correction,
        calibration_scores=scores,
        sorted_scores=_immutable_float64(sorted_scores),
    )


def calibrated_capability_halfspaces(
    halfspaces: AffineCapabilityHalfspacesV1,
    nonnegative_correction: float,
) -> AffineCapabilityHalfspacesV1:
    """Shift every model-side capability constraint inward by one correction."""

    region = _region(halfspaces)
    correction = _finite_scalar(
        nonnegative_correction,
        name="nonnegative_correction",
    )
    if correction < 0.0:
        raise ValueError("nonnegative_correction must be nonnegative")
    return AffineCapabilityHalfspacesV1(
        action_index=region.action_index,
        regret_tolerance=region.regret_tolerance,
        active_class_index=_immutable_int64(region.active_class_index),
        normal=_immutable_float64(region.normal),
        offset=_immutable_float64(region.offset - correction),
        benchmark_action_index=_immutable_int64(region.benchmark_action_index),
        witness_hypothesis_index=_immutable_int64(region.witness_hypothesis_index),
    )


def calibrated_box_robust_center_halfspaces(
    halfspaces: AffineCapabilityHalfspacesV1,
    nonnegative_correction: float,
    half_widths: object,
) -> AffineCapabilityHalfspacesV1:
    """Combine data calibration and exact box objective uncertainty."""

    return box_robust_center_halfspaces(
        calibrated_capability_halfspaces(halfspaces, nonnegative_correction),
        half_widths,
    )


def calibrated_ellipsoid_robust_center_halfspaces(
    halfspaces: AffineCapabilityHalfspacesV1,
    nonnegative_correction: float,
    generator: object,
) -> AffineCapabilityHalfspacesV1:
    """Combine data calibration and exact ellipsoidal objective uncertainty."""

    return ellipsoid_robust_center_halfspaces(
        calibrated_capability_halfspaces(halfspaces, nonnegative_correction),
        generator,
    )


def calibrated_task_atlas(
    halfspaces_by_action: Sequence[AffineCapabilityHalfspacesV1],
    task_parameters: object,
    *,
    nonnegative_correction: float,
) -> CalibratedTaskAtlasV1:
    """Evaluate all calibrated action regions without resolving overlaps."""

    if isinstance(halfspaces_by_action, (str, bytes)) or not isinstance(
        halfspaces_by_action, Sequence
    ):
        raise TypeError("halfspaces_by_action must be a sequence")
    if not halfspaces_by_action:
        raise ValueError("halfspaces_by_action must be nonempty")
    regions = [_region(value) for value in halfspaces_by_action]
    actions = [region.action_index for region in regions]
    if actions != list(range(len(regions))):
        raise ValueError("halfspaces must be ordered by contiguous action index")
    dimension = regions[0].task_dimension
    if any(region.task_dimension != dimension for region in regions):
        raise ValueError("all action regions must have the same task dimension")
    raw = np.asarray(task_parameters)
    if raw.dtype.kind not in "iuf":
        raise ValueError("task_parameters must contain real numeric values")
    tasks = np.ascontiguousarray(raw, dtype=np.float64)
    if tasks.ndim != 2 or tasks.shape[0] < 1 or tasks.shape[1] != dimension:
        raise ValueError(
            "task_parameters must have shape (task_count, task_dimension)"
        )
    if not np.all(np.isfinite(tasks)):
        raise ValueError("task_parameters must be finite")
    correction = _finite_scalar(
        nonnegative_correction,
        name="nonnegative_correction",
    )
    if correction < 0.0:
        raise ValueError("nonnegative_correction must be nonnegative")
    mask = np.column_stack(
        [
            calibrated_capability_halfspaces(region, correction).contains(tasks)
            for region in regions
        ]
    )
    return CalibratedTaskAtlasV1(
        task_parameters=_immutable_float64(tasks),
        nonnegative_correction=correction,
        action_capability_mask=_immutable_bool(mask),
    )


def continuous_pairwise_undercoverage_score(
    halfspaces: AffineCapabilityHalfspacesV1,
    *,
    benchmark_action_index: int,
    realized_gap_intercept: float,
    realized_gap_coefficient: object,
    task_bounds: object,
    maximum_candidate_vertices: int = 200_000,
) -> ContinuousUndercoverageWitnessV1:
    """Maximize realized-minus-model pairwise gap over a task rectangle exactly.

    For one proposed action and benchmark, the model-side pairwise gap is the
    maximum of the witness-affine functions encoded by ``halfspaces``. The
    realized pairwise gap is supplied as one affine function. Their difference
    is concave piecewise affine. Its maximum over the registered rectangle is
    solved as a linear program by exhaustive vertex enumeration with a strict
    caller-controlled complexity cap. This routine targets low-dimensional
    auditable task families, not large generic linear programs.
    """

    region = _region(halfspaces)
    if isinstance(benchmark_action_index, (bool, np.bool_)) or not isinstance(
        benchmark_action_index, Integral
    ):
        raise ValueError("benchmark_action_index must be an integer")
    benchmark = int(benchmark_action_index)
    selected = np.flatnonzero(region.benchmark_action_index == benchmark)
    if selected.size == 0:
        raise ValueError("benchmark_action_index is absent from the halfspaces")
    coefficient = _finite_vector(
        realized_gap_coefficient,
        name="realized_gap_coefficient",
        size=region.task_dimension,
    )
    intercept = _finite_scalar(
        realized_gap_intercept,
        name="realized_gap_intercept",
    )
    raw_bounds = np.asarray(task_bounds)
    if raw_bounds.dtype.kind not in "iuf":
        raise ValueError("task_bounds must contain real numeric values")
    bounds = np.ascontiguousarray(raw_bounds, dtype=np.float64)
    if bounds.shape != (region.task_dimension, 2):
        raise ValueError("task_bounds must have shape (task_dimension, 2)")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("task_bounds must be finite with lower < upper")
    cap = _positive_integer(
        maximum_candidate_vertices,
        name="maximum_candidate_vertices",
    )

    normal = region.normal[selected]
    model_intercept = region.regret_tolerance - region.offset[selected]
    dimension = region.task_dimension
    variable_count = dimension + 1

    rows = [
        np.concatenate((model_normal - coefficient, np.array([1.0])))
        for model_normal in normal
    ]
    rhs = [
        intercept - float(candidate_intercept)
        for candidate_intercept in model_intercept
    ]
    for axis in range(dimension):
        upper = np.zeros(variable_count, dtype=np.float64)
        upper[axis] = 1.0
        rows.append(upper)
        rhs.append(float(bounds[axis, 1]))
        lower = np.zeros(variable_count, dtype=np.float64)
        lower[axis] = -1.0
        rows.append(lower)
        rhs.append(float(-bounds[axis, 0]))
    matrix = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64)
    candidate_system_count = comb(matrix.shape[0], variable_count)
    if candidate_system_count > cap:
        raise ValueError(
            "exact continuous score requires "
            f"{candidate_system_count} candidate vertices, exceeding "
            f"maximum_candidate_vertices={cap}"
        )

    best: FloatArray | None = None
    best_score = -np.inf
    feasible_count = 0
    for active in combinations(range(matrix.shape[0]), variable_count):
        active_matrix = matrix[np.asarray(active)]
        if np.linalg.matrix_rank(active_matrix) < variable_count:
            continue
        candidate = np.linalg.solve(active_matrix, vector[np.asarray(active)])
        if np.all(matrix @ candidate <= vector + _NUMERICAL_ATOL):
            feasible_count += 1
            score = float(candidate[-1])
            if score > best_score + _NUMERICAL_ATOL:
                best = candidate
                best_score = score
            elif abs(score - best_score) <= _NUMERICAL_ATOL and best is not None:
                if tuple(candidate[:-1]) < tuple(best[:-1]):
                    best = candidate
                    best_score = score
    if best is None:
        raise RuntimeError("exact vertex enumeration found no feasible LP vertex")

    theta = np.asarray(best[:-1], dtype=np.float64)
    witness_values = normal @ theta + model_intercept
    active_local = int(np.argmax(witness_values))
    model_gap = float(witness_values[active_local])
    realized_gap = float(intercept + coefficient @ theta)
    residual = realized_gap - model_gap
    if not np.isclose(residual, best_score, atol=1e-8, rtol=1e-8):
        raise RuntimeError("continuous score LP failed its residual identity")
    return ContinuousUndercoverageWitnessV1(
        score=residual,
        task_parameter=_immutable_float64(theta),
        realized_pairwise_gap=realized_gap,
        model_pairwise_gap=model_gap,
        benchmark_action_index=benchmark,
        active_model_halfspace_index=int(selected[active_local]),
        enumerated_candidate_vertex_count=feasible_count,
    )


def calibration_unit_maximum(
    case_witnesses: Sequence[ContinuousUndercoverageWitnessV1],
) -> CalibrationUnitMaximumV1:
    """Aggregate all routed cases into one score for one exchangeability unit."""

    if isinstance(case_witnesses, (str, bytes)) or not isinstance(
        case_witnesses, Sequence
    ):
        raise TypeError("case_witnesses must be a sequence")
    if not case_witnesses:
        raise ValueError("case_witnesses must be nonempty")
    if any(
        not isinstance(value, ContinuousUndercoverageWitnessV1)
        for value in case_witnesses
    ):
        raise TypeError("every case witness must be ContinuousUndercoverageWitnessV1")
    scores = np.asarray([value.score for value in case_witnesses], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("case witness scores must be finite")
    selected = int(np.argmax(scores))
    return CalibrationUnitMaximumV1(
        score=float(scores[selected]),
        selected_case_index=selected,
        witness=case_witnesses[selected],
        case_count=len(case_witnesses),
    )


__all__ = [
    "STATISTICALLY_VALID_CAPABILITY_ATLAS_CLAIM_BOUNDARY",
    "STATISTICALLY_VALID_CAPABILITY_ATLAS_SEMANTICS",
    "STATISTICALLY_VALID_CAPABILITY_ATLAS_VERSION",
    "CalibratedTaskAtlasV1",
    "CalibrationUnitMaximumV1",
    "ContinuousUndercoverageWitnessV1",
    "SplitConformalCapabilityCorrectionV1",
    "calibrated_box_robust_center_halfspaces",
    "calibrated_capability_halfspaces",
    "calibrated_ellipsoid_robust_center_halfspaces",
    "calibrated_task_atlas",
    "calibration_unit_maximum",
    "continuous_pairwise_undercoverage_score",
    "split_conformal_capability_correction",
]
