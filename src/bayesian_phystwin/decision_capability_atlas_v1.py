"""Exact decision-capability atlases for affine registered task families.

A query quotient may leave the complete physical state ambiguous.  The finite
certificate in :mod:`bayesian_phystwin.query_decision_certificate_v1` determines
whether one registered action is nevertheless optimal, or uniformly within a
registered regret tolerance, for every compatible complete belief.

This module evaluates that same exact certificate over a family of tasks whose
hypothesis-wise action losses are affine in a task parameter ``theta``:

    L_theta(i, a) = beta[i, a] + phi[i, a] @ theta.

The resulting atlas reports which tasks the incomplete physical twin is
qualified to decide.  For small finite supports it can also enumerate an exact
half-space representation of one action's continuous capability region.  The
construction does not validate the quotient, the task family, the loss, or the
regret tolerance, and it does not establish real-world or deployment validity.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from numbers import Integral
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from .query_decision_certificate_v1 import query_decision_certificate

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

DECISION_CAPABILITY_ATLAS_VERSION: Final = 1
DECISION_CAPABILITY_ATLAS_SEMANTICS: Final = (
    "exact-affine-task-family-regret-over-registered-query-quotient-v1"
)
DECISION_CAPABILITY_ATLAS_CLAIM_BOUNDARY: Final = (
    "The atlas is exact only for the supplied finite hypotheses, positive prior "
    "support, registered quotient masses, affine task-loss family, action set, "
    "and regret tolerance. It does not validate the quotient or task family, "
    "identify a physical state, justify the loss or tolerance, establish "
    "out-of-support validity, calibrate uncertainty, certify safety, or authorize "
    "deployment."
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


class DecisionCapabilityAtlasV1(NamedTuple):
    """Exact finite-task atlas for one affine loss family."""

    prior_weights: FloatArray
    prior_support_mask: BoolArray
    quotient_weights: FloatArray
    class_index: IntArray
    loss_intercepts: FloatArray
    loss_coefficients: FloatArray
    task_parameters: FloatArray
    pairwise_worst_case_loss_gap: FloatArray
    worst_case_regret: FloatArray
    minimax_action_index: IntArray
    minimax_worst_case_regret: FloatArray
    regret_tolerance: float
    tolerance_admissible_action_mask: BoolArray
    robustly_optimal_action_mask: BoolArray

    @property
    def task_count(self) -> int:
        return int(self.task_parameters.shape[0])

    @property
    def task_dimension(self) -> int:
        return int(self.task_parameters.shape[1])

    @property
    def hypothesis_count(self) -> int:
        return int(self.class_index.size)

    @property
    def quotient_class_count(self) -> int:
        return int(self.quotient_weights.size)

    @property
    def action_count(self) -> int:
        return int(self.worst_case_regret.shape[1])

    @property
    def capability_mask(self) -> BoolArray:
        return _immutable_bool(np.any(self.tolerance_admissible_action_mask, axis=1))

    @property
    def exact_capability_mask(self) -> BoolArray:
        return _immutable_bool(np.any(self.robustly_optimal_action_mask, axis=1))

    @property
    def uniquely_tolerance_identified_mask(self) -> BoolArray:
        return _immutable_bool(
            np.count_nonzero(self.tolerance_admissible_action_mask, axis=1) == 1
        )

    @property
    def uniquely_exactly_identified_mask(self) -> BoolArray:
        return _immutable_bool(
            np.count_nonzero(self.robustly_optimal_action_mask, axis=1) == 1
        )

    def policy_action_index(self, fallback_action_index: int) -> IntArray:
        """Return minimax actions where capable and fallback elsewhere."""

        if isinstance(fallback_action_index, bool) or not isinstance(
            fallback_action_index, Integral
        ):
            raise ValueError("fallback_action_index must be an integer")
        fallback = int(fallback_action_index)
        if not 0 <= fallback < self.action_count:
            raise ValueError("fallback_action_index is outside the action set")
        result = np.full(self.task_count, fallback, dtype=np.int64)
        capable = self.capability_mask
        result[capable] = self.minimax_action_index[capable]
        return _immutable_int64(result)

    def summary(self) -> dict[str, object]:
        return {
            "version": DECISION_CAPABILITY_ATLAS_VERSION,
            "semantics": DECISION_CAPABILITY_ATLAS_SEMANTICS,
            "task_count": self.task_count,
            "task_dimension": self.task_dimension,
            "hypothesis_count": self.hypothesis_count,
            "quotient_class_count": self.quotient_class_count,
            "action_count": self.action_count,
            "regret_tolerance": self.regret_tolerance,
            "capable_task_count": int(np.count_nonzero(self.capability_mask)),
            "exact_task_count": int(np.count_nonzero(self.exact_capability_mask)),
            "uniquely_tolerance_identified_task_count": int(
                np.count_nonzero(self.uniquely_tolerance_identified_mask)
            ),
            "uniquely_exactly_identified_task_count": int(
                np.count_nonzero(self.uniquely_exactly_identified_mask)
            ),
            "claim_boundary": DECISION_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
        }


class AffineCapabilityHalfspacesV1(NamedTuple):
    """Exact task-space half-spaces for one action's capability region."""

    action_index: int
    regret_tolerance: float
    active_class_index: IntArray
    normal: FloatArray
    offset: FloatArray
    benchmark_action_index: IntArray
    witness_hypothesis_index: IntArray

    @property
    def halfspace_count(self) -> int:
        return int(self.offset.size)

    @property
    def task_dimension(self) -> int:
        return int(self.normal.shape[1])

    def contains(self, task_parameters: object) -> BoolArray:
        tasks = _finite_float_array(
            task_parameters,
            name="task_parameters",
            ndim=2,
        )
        if tasks.shape[1] != self.task_dimension:
            raise ValueError("task_parameters has the wrong task dimension")
        if self.halfspace_count == 0:
            return _immutable_bool(np.ones(tasks.shape[0], dtype=np.bool_))
        lhs = tasks @ self.normal.T
        return _immutable_bool(
            np.all(lhs <= self.offset[None, :] + _NUMERICAL_ATOL, axis=1)
        )


class _ValidatedFamily(NamedTuple):
    prior_weights: FloatArray
    prior_support_mask: BoolArray
    quotient_weights: FloatArray
    class_index: IntArray
    loss_intercepts: FloatArray
    loss_coefficients: FloatArray
    task_parameters: FloatArray
    regret_tolerance: float


def _validated_family(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_intercepts: object,
    loss_coefficients: object,
    task_parameters: object,
    *,
    regret_tolerance: float,
) -> _ValidatedFamily:
    intercepts = _finite_float_array(
        loss_intercepts,
        name="loss_intercepts",
        ndim=2,
    )
    coefficients = _finite_float_array(
        loss_coefficients,
        name="loss_coefficients",
        ndim=3,
    )
    tasks = _finite_float_array(
        task_parameters,
        name="task_parameters",
        ndim=2,
    )
    if coefficients.shape[:2] != intercepts.shape:
        raise ValueError(
            "loss_coefficients must have shape "
            "(hypothesis_count, action_count, task_dimension)"
        )
    if coefficients.shape[2] != tasks.shape[1]:
        raise ValueError("task_parameters has the wrong task dimension")

    first_loss = intercepts + np.tensordot(
        coefficients,
        tasks[0],
        axes=(2, 0),
    )
    certificate = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        first_loss,
        regret_tolerance=regret_tolerance,
    )
    if certificate.hypothesis_count != intercepts.shape[0]:
        raise ValueError("loss family has the wrong hypothesis count")
    if certificate.action_count != intercepts.shape[1]:
        raise ValueError("loss family must contain at least two actions")
    return _ValidatedFamily(
        prior_weights=certificate.prior_weights,
        prior_support_mask=certificate.prior_support_mask,
        quotient_weights=certificate.quotient_weights,
        class_index=certificate.class_index,
        loss_intercepts=intercepts,
        loss_coefficients=coefficients,
        task_parameters=tasks,
        regret_tolerance=certificate.regret_tolerance,
    )


def affine_decision_capability_atlas(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_intercepts: object,
    loss_coefficients: object,
    task_parameters: object,
    *,
    regret_tolerance: float = 0.0,
    task_batch_size: int = 256,
) -> DecisionCapabilityAtlasV1:
    """Evaluate the exact quotient regret over an affine task family.

    ``loss_intercepts[i, a] + loss_coefficients[i, a] @ theta`` is the
    registered loss of action ``a`` under hypothesis ``i`` for task ``theta``.
    Every supplied task point is evaluated exactly; no interpolation or learned
    classifier is used.
    """

    family = _validated_family(
        prior_weights,
        quotient_weights,
        class_index,
        loss_intercepts,
        loss_coefficients,
        task_parameters,
        regret_tolerance=regret_tolerance,
    )
    batch_size = _positive_integer(task_batch_size, name="task_batch_size")
    hypotheses, actions = family.loss_intercepts.shape
    task_count = family.task_parameters.shape[0]
    pairwise_intercept = (
        family.loss_intercepts[:, :, None] - family.loss_intercepts[:, None, :]
    )
    pairwise_coefficient = (
        family.loss_coefficients[:, :, None, :]
        - family.loss_coefficients[:, None, :, :]
    )
    pairwise = np.empty((task_count, actions, actions), dtype=np.float64)
    for start in range(0, task_count, batch_size):
        stop = min(start + batch_size, task_count)
        task_batch = family.task_parameters[start:stop]
        differences = pairwise_intercept[None, ...] + np.tensordot(
            task_batch,
            pairwise_coefficient,
            axes=(1, 3),
        )
        # tensordot returns (task, hypothesis, action, benchmark).
        batch_pairwise = np.zeros((stop - start, actions, actions), dtype=np.float64)
        for class_id, class_mass in enumerate(family.quotient_weights):
            if class_mass <= 0.0:
                continue
            members = (family.class_index == class_id) & family.prior_support_mask
            if not np.any(members):  # guarded by query_decision_certificate
                raise RuntimeError("posterior-supported class lost prior support")
            batch_pairwise += float(class_mass) * np.max(
                differences[:, members, :, :],
                axis=1,
            )
        diagonal = np.arange(actions)
        batch_pairwise[:, diagonal, diagonal] = 0.0
        pairwise[start:stop] = batch_pairwise

    regret = np.maximum(np.max(pairwise, axis=2), 0.0)
    minimax_action = np.argmin(regret, axis=1).astype(np.int64, copy=False)
    minimax_regret = regret[np.arange(task_count), minimax_action]
    tolerance_mask = regret <= family.regret_tolerance + _NUMERICAL_ATOL
    robust_mask = np.all(pairwise <= _NUMERICAL_ATOL, axis=2)
    if pairwise.shape != (task_count, actions, actions) or hypotheses < 1:
        raise RuntimeError("internal atlas shape error")

    return DecisionCapabilityAtlasV1(
        prior_weights=family.prior_weights,
        prior_support_mask=family.prior_support_mask,
        quotient_weights=family.quotient_weights,
        class_index=family.class_index,
        loss_intercepts=family.loss_intercepts,
        loss_coefficients=family.loss_coefficients,
        task_parameters=family.task_parameters,
        pairwise_worst_case_loss_gap=_immutable_float64(pairwise),
        worst_case_regret=_immutable_float64(regret),
        minimax_action_index=_immutable_int64(minimax_action),
        minimax_worst_case_regret=_immutable_float64(minimax_regret),
        regret_tolerance=family.regret_tolerance,
        tolerance_admissible_action_mask=_immutable_bool(tolerance_mask),
        robustly_optimal_action_mask=_immutable_bool(robust_mask),
    )


def affine_capability_halfspaces(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_intercepts: object,
    loss_coefficients: object,
    *,
    action_index: int,
    regret_tolerance: float = 0.0,
    maximum_halfspaces: int = 100_000,
) -> AffineCapabilityHalfspacesV1:
    """Enumerate the exact task-space half-spaces for one action.

    The continuous capability region is the set of task parameters for which
    the action's exact worst-case regret does not exceed ``regret_tolerance``.
    The enumeration is exponential only in the number of posterior-supported
    quotient classes.  It fails closed before materializing more than
    ``maximum_halfspaces`` constraints.
    """

    intercepts = _finite_float_array(
        loss_intercepts,
        name="loss_intercepts",
        ndim=2,
    )
    coefficients = _finite_float_array(
        loss_coefficients,
        name="loss_coefficients",
        ndim=3,
    )
    if coefficients.shape[:2] != intercepts.shape or coefficients.shape[2] < 1:
        raise ValueError(
            "loss_coefficients must have shape "
            "(hypothesis_count, action_count, positive_task_dimension)"
        )
    if isinstance(action_index, bool) or not isinstance(action_index, Integral):
        raise ValueError("action_index must be an integer")
    action = int(action_index)
    if not 0 <= action < intercepts.shape[1]:
        raise ValueError("action_index is outside the action set")
    halfspace_limit = _positive_integer(
        maximum_halfspaces,
        name="maximum_halfspaces",
    )

    zero_task = np.zeros((1, coefficients.shape[2]), dtype=np.float64)
    family = _validated_family(
        prior_weights,
        quotient_weights,
        class_index,
        intercepts,
        coefficients,
        zero_task,
        regret_tolerance=regret_tolerance,
    )
    active_classes = np.flatnonzero(family.quotient_weights > 0.0).astype(
        np.int64,
        copy=False,
    )
    member_rosters = [
        np.flatnonzero(
            (family.class_index == class_id) & family.prior_support_mask
        ).astype(np.int64, copy=False)
        for class_id in active_classes
    ]
    combinations_per_benchmark = _integer_product(
        int(roster.size) for roster in member_rosters
    )
    required = combinations_per_benchmark * (intercepts.shape[1] - 1)
    if required > halfspace_limit:
        raise ValueError(
            "exact capability region requires "
            f"{required} half-spaces, exceeding maximum_halfspaces="
            f"{halfspace_limit}"
        )

    normal: list[FloatArray] = []
    offset: list[float] = []
    benchmark: list[int] = []
    witness: list[tuple[int, ...]] = []
    for benchmark_action in range(intercepts.shape[1]):
        if benchmark_action == action:
            continue
        for selected in product(*member_rosters):
            selected_indices = np.asarray(selected, dtype=np.int64)
            masses = family.quotient_weights[active_classes]
            intercept_difference = (
                family.loss_intercepts[selected_indices, action]
                - family.loss_intercepts[selected_indices, benchmark_action]
            )
            coefficient_difference = (
                family.loss_coefficients[selected_indices, action, :]
                - family.loss_coefficients[selected_indices, benchmark_action, :]
            )
            normal.append(np.einsum("c,cd->d", masses, coefficient_difference))
            offset.append(
                family.regret_tolerance - float(np.dot(masses, intercept_difference))
            )
            benchmark.append(benchmark_action)
            witness.append(tuple(int(value) for value in selected))

    normal_array = np.asarray(normal, dtype=np.float64).reshape(
        -1,
        coefficients.shape[2],
    )
    witness_array = np.asarray(witness, dtype=np.int64).reshape(
        -1,
        active_classes.size,
    )
    return AffineCapabilityHalfspacesV1(
        action_index=action,
        regret_tolerance=family.regret_tolerance,
        active_class_index=_immutable_int64(active_classes),
        normal=_immutable_float64(normal_array),
        offset=_immutable_float64(offset),
        benchmark_action_index=_immutable_int64(benchmark),
        witness_hypothesis_index=_immutable_int64(witness_array),
    )


def _integer_product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def capability_polygon_2d(
    halfspaces: AffineCapabilityHalfspacesV1,
    task_bounds: object,
) -> FloatArray:
    """Clip a rectangular two-dimensional task domain by exact half-spaces."""

    if halfspaces.task_dimension != 2:
        raise ValueError("capability_polygon_2d requires a two-dimensional task")
    bounds = _finite_float_array(task_bounds, name="task_bounds", ndim=2)
    if bounds.shape != (2, 2):
        raise ValueError("task_bounds must have shape (2, 2)")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each task bound must have lower < upper")
    polygon = [
        np.array([bounds[0, 0], bounds[1, 0]], dtype=np.float64),
        np.array([bounds[0, 1], bounds[1, 0]], dtype=np.float64),
        np.array([bounds[0, 1], bounds[1, 1]], dtype=np.float64),
        np.array([bounds[0, 0], bounds[1, 1]], dtype=np.float64),
    ]
    for normal, offset in zip(halfspaces.normal, halfspaces.offset, strict=True):
        if not polygon:
            break
        clipped: list[FloatArray] = []
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            start_value = float(np.dot(normal, start) - offset)
            end_value = float(np.dot(normal, end) - offset)
            start_inside = start_value <= _NUMERICAL_ATOL
            end_inside = end_value <= _NUMERICAL_ATOL
            if start_inside:
                clipped.append(start)
            if start_inside != end_inside:
                denominator = start_value - end_value
                if abs(denominator) <= _NUMERICAL_ATOL:
                    continue
                fraction = start_value / denominator
                clipped.append(start + fraction * (end - start))
        polygon = clipped
    if not polygon:
        return _immutable_float64(np.empty((0, 2), dtype=np.float64))
    result = np.asarray(polygon, dtype=np.float64)
    keep = np.ones(result.shape[0], dtype=np.bool_)
    for index in range(1, result.shape[0]):
        keep[index] = not np.allclose(
            result[index],
            result[index - 1],
            atol=_NUMERICAL_ATOL,
            rtol=0.0,
        )
    result = result[keep]
    if result.shape[0] > 1 and np.allclose(
        result[0],
        result[-1],
        atol=_NUMERICAL_ATOL,
        rtol=0.0,
    ):
        result = result[:-1]
    return _immutable_float64(result)


def polygon_area_2d(vertices: object) -> float:
    """Return the nonnegative area of a two-dimensional polygon."""

    polygon = np.asarray(vertices, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("vertices must have shape (vertex_count, 2)")
    if polygon.shape[0] < 3:
        return 0.0
    return 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


__all__ = [
    "DECISION_CAPABILITY_ATLAS_CLAIM_BOUNDARY",
    "DECISION_CAPABILITY_ATLAS_SEMANTICS",
    "DECISION_CAPABILITY_ATLAS_VERSION",
    "AffineCapabilityHalfspacesV1",
    "DecisionCapabilityAtlasV1",
    "affine_capability_halfspaces",
    "affine_decision_capability_atlas",
    "capability_polygon_2d",
    "polygon_area_2d",
]
