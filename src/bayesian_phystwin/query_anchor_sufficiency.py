"""Support--precision planning curves for independent physical anchors.

The curve is a source/calibration planning diagnostic. Its variances are
model-based expectations under a frozen linearization and nuisance prior; they
are not empirical accuracy, coverage, or safety evidence.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .nuisance_aware_information import NuisanceAwareInformationState
from .query_aware_anchor_planning import greedy_query_aware_selection

DEFAULT_ANCHOR_PRECISION_MULTIPLIERS: tuple[float, ...] = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
)
_TOLERANCE = 1e-10


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _immutable(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("anchor-sufficiency arrays must not contain objects")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    _require(result >= 0, f"{name} must be a nonnegative integer")
    return result


def _finite_scalar(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    _require(
        raw.shape == () and raw.dtype.kind in "iuf",
        f"{name} must be a finite real number",
    )
    result = float(raw.item())
    _require(np.isfinite(result), f"{name} must be finite")
    if strictly_positive:
        _require(result > minimum, f"{name} must be positive")
    else:
        _require(result >= minimum, f"{name} must be at least {minimum}")
    if maximum is not None:
        _require(result <= maximum, f"{name} must be at most {maximum}")
    return result


def _positive_precision_vector(value: object) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.ndim == 1
        and raw.size > 0
        and raw.dtype.kind in "iuf"
        and raw.dtype.kind != "b",
        "precision_multipliers must be a nonempty numeric vector",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        "precision_multipliers must be positive and finite",
    )
    _require(
        np.all(np.diff(result) > 0.0),
        "precision_multipliers must be strictly increasing",
    )
    return _immutable(result, np.dtype(np.float64))


def _integer_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == ndim and raw.size == 0 and raw.dtype.kind == "f":
        return _immutable(raw, np.dtype(np.int64))
    integer = np.issubdtype(raw.dtype, np.integer) and not np.issubdtype(
        raw.dtype,
        np.bool_,
    )
    _require(raw.ndim == ndim and integer, f"{name} must be an integer array")
    if np.issubdtype(raw.dtype, np.unsignedinteger):
        _require(
            not np.any(raw > np.iinfo(np.int64).max),
            f"{name} contains integers outside int64 range",
        )
    return _immutable(raw, np.dtype(np.int64))


def _finite_matrix(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2, f"{name} must be a matrix")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


@dataclass(frozen=True, slots=True)
class QueryAnchorSufficiencyCurveV1:
    """Immutable nested support curve for several precision multipliers."""

    precision_multipliers: np.ndarray
    support_counts: np.ndarray
    selected_indices: np.ndarray
    selected_counts: np.ndarray
    query_variance_traces: np.ndarray
    cumulative_costs: np.ndarray
    target_remaining_variance_fraction: float
    first_sufficient_support: np.ndarray

    def __post_init__(self) -> None:
        precision = _positive_precision_vector(self.precision_multipliers)
        support = _integer_array(self.support_counts, name="support_counts", ndim=1)
        indices = _integer_array(
            self.selected_indices,
            name="selected_indices",
            ndim=2,
        )
        selected_counts = _integer_array(
            self.selected_counts,
            name="selected_counts",
            ndim=1,
        )
        first_support = _integer_array(
            self.first_sufficient_support,
            name="first_sufficient_support",
            ndim=1,
        )
        traces = _finite_matrix(
            self.query_variance_traces,
            name="query_variance_traces",
        )
        costs = _finite_matrix(self.cumulative_costs, name="cumulative_costs")
        target = _finite_scalar(
            self.target_remaining_variance_fraction,
            name="target_remaining_variance_fraction",
            minimum=0.0,
            maximum=1.0,
            strictly_positive=True,
        )

        _require(
            len(support) >= 1 and support[0] == 0,
            "support_counts must start at zero",
        )
        maximum_count = len(support) - 1
        precision_count = len(precision)
        _require(
            np.array_equal(support, np.arange(maximum_count + 1)),
            "support_counts must be the consecutive range from zero",
        )
        _require(
            indices.shape == (precision_count, maximum_count),
            "selected_indices shape changed",
        )
        _require(
            selected_counts.shape == first_support.shape == (precision_count,),
            "selected-count diagnostic shapes changed",
        )
        _require(
            traces.shape == costs.shape == (precision_count, maximum_count + 1),
            "trace or cost matrix shape changed",
        )
        _require(
            np.all((selected_counts >= 0) & (selected_counts <= maximum_count)),
            "selected_counts lie outside the support range",
        )
        _require(
            np.all(traces >= 0.0) and np.all(costs >= 0.0),
            "traces and cumulative costs must be nonnegative",
        )
        _require(
            np.allclose(costs[:, 0], 0.0, rtol=0.0, atol=_TOLERANCE),
            "support-zero cumulative cost must be zero",
        )
        initial = float(traces[0, 0])
        _require(initial > 0.0, "initial query variance trace must be positive")
        _require(
            np.allclose(
                traces[:, 0],
                initial,
                rtol=_TOLERANCE,
                atol=_TOLERANCE * max(1.0, initial),
            ),
            "initial query variance trace changed across precision multipliers",
        )

        for row in range(precision_count):
            selected_count = int(selected_counts[row])
            prefix = indices[row, :selected_count]
            _require(
                np.all(prefix >= 0) and len(np.unique(prefix)) == len(prefix),
                "selected index prefixes must be nonnegative and unique",
            )
            _require(
                np.all(indices[row, selected_count:] == -1),
                "unused selected-index suffixes must be -1",
            )
            tolerance = _TOLERANCE * max(1.0, initial, float(np.max(traces[row])))
            _require(
                np.all(np.diff(traces[row]) <= tolerance),
                "query variance must not increase with selected support",
            )
            _require(
                np.all(np.diff(costs[row]) >= -_TOLERANCE),
                "cumulative cost must not decrease with selected support",
            )
            if selected_count < maximum_count:
                _require(
                    np.allclose(
                        traces[row, selected_count + 1 :],
                        traces[row, selected_count],
                        rtol=_TOLERANCE,
                        atol=tolerance,
                    )
                    and np.allclose(
                        costs[row, selected_count + 1 :],
                        costs[row, selected_count],
                        rtol=_TOLERANCE,
                        atol=_TOLERANCE,
                    ),
                    "trace and cost must remain fixed after selection stops",
                )
            crossings = np.flatnonzero(traces[row] / initial <= target + _TOLERANCE)
            expected = int(crossings[0]) if len(crossings) else -1
            _require(
                int(first_support[row]) == expected,
                "first_sufficient_support does not match the target crossing",
            )
            if expected >= 0:
                _require(
                    expected <= selected_count,
                    "sufficiency cannot occur after selection stopped",
                )

        object.__setattr__(self, "precision_multipliers", precision)
        object.__setattr__(self, "support_counts", support)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "selected_counts", selected_counts)
        object.__setattr__(
            self,
            "query_variance_traces",
            _immutable(traces, np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "cumulative_costs",
            _immutable(costs, np.dtype(np.float64)),
        )
        object.__setattr__(self, "target_remaining_variance_fraction", target)
        object.__setattr__(self, "first_sufficient_support", first_support)

    @property
    def initial_query_variance_trace(self) -> float:
        return float(self.query_variance_traces[0, 0])

    @property
    def remaining_variance_fractions(self) -> np.ndarray:
        return _immutable(
            self.query_variance_traces / self.initial_query_variance_trace,
            np.dtype(np.float64),
        )

    @property
    def sufficient_precision_mask(self) -> np.ndarray:
        return _immutable(self.first_sufficient_support >= 0, np.dtype(np.bool_))

    def selected_prefix(self, precision_index: int, support_count: int) -> np.ndarray:
        precision = _integer(precision_index, name="precision_index")
        support = _integer(support_count, name="support_count")
        _require(precision < len(self.precision_multipliers), "precision index invalid")
        _require(support < len(self.support_counts), "support count invalid")
        retained = min(support, int(self.selected_counts[precision]))
        return _immutable(
            self.selected_indices[precision, :retained],
            np.dtype(np.int64),
        )

    def records(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        fractions = self.remaining_variance_fractions
        for precision_index, multiplier in enumerate(self.precision_multipliers):
            for support_value in self.support_counts:
                support = int(support_value)
                prefix = self.selected_prefix(precision_index, support)
                rows.append(
                    {
                        "precision_multiplier": float(multiplier),
                        "support_count": support,
                        "selected_count": len(prefix),
                        "selected_indices": prefix.tolist(),
                        "query_variance_trace": float(
                            self.query_variance_traces[precision_index, support]
                        ),
                        "remaining_variance_fraction": float(
                            fractions[precision_index, support]
                        ),
                        "cumulative_cost": float(
                            self.cumulative_costs[precision_index, support]
                        ),
                        "target_met": bool(
                            fractions[precision_index, support]
                            <= self.target_remaining_variance_fraction + _TOLERANCE
                        ),
                    }
                )
        return tuple(rows)


def evaluate_query_anchor_sufficiency(
    prior: NuisanceAwareInformationState,
    query_jacobian: np.ndarray,
    state_jacobians: Sequence[np.ndarray],
    nuisance_jacobians: Sequence[np.ndarray | None],
    observation_covariances: Sequence[np.ndarray],
    *,
    precision_multipliers: Sequence[float] = DEFAULT_ANCHOR_PRECISION_MULTIPLIERS,
    reliabilities: Sequence[float | np.ndarray] | None = None,
    costs: Sequence[float] | None = None,
    dependence_groups: Sequence[Hashable | None] | None = None,
    maximum_count: int | None = None,
    minimum_trace_reduction: float = 0.0,
    target_remaining_variance_fraction: float = 0.25,
) -> QueryAnchorSufficiencyCurveV1:
    """Evaluate nested anchor support for several information multipliers.

    A multiplier divides every observation covariance. Candidate identities,
    Jacobians, reliabilities, costs, dependence groups, and ordering remain
    fixed. Each smaller support is an exact prefix of the maximum-support plan.
    """

    if not isinstance(prior, NuisanceAwareInformationState):
        raise TypeError("prior must be a NuisanceAwareInformationState")
    candidate_count = len(state_jacobians)
    _require(
        len(nuisance_jacobians) == candidate_count
        and len(observation_covariances) == candidate_count,
        "candidate input counts differ",
    )
    support_limit = (
        candidate_count
        if maximum_count is None
        else _integer(maximum_count, name="maximum_count")
    )
    _require(
        support_limit <= candidate_count,
        "maximum_count cannot exceed the candidate count",
    )
    precision = _positive_precision_vector(precision_multipliers)
    target = _finite_scalar(
        target_remaining_variance_fraction,
        name="target_remaining_variance_fraction",
        minimum=0.0,
        maximum=1.0,
        strictly_positive=True,
    )
    minimum_reduction = _finite_scalar(
        minimum_trace_reduction,
        name="minimum_trace_reduction",
        minimum=0.0,
    )

    precision_count = len(precision)
    selected_indices: np.ndarray = np.full(
        (precision_count, support_limit),
        -1,
        dtype=np.int64,
    )
    selected_counts: np.ndarray = np.zeros(precision_count, dtype=np.int64)
    traces: np.ndarray = np.empty(
        (precision_count, support_limit + 1),
        dtype=np.float64,
    )
    cumulative_costs: np.ndarray = np.empty_like(traces)
    first_support: np.ndarray = np.full(
        precision_count,
        -1,
        dtype=np.int64,
    )

    for row, multiplier in enumerate(precision):
        scaled_covariances: list[np.ndarray] = []
        for covariance in observation_covariances:
            scaled = np.asarray(covariance, dtype=np.float64) / float(multiplier)
            _require(
                np.all(np.isfinite(scaled)),
                "scaled observation covariance contains non-finite values",
            )
            scaled_covariances.append(scaled)
        selection = greedy_query_aware_selection(
            prior,
            query_jacobian,
            state_jacobians,
            nuisance_jacobians,
            scaled_covariances,
            reliabilities=reliabilities,
            costs=costs,
            dependence_groups=dependence_groups,
            count=support_limit,
            minimum_trace_reduction=minimum_reduction,
        )
        selected = _integer_array(
            selection.selected_indices,
            name="selected_indices",
            ndim=1,
        )
        selected_count = len(selected)
        _require(
            selected_count <= support_limit,
            "planner selected more anchors than the support limit",
        )
        _require(
            np.all((selected >= 0) & (selected < candidate_count))
            and len(np.unique(selected)) == selected_count,
            "planner selected invalid or duplicate candidate indices",
        )
        selected_counts[row] = selected_count
        selected_indices[row, :selected_count] = selected

        declared_initial = _finite_scalar(
            selection.initial_query_variance_trace,
            name="initial_query_variance_trace",
            minimum=0.0,
            strictly_positive=True,
        )
        declared_final = float(selection.final_query_variance_trace)
        declared_total_cost = float(selection.total_cost)
        _require(
            np.isfinite(declared_final) and np.isfinite(declared_total_cost),
            "planner final trace and total cost must be finite",
        )
        row_traces: np.ndarray = np.full(
            support_limit + 1,
            declared_initial,
            dtype=np.float64,
        )
        row_costs: np.ndarray = np.zeros(
            support_limit + 1,
            dtype=np.float64,
        )
        trace_scale = max(1.0, abs(declared_initial), abs(declared_final))
        trace_tolerance = _TOLERANCE * trace_scale
        if selected_count:
            reductions = np.asarray(selection.query_trace_reductions, dtype=np.float64)
            selected_costs = np.asarray(selection.selected_costs, dtype=np.float64)
            _require(
                reductions.shape == selected_costs.shape == (selected_count,),
                "planner reduction or cost diagnostic shape changed",
            )
            _require(
                np.all(np.isfinite(reductions)) and np.all(np.isfinite(selected_costs)),
                "planner reduction or cost diagnostics must be finite",
            )
            cumulative_reductions = np.cumsum(reductions)
            row_traces[1 : selected_count + 1] -= cumulative_reductions
            reconstructed_final = float(row_traces[selected_count])
            _require(
                np.isclose(
                    reconstructed_final,
                    declared_final,
                    rtol=_TOLERANCE,
                    atol=trace_tolerance,
                ),
                "query trace reductions do not reconstruct the declared final trace",
            )
            _require(
                declared_final >= -trace_tolerance,
                "reconstructed query variance became materially negative",
            )
            row_traces[selected_count] = max(declared_final, 0.0)
            row_costs[1 : selected_count + 1] = np.cumsum(selected_costs)
            cost_scale = max(
                1.0,
                abs(declared_total_cost),
                abs(float(row_costs[selected_count])),
            )
            _require(
                np.isclose(
                    float(row_costs[selected_count]),
                    declared_total_cost,
                    rtol=_TOLERANCE,
                    atol=_TOLERANCE * cost_scale,
                ),
                "selected costs do not reconstruct the declared total cost",
            )
            row_traces[selected_count + 1 :] = row_traces[selected_count]
            row_costs[selected_count + 1 :] = declared_total_cost
        else:
            _require(
                np.isclose(
                    declared_final,
                    declared_initial,
                    rtol=_TOLERANCE,
                    atol=trace_tolerance,
                ),
                "zero-selection final trace differs from the initial trace",
            )
            _require(
                abs(declared_total_cost) <= _TOLERANCE,
                "zero-selection plan has nonzero total cost",
            )
        _require(
            np.all(np.isfinite(row_traces)) and np.all(np.isfinite(row_costs)),
            "reconstructed query curve contains non-finite values",
        )
        traces[row] = row_traces
        cumulative_costs[row] = row_costs
        crossings = np.flatnonzero(traces[row] / traces[row, 0] <= target + _TOLERANCE)
        if len(crossings):
            first_support[row] = int(crossings[0])

    return QueryAnchorSufficiencyCurveV1(
        precision_multipliers=precision,
        support_counts=np.arange(support_limit + 1, dtype=np.int64),
        selected_indices=selected_indices,
        selected_counts=selected_counts,
        query_variance_traces=traces,
        cumulative_costs=cumulative_costs,
        target_remaining_variance_fraction=target,
        first_sufficient_support=first_support,
    )


__all__ = [
    "DEFAULT_ANCHOR_PRECISION_MULTIPLIERS",
    "QueryAnchorSufficiencyCurveV1",
    "evaluate_query_anchor_sufficiency",
]
