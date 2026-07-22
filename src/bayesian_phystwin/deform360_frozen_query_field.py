"""Frozen, target-queryable displacement fields for Deform360 trajectories.

The field is committed as two trajectories on one shared set of frame-zero
nodes.  A frozen, pointwise decoder then evaluates both the primary and the
comparator at arbitrary frame-zero query positions.  Query locations never
alter the field parameters, neighbor ordering, length scale, or support rule.

This module is deliberately numerical and side-effect free.  It contains no
held-outcome loading, target-trajectory access, scoring, or artifact I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


OperatorId = Literal["nearest-v1", "gaussian-knn-normalized-v1"]
UnsupportedQueryPolicy = Literal["emit-prediction-and-mask-v1"]
_TIE_BREAK = "distance-then-anchor-id"
_EXACT_ANCHOR_RULE = "bit-exact-nodal-value"
_UNSUPPORTED_QUERY_POLICY = "emit-prediction-and-mask-v1"
_MINIMUM_METRIC_SCALE_M = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly_array(
    value: np.ndarray,
    *,
    dtype: np.dtype,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value)
    _require(array.dtype == dtype, f"{name} must have dtype {dtype}")
    _require(array.ndim == ndim, f"{name} must have rank {ndim}")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrozenFieldConfig:
    """An explicit decoder choice selected before held-out evaluation."""

    operator_id: OperatorId
    maximum_support_distance_m: float
    unsupported_query_policy: UnsupportedQueryPolicy
    gaussian_neighbor_count: int | None = None
    gaussian_length_scale_m: float | None = None
    tie_break_rule: str = _TIE_BREAK
    exact_anchor_rule: str = _EXACT_ANCHOR_RULE

    def __post_init__(self) -> None:
        _require(
            self.operator_id in {"nearest-v1", "gaussian-knn-normalized-v1"},
            "unsupported frozen field operator",
        )
        _require(
            np.isfinite(self.maximum_support_distance_m)
            and self.maximum_support_distance_m >= _MINIMUM_METRIC_SCALE_M,
            "maximum_support_distance_m is below the frozen numerical minimum",
        )
        _require(
            self.unsupported_query_policy == _UNSUPPORTED_QUERY_POLICY,
            "unsupported-query policy changed",
        )
        _require(
            self.tie_break_rule == _TIE_BREAK,
            "frozen field tie-break rule changed",
        )
        _require(
            self.exact_anchor_rule == _EXACT_ANCHOR_RULE,
            "frozen field exact-anchor rule changed",
        )
        if self.operator_id == "nearest-v1":
            _require(
                self.gaussian_neighbor_count is None
                and self.gaussian_length_scale_m is None,
                "nearest-v1 does not accept Gaussian parameters",
            )
            return
        count = self.gaussian_neighbor_count
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "Gaussian neighbor count must be a positive integer",
        )
        scale = self.gaussian_length_scale_m
        _require(
            isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and np.isfinite(scale)
            and scale >= _MINIMUM_METRIC_SCALE_M,
            "Gaussian length scale is below the frozen numerical minimum",
        )


@dataclass(frozen=True)
class FrozenFieldGeometry:
    """The only pre-outcome geometry needed for query and center exclusion."""

    anchor_ids: np.ndarray
    anchor_positions_m: np.ndarray
    assimilation_anchor_ids: np.ndarray

    def __post_init__(self) -> None:
        anchor_ids = _readonly_array(
            self.anchor_ids,
            dtype=np.dtype(np.int64),
            name="anchor_ids",
            ndim=1,
        )
        positions = _readonly_array(
            self.anchor_positions_m,
            dtype=np.dtype(np.float32),
            name="anchor_positions_m",
            ndim=2,
        )
        assimilation = _readonly_array(
            self.assimilation_anchor_ids,
            dtype=np.dtype(np.int64),
            name="assimilation_anchor_ids",
            ndim=1,
        )
        _require(
            len(anchor_ids) > 0 and positions.shape == (len(anchor_ids), 3),
            "anchor geometry must have nonempty shape (N, 3)",
        )
        _require(
            np.all(np.isfinite(positions)),
            "anchor positions must be finite",
        )
        _require(
            np.all(np.diff(anchor_ids) > 0),
            "anchor IDs must be strictly increasing",
        )
        _require(
            len(np.unique(positions, axis=0)) == len(positions),
            "anchor positions must be unique",
        )
        _require(
            len(assimilation) == 0 or np.all(np.diff(assimilation) > 0),
            "assimilation anchor IDs must be strictly increasing",
        )
        indices = np.searchsorted(anchor_ids, assimilation)
        _require(
            np.all(indices < len(anchor_ids))
            and np.array_equal(anchor_ids[indices], assimilation),
            "assimilation anchor ID is absent from the field geometry",
        )
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "anchor_positions_m", positions)
        object.__setattr__(self, "assimilation_anchor_ids", assimilation)

    @property
    def assimilation_positions_m(self) -> np.ndarray:
        """Return center positions in canonical assimilation-anchor order."""

        indices = np.searchsorted(self.anchor_ids, self.assimilation_anchor_ids)
        result = self.anchor_positions_m[indices].copy()
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class FrozenNodalDisplacementField:
    """Primary and comparator nodal trajectories under one decoder contract."""

    geometry: FrozenFieldGeometry
    primary_nodal_trajectory_m: np.ndarray
    comparator_nodal_trajectory_m: np.ndarray
    config: FrozenFieldConfig

    def __post_init__(self) -> None:
        primary = _readonly_array(
            self.primary_nodal_trajectory_m,
            dtype=np.dtype(np.float32),
            name="primary_nodal_trajectory_m",
            ndim=3,
        )
        comparator = _readonly_array(
            self.comparator_nodal_trajectory_m,
            dtype=np.dtype(np.float32),
            name="comparator_nodal_trajectory_m",
            ndim=3,
        )
        expected_tail = (len(self.geometry.anchor_ids), 3)
        _require(
            primary.shape == comparator.shape
            and len(primary) > 0
            and primary.shape[1:] == expected_tail,
            "nodal trajectories must share nonempty shape (T, N, 3)",
        )
        _require(
            np.all(np.isfinite(primary)) and np.all(np.isfinite(comparator)),
            "nodal trajectories must be finite",
        )
        _require(
            np.array_equal(primary[0], self.geometry.anchor_positions_m)
            and np.array_equal(comparator[0], self.geometry.anchor_positions_m),
            "both nodal trajectories must equal the anchors at frame zero",
        )
        if self.config.operator_id == "gaussian-knn-normalized-v1":
            _require(
                self.config.gaussian_neighbor_count is not None
                and self.config.gaussian_neighbor_count
                <= len(self.geometry.anchor_ids),
                "Gaussian neighbor count exceeds the anchor count",
            )
        object.__setattr__(self, "primary_nodal_trajectory_m", primary)
        object.__setattr__(self, "comparator_nodal_trajectory_m", comparator)


@dataclass(frozen=True)
class FrameZeroQuerySet:
    """An identity-preserving set of frame-zero positions and nothing else."""

    identity_ids: np.ndarray
    positions_m: np.ndarray

    def __post_init__(self) -> None:
        identities = _readonly_array(
            self.identity_ids,
            dtype=np.dtype(np.int64),
            name="query identity_ids",
            ndim=1,
        )
        positions = _readonly_array(
            self.positions_m,
            dtype=np.dtype(np.float32),
            name="query positions_m",
            ndim=2,
        )
        _require(
            len(identities) > 0 and positions.shape == (len(identities), 3),
            "query geometry must have nonempty shape (M, 3)",
        )
        _require(
            len(np.unique(identities)) == len(identities),
            "query identity IDs must be unique",
        )
        _require(np.all(np.isfinite(positions)), "query positions must be finite")
        object.__setattr__(self, "identity_ids", identities)
        object.__setattr__(self, "positions_m", positions)


@dataclass(frozen=True)
class FieldQueryResult:
    """Two queried trajectories with their shared interpolation diagnostics."""

    identity_ids: np.ndarray
    query_positions_m: np.ndarray
    primary_prediction_m: np.ndarray
    comparator_prediction_m: np.ndarray
    supported_identity_mask: np.ndarray
    exact_anchor_mask: np.ndarray
    nearest_anchor_ids: np.ndarray
    nearest_anchor_distance_m: np.ndarray
    kth_anchor_distance_m: np.ndarray
    neighbor_anchor_ids: np.ndarray
    neighbor_weights: np.ndarray

    def __post_init__(self) -> None:
        identities = np.asarray(self.identity_ids)
        query = np.asarray(self.query_positions_m)
        primary = np.asarray(self.primary_prediction_m)
        comparator = np.asarray(self.comparator_prediction_m)
        count = len(identities)
        _require(
            identities.dtype == np.dtype(np.int64)
            and identities.shape == (count,)
            and query.dtype == np.dtype(np.float32)
            and query.shape == (count, 3),
            "queried identities or positions changed",
        )
        _require(
            primary.dtype == comparator.dtype == np.dtype(np.float32)
            and primary.shape == comparator.shape
            and primary.ndim == 3
            and primary.shape[1:] == (count, 3),
            "queried trajectories must share shape (T, M, 3)",
        )
        diagnostics = {
            "supported_identity_mask": (
                self.supported_identity_mask,
                np.dtype(bool),
                1,
            ),
            "exact_anchor_mask": (self.exact_anchor_mask, np.dtype(bool), 1),
            "nearest_anchor_ids": (self.nearest_anchor_ids, np.dtype(np.int64), 1),
            "nearest_anchor_distance_m": (
                self.nearest_anchor_distance_m,
                np.dtype(np.float64),
                1,
            ),
            "kth_anchor_distance_m": (
                self.kth_anchor_distance_m,
                np.dtype(np.float64),
                1,
            ),
            "neighbor_anchor_ids": (
                self.neighbor_anchor_ids,
                np.dtype(np.int64),
                2,
            ),
            "neighbor_weights": (self.neighbor_weights, np.dtype(np.float64), 2),
        }
        copied: dict[str, np.ndarray] = {}
        for name, (value, dtype, ndim) in diagnostics.items():
            array = _readonly_array(value, dtype=dtype, name=name, ndim=ndim)
            _require(array.shape[0] == count, f"{name} query count changed")
            copied[name] = array
        _require(
            copied["neighbor_anchor_ids"].shape == copied["neighbor_weights"].shape,
            "neighbor IDs and weights must share shape",
        )
        _require(
            np.all(np.isfinite(primary))
            and np.all(np.isfinite(comparator))
            and np.all(np.isfinite(copied["nearest_anchor_distance_m"]))
            and np.all(np.isfinite(copied["kth_anchor_distance_m"]))
            and np.all(np.isfinite(copied["neighbor_weights"])),
            "queried field output must be finite",
        )
        _require(
            np.all(copied["neighbor_weights"] >= 0.0)
            and np.allclose(
                np.sum(copied["neighbor_weights"], axis=1),
                1.0,
                rtol=0.0,
                atol=1e-15,
            ),
            "neighbor weights must be nonnegative and normalized",
        )
        for name, value in (
            ("identity_ids", identities),
            ("query_positions_m", query),
            ("primary_prediction_m", primary),
            ("comparator_prediction_m", comparator),
        ):
            result = value.copy()
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        for name, value in copied.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CenterExclusion:
    """A geometry-only one-to-one mapping of centers onto query identities."""

    assimilation_anchor_ids: np.ndarray
    mapped_query_identity_ids: np.ndarray
    mapped_query_indices: np.ndarray
    assignment_distance_m: np.ndarray
    excluded_query_mask: np.ndarray
    maximum_distance_m: float

    def __post_init__(self) -> None:
        arrays = {
            "assimilation_anchor_ids": (
                self.assimilation_anchor_ids,
                np.dtype(np.int64),
            ),
            "mapped_query_identity_ids": (
                self.mapped_query_identity_ids,
                np.dtype(np.int64),
            ),
            "mapped_query_indices": (self.mapped_query_indices, np.dtype(np.int64)),
            "assignment_distance_m": (
                self.assignment_distance_m,
                np.dtype(np.float64),
            ),
            "excluded_query_mask": (self.excluded_query_mask, np.dtype(bool)),
        }
        copied: dict[str, np.ndarray] = {}
        for name, (value, dtype) in arrays.items():
            copied[name] = _readonly_array(value, dtype=dtype, name=name, ndim=1)
        center_count = len(copied["assimilation_anchor_ids"])
        _require(
            all(
                len(copied[name]) == center_count
                for name in (
                    "mapped_query_identity_ids",
                    "mapped_query_indices",
                    "assignment_distance_m",
                )
            ),
            "center exclusion arrays differ in length",
        )
        _require(
            len(np.unique(copied["mapped_query_identity_ids"])) == center_count
            and len(np.unique(copied["mapped_query_indices"])) == center_count,
            "center exclusion mapping contains collisions",
        )
        _require(
            np.isfinite(self.maximum_distance_m) and self.maximum_distance_m > 0.0,
            "center exclusion radius must be positive",
        )
        _require(
            np.all(np.isfinite(copied["assignment_distance_m"]))
            and np.all(copied["assignment_distance_m"] <= self.maximum_distance_m),
            "center exclusion assignment exceeds its radius",
        )
        _require(
            int(np.sum(copied["excluded_query_mask"])) == center_count,
            "center exclusion mask count changed",
        )
        for name, value in copied.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RadiusUnionCenterExclusion:
    """An x0-only radius union plus deterministic per-center nearest audits."""

    assimilation_anchor_ids: np.ndarray
    nearest_query_identity_ids: np.ndarray
    nearest_query_indices: np.ndarray
    nearest_query_distance_m: np.ndarray
    center_within_radius_mask: np.ndarray
    excluded_query_mask: np.ndarray
    maximum_distance_m: float

    def __post_init__(self) -> None:
        arrays = {
            "assimilation_anchor_ids": (
                self.assimilation_anchor_ids,
                np.dtype(np.int64),
            ),
            "nearest_query_identity_ids": (
                self.nearest_query_identity_ids,
                np.dtype(np.int64),
            ),
            "nearest_query_indices": (
                self.nearest_query_indices,
                np.dtype(np.int64),
            ),
            "nearest_query_distance_m": (
                self.nearest_query_distance_m,
                np.dtype(np.float64),
            ),
            "center_within_radius_mask": (
                self.center_within_radius_mask,
                np.dtype(bool),
            ),
            "excluded_query_mask": (self.excluded_query_mask, np.dtype(bool)),
        }
        copied: dict[str, np.ndarray] = {}
        for name, (value, dtype) in arrays.items():
            copied[name] = _readonly_array(value, dtype=dtype, name=name, ndim=1)
        center_count = len(copied["assimilation_anchor_ids"])
        query_count = len(copied["excluded_query_mask"])
        _require(center_count > 0, "radius-union exclusion requires a center")
        _require(query_count > 0, "radius-union exclusion requires a query")
        _require(
            all(
                len(copied[name]) == center_count
                for name in (
                    "nearest_query_identity_ids",
                    "nearest_query_indices",
                    "nearest_query_distance_m",
                    "center_within_radius_mask",
                )
            ),
            "radius-union per-center arrays differ in length",
        )
        _require(
            np.all(np.diff(copied["assimilation_anchor_ids"]) > 0),
            "radius-union assimilation anchor IDs must be strictly increasing",
        )
        _require(
            np.all(
                (0 <= copied["nearest_query_indices"])
                & (copied["nearest_query_indices"] < query_count)
            ),
            "radius-union nearest query index is invalid",
        )
        _require(
            np.isfinite(self.maximum_distance_m) and self.maximum_distance_m > 0.0,
            "radius-union exclusion radius must be positive",
        )
        distance = copied["nearest_query_distance_m"]
        _require(
            np.all(np.isfinite(distance)) and np.all(distance >= 0.0),
            "radius-union nearest query distances must be finite and nonnegative",
        )
        within = copied["center_within_radius_mask"]
        _require(
            np.array_equal(within, distance <= self.maximum_distance_m),
            "radius-union center-within-radius mask changed",
        )
        _require(
            bool(np.any(within)) == bool(np.any(copied["excluded_query_mask"])),
            "radius-union center and query coverage disagree",
        )
        for name, value in copied.items():
            object.__setattr__(self, name, value)


def build_frozen_nodal_field(
    frame_zero_points_m: np.ndarray,
    primary_prediction_m: np.ndarray,
    comparator_prediction_m: np.ndarray,
    center_ids: np.ndarray,
    *,
    config: FrozenFieldConfig,
) -> FrozenNodalDisplacementField:
    """Commit two nodal trajectories under one explicitly supplied decoder."""

    frame_zero = np.asarray(frame_zero_points_m)
    centers = np.asarray(center_ids)
    _require(
        frame_zero.dtype == np.dtype(np.float32)
        and frame_zero.ndim == 2
        and frame_zero.shape[1] == 3,
        "frame_zero_points_m must have dtype float32 and shape (N, 3)",
    )
    _require(
        centers.dtype.kind in "iu" and centers.ndim == 1,
        "center_ids must be an integer vector",
    )
    normalized_centers = np.sort(centers.astype(np.int64, copy=False))
    _require(
        len(np.unique(normalized_centers)) == len(normalized_centers)
        and np.all((0 <= normalized_centers) & (normalized_centers < len(frame_zero))),
        "center_ids must be unique valid anchor indices",
    )
    geometry = FrozenFieldGeometry(
        anchor_ids=np.arange(len(frame_zero), dtype=np.int64),
        anchor_positions_m=frame_zero,
        assimilation_anchor_ids=normalized_centers,
    )
    return FrozenNodalDisplacementField(
        geometry=geometry,
        primary_nodal_trajectory_m=primary_prediction_m,
        comparator_nodal_trajectory_m=comparator_prediction_m,
        config=config,
    )


def _neighbor_indices_and_weights(
    field: FrozenNodalDisplacementField,
    query_position_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    anchors = field.geometry.anchor_positions_m
    anchor_ids = field.geometry.anchor_ids
    delta = anchors.astype(np.float64) - query_position_m.astype(np.float64)
    squared_distance = np.sum(np.square(delta), axis=1, dtype=np.float64)
    if field.config.operator_id == "nearest-v1":
        neighbor_count = 1
    else:
        neighbor_count = int(field.config.gaussian_neighbor_count or 0)
    order = np.lexsort((anchor_ids, squared_distance))[:neighbor_count]
    distance = np.sqrt(squared_distance[order])

    anchor_bits = anchors.view(np.uint32).reshape(len(anchors), 3)
    query_bits = np.asarray(query_position_m).view(np.uint32).reshape(3)
    exact_values = np.flatnonzero(np.all(anchor_bits == query_bits[None], axis=1))
    exact_index = None if len(exact_values) == 0 else int(exact_values[0])

    if field.config.operator_id == "nearest-v1":
        weights = np.ones(1, dtype=np.float64)
    elif exact_index is not None:
        exact_local = np.flatnonzero(order == exact_index)
        _require(len(exact_local) == 1, "exact anchor absent from Gaussian neighbors")
        weights = np.zeros(neighbor_count, dtype=np.float64)
        weights[int(exact_local[0])] = 1.0
    else:
        length_scale = float(field.config.gaussian_length_scale_m or 0.0)
        selected_square = squared_distance[order]
        relative_square = np.maximum(selected_square - selected_square[0], 0.0)
        relative_energy = relative_square / (2.0 * length_scale**2)
        # Values beyond this point are already numerically indistinguishable
        # from zero in float64.  Clipping also prevents otherwise valid finite
        # geometries from overflowing for an unusually small locked scale.
        relative_energy = np.minimum(relative_energy, 745.0)
        weights = np.exp(-relative_energy)
        weights[relative_energy >= 745.0] = 0.0
        weights /= np.sum(weights, dtype=np.float64)
    return order, distance, weights, exact_index


def query_frozen_nodal_field(
    field: FrozenNodalDisplacementField,
    queries: FrameZeroQuerySet,
) -> FieldQueryResult:
    """Evaluate both arms with shared, pointwise, target-count-invariant weights."""

    query_count = len(queries.identity_ids)
    frame_count = len(field.primary_nodal_trajectory_m)
    neighbor_count = (
        1
        if field.config.operator_id == "nearest-v1"
        else int(field.config.gaussian_neighbor_count or 0)
    )
    primary_output = np.empty((frame_count, query_count, 3), dtype=np.float32)
    comparator_output = np.empty_like(primary_output)
    supported = np.zeros(query_count, dtype=bool)
    exact = np.zeros(query_count, dtype=bool)
    nearest_ids = np.empty(query_count, dtype=np.int64)
    nearest_distance = np.empty(query_count, dtype=np.float64)
    kth_distance = np.empty(query_count, dtype=np.float64)
    neighbor_ids = np.empty((query_count, neighbor_count), dtype=np.int64)
    neighbor_weights = np.empty((query_count, neighbor_count), dtype=np.float64)

    anchors64 = field.geometry.anchor_positions_m.astype(np.float64)
    primary_displacement = (
        field.primary_nodal_trajectory_m.astype(np.float64) - anchors64[None]
    )
    comparator_displacement = (
        field.comparator_nodal_trajectory_m.astype(np.float64) - anchors64[None]
    )
    for query_index, query_position in enumerate(queries.positions_m):
        indices, distances, weights, exact_index = _neighbor_indices_and_weights(
            field, query_position
        )
        neighbor_ids[query_index] = field.geometry.anchor_ids[indices]
        neighbor_weights[query_index] = weights
        nearest_ids[query_index] = neighbor_ids[query_index, 0]
        nearest_distance[query_index] = distances[0]
        kth_distance[query_index] = distances[-1]
        supported[query_index] = distances[0] <= field.config.maximum_support_distance_m
        query64 = query_position.astype(np.float64)
        primary_delta = np.sum(
            primary_displacement[:, indices] * weights[None, :, None],
            axis=1,
            dtype=np.float64,
        )
        comparator_delta = np.sum(
            comparator_displacement[:, indices] * weights[None, :, None],
            axis=1,
            dtype=np.float64,
        )
        primary_output[:, query_index] = (query64[None] + primary_delta).astype(
            np.float32
        )
        comparator_output[:, query_index] = (query64[None] + comparator_delta).astype(
            np.float32
        )
        if exact_index is not None:
            exact[query_index] = True
            primary_output[:, query_index] = field.primary_nodal_trajectory_m[
                :, exact_index
            ]
            comparator_output[:, query_index] = field.comparator_nodal_trajectory_m[
                :, exact_index
            ]

    return FieldQueryResult(
        identity_ids=queries.identity_ids,
        query_positions_m=queries.positions_m,
        primary_prediction_m=primary_output,
        comparator_prediction_m=comparator_output,
        supported_identity_mask=supported,
        exact_anchor_mask=exact,
        nearest_anchor_ids=nearest_ids,
        nearest_anchor_distance_m=nearest_distance,
        kth_anchor_distance_m=kth_distance,
        neighbor_anchor_ids=neighbor_ids,
        neighbor_weights=neighbor_weights,
    )


def _rectangular_min_cost_assignment(cost: np.ndarray) -> np.ndarray:
    """Return the deterministic Hungarian row-to-column assignment for N <= M."""

    matrix = np.asarray(cost, dtype=np.float64)
    _require(
        matrix.ndim == 2
        and matrix.shape[0] > 0
        and matrix.shape[0] <= matrix.shape[1]
        and np.all(np.isfinite(matrix)),
        "assignment cost must be finite with 0 < rows <= columns",
    )
    row_count, column_count = matrix.shape
    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    matched_row = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    matrix[row0 - 1, column - 1]
                    - row_potential[row0]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = column0
                if minimum[column] < delta or (
                    minimum[column] == delta and column < column1
                ):
                    delta = minimum[column]
                    column1 = column
            _require(np.isfinite(delta), "assignment has no augmenting path")
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = np.full(row_count, -1, dtype=np.int64)
    for column in range(1, column_count + 1):
        if matched_row[column] > 0:
            assignment[matched_row[column] - 1] = column - 1
    _require(np.all(assignment >= 0), "assignment does not cover every row")
    return assignment


def map_assimilation_centers_to_queries(
    geometry: FrozenFieldGeometry,
    queries: FrameZeroQuerySet,
    *,
    maximum_distance_m: float,
) -> CenterExclusion:
    """Map centers using only frame-zero geometry and deterministic identity ties."""

    _require(
        np.isfinite(maximum_distance_m) and maximum_distance_m > 0.0,
        "center exclusion radius must be positive",
    )
    centers = geometry.assimilation_positions_m.astype(np.float64)
    _require(
        len(centers) > 0,
        "center exclusion requires at least one assimilation center",
    )
    _require(
        len(queries.identity_ids) >= len(centers),
        "too few query identities for center exclusion",
    )
    query_order = np.argsort(queries.identity_ids, kind="stable")
    ordered_queries = queries.positions_m[query_order].astype(np.float64)
    distance = np.linalg.norm(centers[:, None, :] - ordered_queries[None, :, :], axis=2)
    allowed = distance <= maximum_distance_m
    _require(
        np.all(np.any(allowed, axis=1)),
        "an assimilation center has no query identity within the exclusion radius",
    )
    penalty = (len(centers) + 1) * max(1.0, maximum_distance_m) + 1.0
    assignment = _rectangular_min_cost_assignment(np.where(allowed, distance, penalty))
    assigned_distance = distance[np.arange(len(centers)), assignment]
    _require(
        np.all(assigned_distance <= maximum_distance_m),
        "no collision-free center exclusion assignment exists within the radius",
    )
    mapped_indices = query_order[assignment]
    mapped_identities = queries.identity_ids[mapped_indices]
    excluded = np.zeros(len(queries.identity_ids), dtype=bool)
    excluded[mapped_indices] = True
    return CenterExclusion(
        assimilation_anchor_ids=geometry.assimilation_anchor_ids,
        mapped_query_identity_ids=mapped_identities,
        mapped_query_indices=mapped_indices.astype(np.int64, copy=False),
        assignment_distance_m=assigned_distance,
        excluded_query_mask=excluded,
        maximum_distance_m=float(maximum_distance_m),
    )


def build_radius_union_center_exclusion(
    geometry: FrozenFieldGeometry,
    queries: FrameZeroQuerySet,
    *,
    maximum_distance_m: float,
) -> RadiusUnionCenterExclusion:
    """Exclude every x0 query within radius of any assimilation center.

    The per-center nearest-query records are audit metadata only.  Queries are
    sorted by identity before nearest-neighbor selection, so exact distance ties
    select the smaller identity independently of input order.  A center whose
    nearest query lies outside the radius excludes no query by itself.
    """

    _require(
        np.isfinite(maximum_distance_m) and maximum_distance_m > 0.0,
        "radius-union exclusion radius must be positive",
    )
    centers = geometry.assimilation_positions_m.astype(np.float64)
    _require(
        len(centers) > 0,
        "radius-union exclusion requires at least one assimilation center",
    )
    query_order = np.argsort(queries.identity_ids, kind="stable")
    ordered_queries = queries.positions_m[query_order].astype(np.float64)
    distance = np.linalg.norm(centers[:, None, :] - ordered_queries[None, :, :], axis=2)
    _require(
        np.all(np.isfinite(distance)),
        "radius-union center-to-query distances must be finite",
    )

    nearest_ordered_indices = np.argmin(distance, axis=1)
    nearest_distance = distance[np.arange(len(centers)), nearest_ordered_indices]
    nearest_indices = query_order[nearest_ordered_indices]
    nearest_identities = queries.identity_ids[nearest_indices]
    center_within_radius = nearest_distance <= maximum_distance_m

    excluded = np.zeros(len(queries.identity_ids), dtype=bool)
    excluded[query_order] = np.any(distance <= maximum_distance_m, axis=0)
    return RadiusUnionCenterExclusion(
        assimilation_anchor_ids=geometry.assimilation_anchor_ids,
        nearest_query_identity_ids=nearest_identities,
        nearest_query_indices=nearest_indices.astype(np.int64, copy=False),
        nearest_query_distance_m=nearest_distance,
        center_within_radius_mask=center_within_radius,
        excluded_query_mask=excluded,
        maximum_distance_m=float(maximum_distance_m),
    )


__all__ = [
    "CenterExclusion",
    "FieldQueryResult",
    "FrameZeroQuerySet",
    "FrozenFieldConfig",
    "FrozenFieldGeometry",
    "FrozenNodalDisplacementField",
    "RadiusUnionCenterExclusion",
    "UnsupportedQueryPolicy",
    "build_frozen_nodal_field",
    "build_radius_union_center_exclusion",
    "map_assimilation_centers_to_queries",
    "query_frozen_nodal_field",
]
