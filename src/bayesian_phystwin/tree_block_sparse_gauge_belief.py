"""Prior-aware IRLS with exact block-tree gauge elimination."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    _finite_array,
    _positive_semidefinite_square_root,
    _readonly,
    _regularized_precision,
    _require,
)
from ._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _full_covariance,
    _group_layout,
    _student_t_mixture_statistics,
    _whiten,
)
from .sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
    _anchor_prediction,
    _NuisanceLayout,
    _observation_prediction,
    _whiten_sparse_observations,
)
from .tree_block_gaussian import (
    TreeBlockFactorizationV1,
    TreeBlockNormalSystemV1,
)

TREE_BLOCK_SPARSE_GAUGE_SOLVER_SCHEMA = (
    "bayesian_phystwin.tree_block_sparse_gauge_solver"
)
TREE_BLOCK_SPARSE_GAUGE_SOLVER_VERSION = 1
TREE_BLOCK_POSTERIOR_COVARIANCE_SCHEMA = (
    "bayesian_phystwin.tree_block_posterior_covariance"
)
TREE_BLOCK_POSTERIOR_COVARIANCE_VERSION = 1
TREE_BLOCK_GAUGE_AWARE_RESULT_SCHEMA = "bayesian_phystwin.tree_block_gauge_aware_result"
TREE_BLOCK_GAUGE_AWARE_RESULT_VERSION = 1


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _materialization_budget(
    maximum_bytes: int | None,
    *,
    required_bytes: int,
) -> None:
    if maximum_bytes is None:
        return
    _require(
        type(maximum_bytes) is int and maximum_bytes >= 0,
        "maximum_bytes must be a nonnegative integer or None",
    )
    if required_bytes > maximum_bytes:
        raise MemoryError(
            f"covariance materialization requires {required_bytes} bytes, "
            f"exceeding the {maximum_bytes}-byte limit"
        )


@dataclass(frozen=True, slots=True)
class _GlobalLayout:
    state_count: int
    shared_count: int
    view_count: int
    anchor_bias_count: int

    @property
    def state_slice(self) -> slice:
        return slice(0, self.state_count)

    @property
    def shared_slice(self) -> slice:
        return slice(
            self.state_count,
            self.state_count + self.shared_count,
        )

    @property
    def view_slice(self) -> slice:
        return slice(
            self.shared_slice.stop,
            self.shared_slice.stop + self.view_count,
        )

    @property
    def anchor_bias_slice(self) -> slice:
        return slice(
            self.view_slice.stop,
            self.view_slice.stop + self.anchor_bias_count,
        )

    @property
    def bias_count(self) -> int:
        return self.shared_count + self.view_count + self.anchor_bias_count

    @property
    def global_count(self) -> int:
        return self.anchor_bias_slice.stop


def _tree_prior_blocks(
    gauge: TreeSparseGaugeDesignV1,
) -> tuple[np.ndarray, np.ndarray]:
    node_precision = np.zeros(
        (gauge.gauge_count, gauge.block_size, gauge.block_size),
        dtype=np.float64,
    )
    parent_coupling = np.zeros_like(node_precision)
    identity = np.eye(gauge.block_size, dtype=np.float64)
    for index in range(gauge.gauge_count):
        inverse_scale = np.linalg.solve(
            gauge.innovation_scale_tril[index],
            identity,
        )
        innovation_precision = inverse_scale.T @ inverse_scale
        node_precision[index] += innovation_precision
        parent = int(gauge.parent_indices[index])
        if parent < 0:
            continue
        transition = gauge.transition_matrices[index]
        parent_coupling[index] = -innovation_precision @ transition
        node_precision[parent] += transition.T @ innovation_precision @ transition
    return node_precision, parent_coupling


def _bias_prior_precision(
    batch: GaugeAwareObservationBatch,
    layout: _GlobalLayout,
    config: PriorAwareGaugeConfigV1,
) -> np.ndarray:
    precision: np.ndarray = np.zeros(
        (layout.global_count, layout.global_count),
        dtype=np.float64,
    )
    if layout.shared_count:
        precision[layout.shared_slice, layout.shared_slice] = (
            np.eye(layout.shared_count) / config.shared_bias_prior_std_m**2
        )
    if layout.view_count:
        precision[layout.view_slice, layout.view_slice] = (
            np.eye(layout.view_count) / config.view_bias_prior_std_m**2
        )
    if layout.anchor_bias_count:
        if batch.anchor_bias_prior_covariance is None:
            raise ValueError("anchor bias prior covariance is missing")
        precision[
            layout.anchor_bias_slice,
            layout.anchor_bias_slice,
        ] = _regularized_precision(
            np.asarray(batch.anchor_bias_prior_covariance),
            "anchor bias prior covariance",
            eigenvalue_floor=config.prior_eigenvalue_floor,
        )
    return precision


def _add_symmetric_cross(
    matrix: np.ndarray,
    left: slice,
    right: slice,
    value: np.ndarray,
) -> None:
    matrix[left, right] += value
    matrix[right, left] += value.T


def _build_tree_system(
    *,
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    layout: _GlobalLayout,
    observation_weight: np.ndarray,
    state: np.ndarray,
    local_gauge: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    target: np.ndarray,
    anchor_weight: np.ndarray,
    anchor_state: np.ndarray,
    anchor_bias: np.ndarray,
    anchor_target: np.ndarray,
    config: PriorAwareGaugeConfigV1,
    state_prior_precision: np.ndarray | None,
) -> TreeBlockNormalSystemV1:
    node_precision, parent_coupling = _tree_prior_blocks(gauge)
    global_precision = _bias_prior_precision(batch, layout, config)
    if state_prior_precision is not None:
        prior = np.asarray(state_prior_precision, dtype=np.float64)
        _require(
            prior.shape == (layout.state_count, layout.state_count),
            "state prior precision shape changed",
        )
        global_precision[layout.state_slice, layout.state_slice] += prior

    global_coupling = np.zeros(
        (gauge.gauge_count, gauge.block_size, layout.global_count),
        dtype=np.float64,
    )
    node_right = np.zeros(
        (gauge.gauge_count, gauge.block_size),
        dtype=np.float64,
    )
    global_right: np.ndarray = np.zeros(layout.global_count, dtype=np.float64)
    state_slice = layout.state_slice

    global_precision[state_slice, state_slice] += np.einsum(
        "m,mci,mcj->ij",
        observation_weight,
        state,
        state,
        optimize=True,
    )
    global_right[state_slice] += np.einsum(
        "m,mci,mc->i",
        observation_weight,
        state,
        target,
        optimize=True,
    )
    if layout.shared_count:
        shared_slice = layout.shared_slice
        global_precision[shared_slice, shared_slice] += np.einsum(
            "m,mci,mcj->ij",
            observation_weight,
            shared,
            shared,
            optimize=True,
        )
        _add_symmetric_cross(
            global_precision,
            state_slice,
            shared_slice,
            np.einsum(
                "m,mci,mcj->ij",
                observation_weight,
                state,
                shared,
                optimize=True,
            ),
        )
        global_right[shared_slice] += np.einsum(
            "m,mci,mc->i",
            observation_weight,
            shared,
            target,
            optimize=True,
        )
    if layout.view_count:
        view_slice = layout.view_slice
        global_precision[view_slice, view_slice] += np.einsum(
            "m,mci,mcj->ij",
            observation_weight,
            view,
            view,
            optimize=True,
        )
        _add_symmetric_cross(
            global_precision,
            state_slice,
            view_slice,
            np.einsum(
                "m,mci,mcj->ij",
                observation_weight,
                state,
                view,
                optimize=True,
            ),
        )
        global_right[view_slice] += np.einsum(
            "m,mci,mc->i",
            observation_weight,
            view,
            target,
            optimize=True,
        )
    if layout.shared_count and layout.view_count:
        _add_symmetric_cross(
            global_precision,
            layout.shared_slice,
            layout.view_slice,
            np.einsum(
                "m,mci,mcj->ij",
                observation_weight,
                shared,
                view,
                optimize=True,
            ),
        )

    for gauge_index in range(gauge.gauge_count):
        selected = gauge.gauge_indices == gauge_index
        if not np.any(selected):
            continue
        local_weight = observation_weight[selected]
        local_design = local_gauge[selected]
        node_precision[gauge_index] += np.einsum(
            "m,mci,mcj->ij",
            local_weight,
            local_design,
            local_design,
            optimize=True,
        )
        global_coupling[gauge_index, :, state_slice] += np.einsum(
            "m,mci,mcj->ij",
            local_weight,
            local_design,
            state[selected],
            optimize=True,
        )
        if layout.shared_count:
            global_coupling[
                gauge_index,
                :,
                layout.shared_slice,
            ] += np.einsum(
                "m,mci,mcj->ij",
                local_weight,
                local_design,
                shared[selected],
                optimize=True,
            )
        if layout.view_count:
            global_coupling[
                gauge_index,
                :,
                layout.view_slice,
            ] += np.einsum(
                "m,mci,mcj->ij",
                local_weight,
                local_design,
                view[selected],
                optimize=True,
            )
        node_right[gauge_index] += np.einsum(
            "m,mci,mc->i",
            local_weight,
            local_design,
            target[selected],
            optimize=True,
        )

    if len(anchor_weight):
        global_precision[state_slice, state_slice] += np.einsum(
            "a,aci,acj->ij",
            anchor_weight,
            anchor_state,
            anchor_state,
            optimize=True,
        )
        global_right[state_slice] += np.einsum(
            "a,aci,ac->i",
            anchor_weight,
            anchor_state,
            anchor_target,
            optimize=True,
        )
        if layout.anchor_bias_count:
            bias_slice = layout.anchor_bias_slice
            global_precision[bias_slice, bias_slice] += np.einsum(
                "a,aci,acj->ij",
                anchor_weight,
                anchor_bias,
                anchor_bias,
                optimize=True,
            )
            _add_symmetric_cross(
                global_precision,
                state_slice,
                bias_slice,
                np.einsum(
                    "a,aci,acj->ij",
                    anchor_weight,
                    anchor_state,
                    anchor_bias,
                    optimize=True,
                ),
            )
            global_right[bias_slice] += np.einsum(
                "a,aci,ac->i",
                anchor_weight,
                anchor_bias,
                anchor_target,
                optimize=True,
            )

    return TreeBlockNormalSystemV1(
        parent_indices=gauge.parent_indices,
        node_precision=node_precision,
        parent_coupling=parent_coupling,
        global_coupling=global_coupling,
        global_precision=0.5 * (global_precision + global_precision.T),
        node_right=node_right,
        global_right=global_right,
    )


def _known_state_information(
    observation_weight: np.ndarray,
    state: np.ndarray,
    anchor_weight: np.ndarray,
    anchor_state: np.ndarray,
) -> np.ndarray:
    known = np.einsum(
        "m,mci,mcj->ij",
        observation_weight,
        state,
        state,
        optimize=True,
    )
    if len(anchor_weight):
        known += np.einsum(
            "a,aci,acj->ij",
            anchor_weight,
            anchor_state,
            anchor_state,
            optimize=True,
        )
    return 0.5 * (known + known.T)


def _prior_aware_tree_basis(
    *,
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    observation_weight: np.ndarray,
    state: np.ndarray,
    local_gauge: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    anchor_weight: np.ndarray,
    anchor_state: np.ndarray,
    anchor_bias: np.ndarray,
    query: np.ndarray,
    state_prior: np.ndarray,
    config: PriorAwareGaugeConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    layout = _GlobalLayout(
        state_count=state.shape[2],
        shared_count=shared.shape[2],
        view_count=view.shape[2],
        anchor_bias_count=anchor_bias.shape[2],
    )
    system = _build_tree_system(
        batch=batch,
        gauge=gauge,
        layout=layout,
        observation_weight=observation_weight,
        state=state,
        local_gauge=local_gauge,
        shared=shared,
        view=view,
        target=np.zeros_like(batch.innovation_m),
        anchor_weight=anchor_weight,
        anchor_state=anchor_state,
        anchor_bias=anchor_bias,
        anchor_target=np.zeros((len(anchor_weight), 3), dtype=np.float64),
        config=config,
        state_prior_precision=None,
    )
    elimination = system.eliminate_nodes(
        maximum_condition_number=config.maximum_condition_number
    )
    global_information = elimination.global_schur
    state_slice = layout.state_slice
    conditional = global_information[state_slice, state_slice]
    bias_count = layout.bias_count
    bias_condition = 0.0
    if bias_count:
        bias_slice = slice(layout.state_count, layout.global_count)
        cross = global_information[state_slice, bias_slice]
        bias_information = global_information[bias_slice, bias_slice]
        bias_condition = float(np.linalg.cond(bias_information))
        if (
            not np.isfinite(bias_condition)
            or bias_condition > config.maximum_condition_number
        ):
            raise np.linalg.LinAlgError(
                "identifiability bias Schur complement is ill-conditioned"
            )
        conditional = conditional - cross @ np.linalg.solve(
            bias_information,
            cross.T,
        )
    conditional = 0.5 * (conditional + conditional.T)
    known = _known_state_information(
        observation_weight,
        state,
        anchor_weight,
        anchor_state,
    )
    state_square_root = _positive_semidefinite_square_root(
        state_prior,
        "state prior covariance",
        eigenvalue_floor=config.prior_eigenvalue_floor,
    )
    standardized = state_square_root.T @ conditional @ state_square_root
    standardized = 0.5 * (standardized + standardized.T)
    eigenvalues, eigenvectors = np.linalg.eigh(standardized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum_information = float(np.max(eigenvalues, initial=0.0))
    query_flat = query.reshape(-1, query.shape[2])
    candidates: list[tuple[np.ndarray, float, float, float]] = []
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= config.prior_eigenvalue_floor:
            continue
        direction = state_square_root @ eigenvectors[:, index]
        known_value = float(direction @ known @ direction)
        conditional_value = float(direction @ conditional @ direction)
        identifiable = conditional_value / max(
            known_value,
            config.prior_eigenvalue_floor,
        )
        candidates.append(
            (
                direction,
                float(np.linalg.norm(query_flat @ direction)),
                float(eigenvalue),
                identifiable,
            )
        )
    maximum_query = max((item[1] for item in candidates), default=0.0)
    retained: list[np.ndarray] = []
    identifiable_fractions: list[float] = []
    query_fractions: list[float] = []
    for direction, query_norm, information, identifiable in candidates:
        information_fraction = information / max(
            maximum_information,
            config.prior_eigenvalue_floor,
        )
        query_fraction = query_norm / maximum_query if maximum_query else 0.0
        if (
            information_fraction >= config.minimum_conditional_information_fraction
            and identifiable >= config.minimum_identifiable_fraction
            and query_fraction >= config.minimum_query_sensitivity_fraction
        ):
            retained.append(direction)
            identifiable_fractions.append(min(1.0, max(0.0, identifiable)))
            query_fractions.append(min(1.0, max(0.0, query_fraction)))
    mapping = (
        np.column_stack(retained)
        if retained
        else np.zeros((state.shape[2], 0), dtype=np.float64)
    )
    return (
        mapping,
        np.asarray(identifiable_fractions),
        np.asarray(query_fractions),
        {
            "maximum_conditional_information_eigenvalue": maximum_information,
            "maximum_query_sensitivity_norm": maximum_query,
            "identification_tree_maximum_node_condition_number": (
                elimination.maximum_node_condition_number
            ),
            "identification_bias_condition_number": bias_condition,
            "identification_global_schur_dimension": float(layout.global_count),
        },
    )


def _legacy_order_solution(
    global_solution: np.ndarray,
    node_solution: np.ndarray,
    *,
    layout: _GlobalLayout,
) -> np.ndarray:
    return np.concatenate(
        (
            global_solution[layout.state_slice],
            node_solution.reshape(-1),
            global_solution[layout.shared_slice],
            global_solution[layout.view_slice],
            global_solution[layout.anchor_bias_slice],
        )
    )


@dataclass(frozen=True, slots=True)
class TreeBlockPosteriorCovarianceV1:
    """Posterior precision factors with prior-preserving state expansion."""

    state_prior_covariance: np.ndarray
    state_mapping: np.ndarray
    factorization: TreeBlockFactorizationV1
    bias_count: int

    def __post_init__(self) -> None:
        state_prior = _finite_array(
            self.state_prior_covariance,
            "state_prior_covariance",
            2,
        )
        _require(
            state_prior.shape[0] == state_prior.shape[1],
            "state_prior_covariance must be square",
        )
        _require(
            np.allclose(state_prior, state_prior.T, atol=1e-10, rtol=1e-10),
            "state_prior_covariance must be symmetric",
        )
        mapping = _finite_array(self.state_mapping, "state_mapping", 2)
        _require(
            mapping.shape[0] == len(state_prior),
            "state_mapping has changed state dimension",
        )
        if not isinstance(self.factorization, TreeBlockFactorizationV1):
            raise TypeError("factorization must be a TreeBlockFactorizationV1")
        if type(self.bias_count) is not int or self.bias_count < 0:
            raise ValueError("bias_count must be a nonnegative integer")
        _require(
            self.factorization.global_size == mapping.shape[1] + self.bias_count,
            "factorization global dimension differs from state/bias layout",
        )
        object.__setattr__(
            self,
            "state_prior_covariance",
            _readonly(state_prior),
        )
        object.__setattr__(self, "state_mapping", _readonly(mapping))

    @property
    def representation(self) -> str:
        return "tree-block-posterior-precision-v1"

    @property
    def state_count(self) -> int:
        return len(self.state_prior_covariance)

    @property
    def retained_state_count(self) -> int:
        return self.state_mapping.shape[1]

    @property
    def gauge_parameter_count(self) -> int:
        return self.factorization.node_count * self.factorization.block_size

    @property
    def nuisance_count(self) -> int:
        return self.gauge_parameter_count + self.bias_count

    @property
    def dimension(self) -> int:
        return self.state_count + self.nuisance_count

    @property
    def stored_nbytes(self) -> int:
        return int(
            self.state_prior_covariance.nbytes
            + self.state_mapping.nbytes
            + self.factorization.stored_nbytes
        )

    @property
    def estimated_dense_covariance_bytes(self) -> int:
        return self.dimension**2 * np.dtype(np.float64).itemsize

    @property
    def estimated_peak_materialization_bytes(self) -> int:
        internal = self.factorization.dimension**2 * np.dtype(np.float64).itemsize
        return internal + self.estimated_dense_covariance_bytes

    @property
    def dense_materialized(self) -> bool:
        return False

    def state_marginal_covariance(self) -> np.ndarray:
        """Return the state marginal without materializing gauge covariance."""

        global_covariance = self.factorization.global_marginal_covariance()
        reduced = global_covariance[
            : self.retained_state_count,
            : self.retained_state_count,
        ]
        result = np.array(self.state_prior_covariance, copy=True)
        result += (
            self.state_mapping
            @ (reduced - np.eye(self.retained_state_count))
            @ self.state_mapping.T
        )
        return _readonly(0.5 * (result + result.T))

    def materialize(
        self,
        *,
        maximum_bytes: int | None = None,
    ) -> np.ndarray:
        """Materialize the historical full covariance after a peak budget."""

        _materialization_budget(
            maximum_bytes,
            required_bytes=self.estimated_peak_materialization_bytes,
        )
        internal = self.factorization.materialize_covariance()
        retained = self.retained_state_count
        global_size = self.factorization.global_size
        gauge_start = global_size
        gauge_stop = gauge_start + self.gauge_parameter_count
        order = np.concatenate(
            (
                np.arange(retained),
                np.arange(gauge_start, gauge_stop),
                np.arange(retained, global_size),
            )
        )
        reduced = internal[np.ix_(order, order)]
        return _readonly(
            _full_covariance(
                self.state_prior_covariance,
                self.state_mapping,
                reduced,
                self.nuisance_count,
            )
        )

    def descriptor(self) -> Mapping[str, Any]:
        return frozen_finite_json_mapping(
            {
                "schema": TREE_BLOCK_POSTERIOR_COVARIANCE_SCHEMA,
                "schema_version": TREE_BLOCK_POSTERIOR_COVARIANCE_VERSION,
                "representation": self.representation,
                "state_count": self.state_count,
                "retained_state_count": self.retained_state_count,
                "gauge_parameter_count": self.gauge_parameter_count,
                "bias_count": self.bias_count,
                "dimension": self.dimension,
                "stored_nbytes": self.stored_nbytes,
                "estimated_dense_covariance_bytes": (
                    self.estimated_dense_covariance_bytes
                ),
                "estimated_peak_materialization_bytes": (
                    self.estimated_peak_materialization_bytes
                ),
                "state_prior_covariance_sha256": _array_sha256(
                    self.state_prior_covariance
                ),
                "state_mapping_sha256": _array_sha256(self.state_mapping),
                "factorization": dict(self.factorization.descriptor()),
            },
            name="tree-block posterior covariance descriptor",
        )


@dataclass(frozen=True, slots=True)
class TreeBlockGaugeAwareBeliefResultV1:
    """Gauge-aware result whose accepted covariance remains tree-factorized."""

    inference_admissible: bool
    reason: str
    state_coefficients: np.ndarray
    gauge_delta: np.ndarray
    shared_bias_coefficients: np.ndarray
    view_bias_coefficients: np.ndarray
    anchor_bias_coefficients: np.ndarray
    covariance: TreeBlockPosteriorCovarianceV1
    identifiable_state_transform: np.ndarray
    identifiable_fractions: np.ndarray
    query_sensitivity_fractions: np.ndarray
    robust_weights: np.ndarray
    anchor_robust_weights: np.ndarray
    diagnostics: Mapping[str, Any]
    input_lineage: Mapping[str, Any] = field(default_factory=dict)
    _result_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.inference_admissible) is not bool:
            raise TypeError("inference_admissible must be a bool")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("reason must be a nonempty string")
        if not isinstance(self.covariance, TreeBlockPosteriorCovarianceV1):
            raise TypeError("covariance must be a TreeBlockPosteriorCovarianceV1")
        vectors: dict[str, np.ndarray] = {}
        for name in (
            "state_coefficients",
            "gauge_delta",
            "shared_bias_coefficients",
            "view_bias_coefficients",
            "anchor_bias_coefficients",
            "identifiable_fractions",
            "query_sensitivity_fractions",
            "robust_weights",
            "anchor_robust_weights",
        ):
            vectors[name] = _finite_array(getattr(self, name), name, 1)
        transform = _finite_array(
            self.identifiable_state_transform,
            "identifiable_state_transform",
            2,
        )
        state_count = len(vectors["state_coefficients"])
        _require(
            transform.shape[0] == state_count,
            "identifiable_state_transform has changed shape",
        )
        _require(
            vectors["identifiable_fractions"].shape
            == vectors["query_sensitivity_fractions"].shape
            == (transform.shape[1],),
            "identifiability diagnostics have changed shape",
        )
        parameter_count = sum(
            len(vectors[name])
            for name in (
                "state_coefficients",
                "gauge_delta",
                "shared_bias_coefficients",
                "view_bias_coefficients",
                "anchor_bias_coefficients",
            )
        )
        _require(
            self.covariance.dimension == parameter_count,
            "covariance dimension differs from result coefficients",
        )
        if not self.inference_admissible:
            for name in (
                "state_coefficients",
                "gauge_delta",
                "shared_bias_coefficients",
                "view_bias_coefficients",
                "anchor_bias_coefficients",
            ):
                _require(
                    np.count_nonzero(vectors[name]) == 0,
                    "rejected results must preserve zero candidate coefficients",
                )
        for name, value in vectors.items():
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(
            self,
            "identifiable_state_transform",
            _readonly(transform),
        )
        object.__setattr__(
            self,
            "diagnostics",
            frozen_finite_json_mapping(
                self.diagnostics,
                name="diagnostics",
            ),
        )
        object.__setattr__(
            self,
            "input_lineage",
            frozen_finite_json_mapping(
                self.input_lineage,
                name="input_lineage",
            ),
        )
        object.__setattr__(self, "_result_id", _canonical_id(self.descriptor()))

    @property
    def accepted(self) -> bool:
        return self.inference_admissible

    @property
    def result_id(self) -> str:
        return self._result_id

    @property
    def dense_covariance_materialized(self) -> bool:
        return False

    def materialize_posterior_covariance(
        self,
        *,
        maximum_bytes: int | None = None,
    ) -> np.ndarray:
        return self.covariance.materialize(maximum_bytes=maximum_bytes)

    def to_legacy(
        self,
        *,
        maximum_covariance_bytes: int | None = None,
    ) -> GaugeAwareBeliefResult:
        return GaugeAwareBeliefResult(
            inference_admissible=self.inference_admissible,
            reason=self.reason,
            state_coefficients=self.state_coefficients,
            gauge_delta=self.gauge_delta,
            shared_bias_coefficients=self.shared_bias_coefficients,
            view_bias_coefficients=self.view_bias_coefficients,
            anchor_bias_coefficients=self.anchor_bias_coefficients,
            posterior_covariance=self.materialize_posterior_covariance(
                maximum_bytes=maximum_covariance_bytes
            ),
            identifiable_state_transform=self.identifiable_state_transform,
            identifiable_fractions=self.identifiable_fractions,
            query_sensitivity_fractions=self.query_sensitivity_fractions,
            robust_weights=self.robust_weights,
            anchor_robust_weights=self.anchor_robust_weights,
            diagnostics=self.diagnostics,
            input_lineage=self.input_lineage,
        )

    def descriptor(self) -> Mapping[str, Any]:
        arrays = {
            name: _array_sha256(getattr(self, name))
            for name in (
                "state_coefficients",
                "gauge_delta",
                "shared_bias_coefficients",
                "view_bias_coefficients",
                "anchor_bias_coefficients",
                "identifiable_state_transform",
                "identifiable_fractions",
                "query_sensitivity_fractions",
                "robust_weights",
                "anchor_robust_weights",
            )
        }
        return frozen_finite_json_mapping(
            {
                "schema": TREE_BLOCK_GAUGE_AWARE_RESULT_SCHEMA,
                "schema_version": TREE_BLOCK_GAUGE_AWARE_RESULT_VERSION,
                "inference_admissible": self.inference_admissible,
                "reason": self.reason,
                "arrays": arrays,
                "covariance": dict(self.covariance.descriptor()),
                "diagnostics": plain_json(self.diagnostics),
                "input_lineage": plain_json(self.input_lineage),
            },
            name="tree-block gauge-aware result descriptor",
        )


def _prior_covariance(
    *,
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    state_prior: np.ndarray,
    state_design: np.ndarray,
    local_gauge: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    anchor_state: np.ndarray,
    anchor_bias: np.ndarray,
    config: PriorAwareGaugeConfigV1,
) -> TreeBlockPosteriorCovarianceV1:
    state_mapping = _positive_semidefinite_square_root(
        state_prior,
        "state prior covariance",
        eigenvalue_floor=config.prior_eigenvalue_floor,
    )
    layout = _GlobalLayout(
        state_count=state_mapping.shape[1],
        shared_count=shared.shape[2],
        view_count=view.shape[2],
        anchor_bias_count=anchor_bias.shape[2],
    )
    system = _build_tree_system(
        batch=batch,
        gauge=gauge,
        layout=layout,
        observation_weight=np.zeros(len(state_design)),
        state=np.einsum(
            "mcs,sr->mcr",
            state_design,
            state_mapping,
            optimize=True,
        ),
        local_gauge=local_gauge,
        shared=shared,
        view=view,
        target=np.zeros((len(state_design), 3), dtype=np.float64),
        anchor_weight=np.zeros(len(anchor_state)),
        anchor_state=np.einsum(
            "acs,sr->acr",
            anchor_state,
            state_mapping,
            optimize=True,
        ),
        anchor_bias=anchor_bias,
        anchor_target=np.zeros((len(anchor_state), 3), dtype=np.float64),
        config=config,
        state_prior_precision=np.eye(state_mapping.shape[1]),
    )
    factorization = system.eliminate_nodes(
        maximum_condition_number=config.maximum_condition_number
    ).factor_global(maximum_condition_number=config.maximum_condition_number)
    return TreeBlockPosteriorCovarianceV1(
        state_prior_covariance=state_prior,
        state_mapping=state_mapping,
        factorization=factorization,
        bias_count=layout.bias_count,
    )


def _fallback_result(
    *,
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    reason: str,
    diagnostics: Mapping[str, Any],
    covariance: TreeBlockPosteriorCovarianceV1,
) -> TreeBlockGaugeAwareBeliefResultV1:
    state_count = batch.state_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    anchor_count = (
        0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
    )
    return TreeBlockGaugeAwareBeliefResultV1(
        inference_admissible=False,
        reason=reason,
        state_coefficients=np.zeros(state_count),
        gauge_delta=np.zeros(gauge.gauge_parameter_count),
        shared_bias_coefficients=np.zeros(shared_count),
        view_bias_coefficients=np.zeros(view_count),
        anchor_bias_coefficients=np.zeros(anchor_bias_count),
        covariance=covariance,
        identifiable_state_transform=np.zeros((state_count, 0)),
        identifiable_fractions=np.zeros(0),
        query_sensitivity_fractions=np.zeros(0),
        robust_weights=np.zeros(len(batch.innovation_m)),
        anchor_robust_weights=np.zeros(anchor_count),
        diagnostics={
            **diagnostics,
            "result_covariance_representation": covariance.representation,
            "result_dense_covariance_materialized": False,
            "result_covariance_stored_nbytes": covariance.stored_nbytes,
            "result_estimated_dense_covariance_bytes": (
                covariance.estimated_dense_covariance_bytes
            ),
        },
        input_lineage=batch.metadata or {},
    )


def update_tree_block_sparse_prior_aware_gauge_belief(
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> TreeBlockGaugeAwareBeliefResultV1:
    """Infer state and gauges without any dense gauge-sized matrix."""

    if not isinstance(batch, GaugeAwareObservationBatch):
        raise TypeError("batch must be a GaugeAwareObservationBatch")
    if not isinstance(gauge, TreeSparseGaugeDesignV1):
        raise TypeError("gauge must be a TreeSparseGaugeDesignV1")
    _require(
        batch.gauge_jacobian.shape[2] == 0
        and batch.gauge_prior_covariance.shape == (0, 0),
        "batch must leave gauge ownership to TreeSparseGaugeDesignV1",
    )
    _require(
        gauge.observation_count == len(batch.innovation_m),
        "tree gauge row count differs from the observation batch",
    )
    cfg = config or PriorAwareGaugeConfigV1()
    if batch.prior_nominal_probability is None or batch.composite_weight is None:
        raise ValueError("validated observation mixture metadata is missing")
    observation_nominal = np.asarray(batch.prior_nominal_probability)
    observation_composite = np.asarray(batch.composite_weight)

    state_count = batch.state_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    nuisance_layout = _NuisanceLayout(
        gauge_parameter_count=gauge.gauge_parameter_count,
        shared_count=shared_count,
        view_count=view_count,
        anchor_bias_count=anchor_bias_count,
    )

    anchor_count = (
        0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
    )
    if anchor_count:
        if (
            batch.anchor_innovation_m is None
            or batch.anchor_covariance_m2 is None
            or batch.anchor_state_jacobian is None
            or batch.anchor_correlation_group_ids is None
            or batch.anchor_prior_reliability is None
            or batch.anchor_prior_nominal_probability is None
            or batch.anchor_composite_weight is None
        ):
            raise ValueError("validated anchor mixture metadata is missing")
        anchor_innovation = np.asarray(batch.anchor_innovation_m)
        anchor_covariance = np.asarray(batch.anchor_covariance_m2)
        anchor_state = np.asarray(batch.anchor_state_jacobian)
        anchor_groups_input = batch.anchor_correlation_group_ids
        anchor_reliability = np.asarray(batch.anchor_prior_reliability)
        anchor_nominal = np.asarray(batch.anchor_prior_nominal_probability)
        anchor_composite = np.asarray(batch.anchor_composite_weight)
        anchor_bias = (
            np.zeros((anchor_count, 3, anchor_bias_count))
            if batch.anchor_bias_jacobian is None
            else np.asarray(batch.anchor_bias_jacobian)
        )
    else:
        anchor_innovation = np.zeros((0, 3))
        anchor_covariance = np.zeros((0, 3, 3))
        anchor_state = np.zeros((0, 3, state_count))
        anchor_groups_input = ()
        anchor_reliability = np.zeros(0)
        anchor_nominal = np.zeros(0)
        anchor_composite = np.zeros(0)
        anchor_bias = np.zeros((0, 3, anchor_bias_count))

    (
        target,
        state_white,
        local_gauge_white,
        shared_white,
        view_white,
        whiteners,
    ) = _whiten_sparse_observations(batch, gauge)
    if anchor_count:
        (
            anchor_target,
            (anchor_state_white, anchor_bias_white),
            anchor_whiteners,
        ) = _whiten(
            anchor_innovation,
            anchor_covariance,
            (anchor_state, anchor_bias),
            name="anchor",
        )
    else:
        anchor_target = np.zeros((0, 3))
        anchor_state_white = anchor_state
        anchor_bias_white = anchor_bias
        anchor_whiteners = np.zeros((0, 3, 3))

    association_probability = np.asarray(batch.association_probability)
    observation_row_weight = batch.prior_reliability * association_probability
    (
        observation_groups,
        observation_indices,
        observation_base,
        observation_prior,
        observation_group_power,
    ) = _group_layout(
        batch.correlation_group_ids,
        observation_row_weight,
        observation_nominal,
        observation_composite,
        cfg.effective_samples_per_correlation_group,
        composite_weight_mode=batch.composite_weight_mode,
    )
    if anchor_count:
        (
            anchor_groups,
            anchor_indices,
            anchor_base,
            anchor_prior,
            anchor_group_power,
        ) = _group_layout(
            anchor_groups_input,
            anchor_reliability,
            anchor_nominal,
            anchor_composite,
            cfg.effective_samples_per_anchor_correlation_group,
            composite_weight_mode=batch.anchor_composite_weight_mode,
        )
    else:
        anchor_groups, anchor_indices = (), ()
        anchor_base = np.zeros(0)
        anchor_prior = np.zeros(0)
        anchor_group_power = np.zeros(0)

    state_prior = (
        np.eye(state_count) * cfg.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2)
    )
    fallback_covariance = _prior_covariance(
        batch=batch,
        gauge=gauge,
        state_prior=state_prior,
        state_design=state_white,
        local_gauge=local_gauge_white,
        shared=shared_white,
        view=view_white,
        anchor_state=anchor_state_white,
        anchor_bias=anchor_bias_white,
        config=cfg,
    )

    expected_observation = (
        observation_prior
        + (1.0 - observation_prior) / cfg.outlier_covariance_multiplier
    )
    expected_anchor = (
        anchor_prior + (1.0 - anchor_prior) / cfg.outlier_covariance_multiplier
    )
    identification_weight = observation_base.copy()
    for position, selected in enumerate(observation_indices):
        identification_weight[selected] *= expected_observation[position]
    anchor_identification_weight = anchor_base.copy()
    for position, selected in enumerate(anchor_indices):
        anchor_identification_weight[selected] *= expected_anchor[position]

    diagnostics: dict[str, Any] = {
        "identifiability_mode": "prior-aware-tree-block-schur-v1",
        "robust_likelihood": "grouped nominal/outlier Student-t mixture",
        "robust_likelihood_objective": (
            "exact-group-mixture-gradient"
            if cfg.minimum_robust_precision == 0.0
            else "precision-floored-group-mixture-approximation"
        ),
        "posterior_covariance_kind": ("working-gauss-newton-irls-tree-factorization"),
        "minimum_robust_precision": cfg.minimum_robust_precision,
        "prior_nominal_probability_used_inside_mixture": True,
        "association_probability_used_as_reliability": False,
        "association_probability_used_as_row_power": True,
        "row_reliability_semantics": ("conditional-covariance-precision-scaling"),
        "row_association_semantics": "generalized-Bayes-row-power-v1",
        "group_composite_weight_semantics": ("generalized-Bayes likelihood power"),
        "observation_composite_weight_mode": batch.composite_weight_mode,
        "anchor_composite_weight_mode": batch.anchor_composite_weight_mode,
        "tree_block_sparse_gauge_solver_version": (
            TREE_BLOCK_SPARSE_GAUGE_SOLVER_VERSION
        ),
        "gauge_prior_representation": ("tree-transition-innovation-factor-v1"),
        "dense_gauge_prior_covariance_materialized": False,
        "tree_prior_information_matrix_materialized": False,
        "dense_nuisance_normal_matrix_materialized": False,
        "dense_joint_normal_matrix_materialized": False,
        "native_sparse_gauge_design_materialized": False,
        "native_sparse_gauge_block_size": gauge.block_size,
        "native_sparse_gauge_count": gauge.gauge_count,
        "native_sparse_gauge_parameter_count": gauge.gauge_parameter_count,
        "dense_gauge_design_avoided_bytes": (gauge.equivalent_dense_design_bytes),
        "dense_gauge_prior_avoided_bytes": (gauge.dense_gauge_prior_avoided_bytes),
        "tree_factor_storage_nbytes": gauge.tree_factor_storage_nbytes,
        "exact_reduced_mixture_hessian_materialized": False,
    }
    try:
        (
            state_mapping,
            identifiable,
            query_fraction,
            basis_diagnostics,
        ) = _prior_aware_tree_basis(
            batch=batch,
            gauge=gauge,
            observation_weight=identification_weight,
            state=state_white,
            local_gauge=local_gauge_white,
            shared=shared_white,
            view=view_white,
            anchor_weight=anchor_identification_weight,
            anchor_state=anchor_state_white,
            anchor_bias=anchor_bias_white,
            query=batch.query_state_jacobian,
            state_prior=state_prior,
            config=cfg,
        )
    except np.linalg.LinAlgError:
        return _fallback_result(
            batch=batch,
            gauge=gauge,
            reason="ill-conditioned-identifiability-system",
            diagnostics=diagnostics,
            covariance=fallback_covariance,
        )
    diagnostics.update(basis_diagnostics)
    if not state_mapping.shape[1]:
        return _fallback_result(
            batch=batch,
            gauge=gauge,
            reason="no-identifiable-query-state",
            diagnostics=diagnostics,
            covariance=fallback_covariance,
        )

    retained = state_mapping.shape[1]
    state_reduced_white = np.einsum(
        "mcs,sr->mcr",
        state_white,
        state_mapping,
        optimize=True,
    )
    state_reduced_raw = np.einsum(
        "mcs,sr->mcr",
        batch.state_jacobian,
        state_mapping,
        optimize=True,
    )
    anchor_state_reduced_white = np.einsum(
        "acs,sr->acr",
        anchor_state_white,
        state_mapping,
        optimize=True,
    )
    anchor_state_reduced_raw = np.einsum(
        "acs,sr->acr",
        anchor_state,
        state_mapping,
        optimize=True,
    )
    layout = _GlobalLayout(
        state_count=retained,
        shared_count=shared_count,
        view_count=view_count,
        anchor_bias_count=anchor_bias_count,
    )
    global_solution: np.ndarray = np.zeros(layout.global_count, dtype=np.float64)
    node_solution = np.zeros(
        (gauge.gauge_count, gauge.block_size),
        dtype=np.float64,
    )
    observation_precision = expected_observation.copy()
    anchor_precision = expected_anchor.copy()
    observation_responsibility = np.clip(
        observation_prior,
        cfg.probability_floor,
        1.0 - cfg.probability_floor,
    )
    anchor_responsibility = np.clip(
        anchor_prior,
        cfg.probability_floor,
        1.0 - cfg.probability_floor,
    )
    observation_floor_active: np.ndarray = np.zeros(
        len(observation_groups),
        dtype=bool,
    )
    anchor_floor_active: np.ndarray = np.zeros(len(anchor_groups), dtype=bool)

    def current_legacy_solution() -> np.ndarray:
        return _legacy_order_solution(
            global_solution,
            node_solution,
            layout=layout,
        )

    def refresh_mixture_statistics() -> None:
        legacy_solution = current_legacy_solution()
        residual = batch.innovation_m - _observation_prediction(
            state_reduced_raw,
            gauge.local_gauge_jacobian,
            gauge.gauge_indices,
            batch.shared_bias_jacobian,
            batch.view_bias_jacobian,
            legacy_solution,
            retained=retained,
            layout=nuisance_layout,
        )
        white_residual = np.einsum(
            "mij,mj->mi",
            whiteners,
            residual,
            optimize=True,
        )
        for position, selected in enumerate(observation_indices):
            active = selected[observation_row_weight[selected] > 0.0]
            if not len(active):
                observation_precision[position] = 0.0
                observation_responsibility[position] = float(
                    np.clip(
                        observation_prior[position],
                        cfg.probability_floor,
                        1.0 - cfg.probability_floor,
                    )
                )
                observation_floor_active[position] = False
                continue
            squared_mahalanobis = float(
                np.sum(
                    observation_row_weight[active]
                    * np.sum(np.square(white_residual[active]), axis=1)
                )
            )
            statistics = _student_t_mixture_statistics(
                squared_mahalanobis,
                3 * len(active),
                float(observation_prior[position]),
                cfg,
            )
            observation_precision[position] = statistics.expected_precision
            observation_responsibility[position] = (
                statistics.posterior_nominal_probability
            )
            observation_floor_active[position] = statistics.precision_floor_active

        if not anchor_count:
            return
        anchor_residual = anchor_innovation - _anchor_prediction(
            anchor_state_reduced_raw,
            anchor_bias,
            legacy_solution,
            retained=retained,
            layout=nuisance_layout,
        )
        white_anchor_residual = np.einsum(
            "aij,aj->ai",
            anchor_whiteners,
            anchor_residual,
            optimize=True,
        )
        for position, selected in enumerate(anchor_indices):
            active = selected[anchor_reliability[selected] > 0.0]
            if not len(active):
                anchor_precision[position] = 0.0
                anchor_responsibility[position] = float(
                    np.clip(
                        anchor_prior[position],
                        cfg.probability_floor,
                        1.0 - cfg.probability_floor,
                    )
                )
                anchor_floor_active[position] = False
                continue
            squared_mahalanobis = float(
                np.sum(
                    anchor_reliability[active]
                    * np.sum(np.square(white_anchor_residual[active]), axis=1)
                )
            )
            statistics = _student_t_mixture_statistics(
                squared_mahalanobis,
                3 * len(active),
                float(anchor_prior[position]),
                cfg,
            )
            anchor_precision[position] = statistics.expected_precision
            anchor_responsibility[position] = statistics.posterior_nominal_probability
            anchor_floor_active[position] = statistics.precision_floor_active

    def build_system() -> TreeBlockNormalSystemV1:
        row_precision = np.zeros(len(observation_base))
        for position, selected in enumerate(observation_indices):
            row_precision[selected] = observation_precision[position]
        anchor_row_precision = np.zeros(len(anchor_base))
        for position, selected in enumerate(anchor_indices):
            anchor_row_precision[selected] = anchor_precision[position]
        return _build_tree_system(
            batch=batch,
            gauge=gauge,
            layout=layout,
            observation_weight=observation_base * row_precision,
            state=state_reduced_white,
            local_gauge=local_gauge_white,
            shared=shared_white,
            view=view_white,
            target=target,
            anchor_weight=anchor_base * anchor_row_precision,
            anchor_state=anchor_state_reduced_white,
            anchor_bias=anchor_bias_white,
            anchor_target=anchor_target,
            config=cfg,
            state_prior_precision=np.eye(retained),
        )

    refresh_mixture_statistics()
    fixed_point_converged = False
    iteration_count = 0
    solution_delta = float("inf")
    stationarity_norm = float("inf")
    final_system: TreeBlockNormalSystemV1 | None = None
    final_factorization: TreeBlockFactorizationV1 | None = None
    for iteration in range(cfg.maximum_iterations):
        iteration_count = iteration + 1
        system = build_system()
        try:
            elimination = system.eliminate_nodes(
                maximum_condition_number=cfg.maximum_condition_number
            )
            factorization = elimination.factor_global(
                maximum_condition_number=cfg.maximum_condition_number
            )
        except np.linalg.LinAlgError:
            return _fallback_result(
                batch=batch,
                gauge=gauge,
                reason="ill-conditioned-posterior",
                diagnostics=diagnostics,
                covariance=fallback_covariance,
            )
        previous = np.concatenate((global_solution, node_solution.reshape(-1)))
        global_solution, node_solution = factorization.solve(
            system.global_right,
            system.node_right,
        )
        current = np.concatenate((global_solution, node_solution.reshape(-1)))
        solution_delta = float(np.linalg.norm(current - previous))
        refresh_mixture_statistics()

        final_system = build_system()
        try:
            final_elimination = final_system.eliminate_nodes(
                maximum_condition_number=cfg.maximum_condition_number
            )
            final_factorization = final_elimination.factor_global(
                maximum_condition_number=cfg.maximum_condition_number
            )
        except np.linalg.LinAlgError:
            return _fallback_result(
                batch=batch,
                gauge=gauge,
                reason="ill-conditioned-final-posterior",
                diagnostics=diagnostics,
                covariance=fallback_covariance,
            )
        global_residual, node_residual = final_system.residual(
            global_solution,
            node_solution,
        )
        stationarity_norm = float(
            np.linalg.norm(np.concatenate((global_residual, node_residual.reshape(-1))))
        )
        solution_scale = 1.0 + float(np.linalg.norm(current))
        right_scale = 1.0 + float(
            np.linalg.norm(
                np.concatenate(
                    (
                        final_system.global_right,
                        final_system.node_right.reshape(-1),
                    )
                )
            )
        )
        if (
            solution_delta <= cfg.convergence_tolerance * solution_scale
            and stationarity_norm <= cfg.convergence_tolerance * right_scale
        ):
            fixed_point_converged = True
            break

    if final_system is None or final_factorization is None:
        raise RuntimeError("tree-block solver produced no final system")

    state_coefficients = state_mapping @ global_solution[layout.state_slice]
    query_update = np.einsum(
        "qcs,s->qc",
        batch.query_state_jacobian,
        state_coefficients,
        optimize=True,
    )
    maximum_update = float(np.max(np.linalg.norm(query_update, axis=1), initial=0.0))
    relative_limit = (
        cfg.maximum_update_to_physical_response_ratio * batch.physical_response_scale_m
    )
    update_limit = min(cfg.maximum_state_update_m, relative_limit)
    full_solution = np.concatenate(
        (
            state_coefficients,
            node_solution.reshape(-1),
            global_solution[layout.shared_slice],
            global_solution[layout.view_slice],
            global_solution[layout.anchor_bias_slice],
        )
    )
    diagnostics.update(
        {
            "iterations": iteration_count,
            "mixture_fixed_point_converged": fixed_point_converged,
            "mixture_solution_delta": solution_delta,
            "mixture_stationarity_norm": stationarity_norm,
            "posterior_solver": "tree-block-leaf-schur-cholesky-v1",
            "tree_block_global_schur_dimension": layout.global_count,
            "tree_block_joint_dimension": final_system.dimension,
            "tree_block_system_stored_nbytes": final_system.stored_nbytes,
            "tree_block_factorization_stored_nbytes": (
                final_factorization.stored_nbytes
            ),
            "dense_joint_normal_avoided_bytes": (
                final_system.estimated_dense_precision_bytes
            ),
            "maximum_eliminated_node_condition_number": (
                final_factorization.maximum_node_condition_number
            ),
            "global_schur_condition_number": (
                final_factorization.global_condition_number
            ),
            "maximum_query_state_update_m": maximum_update,
            "active_state_update_limit_m": update_limit,
            "observation_group_ids": list(observation_groups),
            "observation_group_power": observation_group_power.tolist(),
            "observation_group_posterior_nominal_probability": (
                observation_responsibility.tolist()
            ),
            "observation_group_precision_floor_active": (
                observation_floor_active.tolist()
            ),
            "anchor_group_ids": list(anchor_groups),
            "anchor_group_power": anchor_group_power.tolist(),
            "anchor_group_posterior_nominal_probability": (
                anchor_responsibility.tolist()
            ),
            "anchor_group_precision_floor_active": (anchor_floor_active.tolist()),
        }
    )
    if not np.all(np.isfinite(full_solution)) or maximum_update > update_limit:
        return _fallback_result(
            batch=batch,
            gauge=gauge,
            reason="implausible-state-update",
            diagnostics=diagnostics,
            covariance=fallback_covariance,
        )

    covariance = TreeBlockPosteriorCovarianceV1(
        state_prior_covariance=state_prior,
        state_mapping=state_mapping,
        factorization=final_factorization,
        bias_count=layout.bias_count,
    )
    ordinary_robust = np.zeros(len(observation_base))
    for position, selected in enumerate(observation_indices):
        ordinary_robust[selected] = observation_precision[position]
    anchor_robust = np.zeros(len(anchor_base))
    for position, selected in enumerate(anchor_indices):
        anchor_robust[selected] = anchor_precision[position]
    return TreeBlockGaugeAwareBeliefResultV1(
        inference_admissible=True,
        reason="inference-admissible",
        state_coefficients=state_coefficients,
        gauge_delta=node_solution.reshape(-1),
        shared_bias_coefficients=global_solution[layout.shared_slice],
        view_bias_coefficients=global_solution[layout.view_slice],
        anchor_bias_coefficients=global_solution[layout.anchor_bias_slice],
        covariance=covariance,
        identifiable_state_transform=state_mapping,
        identifiable_fractions=identifiable,
        query_sensitivity_fractions=query_fraction,
        robust_weights=ordinary_robust,
        anchor_robust_weights=anchor_robust,
        diagnostics={
            **diagnostics,
            "result_covariance_representation": covariance.representation,
            "result_dense_covariance_materialized": False,
            "result_covariance_stored_nbytes": covariance.stored_nbytes,
            "result_estimated_dense_covariance_bytes": (
                covariance.estimated_dense_covariance_bytes
            ),
        },
        input_lineage=batch.metadata or {},
    )


__all__ = [
    "TREE_BLOCK_GAUGE_AWARE_RESULT_SCHEMA",
    "TREE_BLOCK_GAUGE_AWARE_RESULT_VERSION",
    "TREE_BLOCK_POSTERIOR_COVARIANCE_SCHEMA",
    "TREE_BLOCK_POSTERIOR_COVARIANCE_VERSION",
    "TREE_BLOCK_SPARSE_GAUGE_SOLVER_SCHEMA",
    "TREE_BLOCK_SPARSE_GAUGE_SOLVER_VERSION",
    "TreeBlockGaugeAwareBeliefResultV1",
    "TreeBlockPosteriorCovarianceV1",
    "update_tree_block_sparse_prior_aware_gauge_belief",
]
