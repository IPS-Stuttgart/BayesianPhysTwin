"""Native sparse prior-aware inference for block-local gauge Jacobians."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    _block_diagonal,
    _finite_array,
    _positive_definite_whitener,
    _positive_semidefinite_square_root,
    _readonly,
    _regularized_precision,
    _require,
)
from ._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _full_covariance,
    _group_layout,
    _solve_spd_system,
    _spd_covariance,
    _student_t_mixture_statistics,
    _whiten,
)

SPARSE_PRIOR_AWARE_GAUGE_SOLVER_VERSION = 1


@dataclass(frozen=True, slots=True)
class SparseGaugeDesignV1:
    """One local gauge block per observation and a complete joint prior."""

    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    gauge_prior_covariance: np.ndarray
    gauge_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        local = _finite_array(
            self.local_gauge_jacobian,
            "local_gauge_jacobian",
            3,
        )
        _require(
            local.shape[1] == 3 and local.shape[2] >= 1,
            "local_gauge_jacobian must have shape (M, 3, G) with G >= 1",
        )
        raw_indices = np.asarray(self.gauge_indices)
        _require(
            raw_indices.ndim == 1
            and np.issubdtype(raw_indices.dtype, np.integer)
            and raw_indices.dtype.kind != "b",
            "gauge_indices must be an integer vector",
        )
        indices = np.asarray(raw_indices, dtype=np.int64)
        _require(
            indices.shape == (len(local),),
            "gauge_indices must contain one value per observation",
        )
        _require(
            isinstance(self.gauge_ids, tuple)
            and all(isinstance(value, str) for value in self.gauge_ids),
            "gauge_ids must be a tuple of strings",
        )
        gauge_ids = tuple(self.gauge_ids)
        _require(bool(gauge_ids) and all(gauge_ids), "gauge_ids must be nonempty")
        _require(
            len(set(gauge_ids)) == len(gauge_ids),
            "gauge_ids must be unique",
        )
        gauge_count = len(gauge_ids)
        block_size = local.shape[2]
        gauge_parameter_count = gauge_count * block_size
        prior = _finite_array(
            self.gauge_prior_covariance,
            "gauge_prior_covariance",
            2,
        )
        _require(
            prior.shape == (gauge_parameter_count, gauge_parameter_count),
            "gauge prior covariance has changed shape",
        )
        _regularized_precision(
            prior,
            "gauge prior covariance",
            eigenvalue_floor=1e-12,
        )
        _require(
            np.all((indices >= 0) & (indices < gauge_count)),
            "gauge_indices reference an unknown gauge",
        )
        object.__setattr__(self, "local_gauge_jacobian", _readonly(local))
        object.__setattr__(self, "gauge_indices", _readonly(indices, dtype=np.int64))
        object.__setattr__(self, "gauge_prior_covariance", _readonly(prior))
        object.__setattr__(self, "gauge_ids", gauge_ids)

    @property
    def observation_count(self) -> int:
        return len(self.local_gauge_jacobian)

    @property
    def block_size(self) -> int:
        return self.local_gauge_jacobian.shape[2]

    @property
    def gauge_count(self) -> int:
        return len(self.gauge_ids)

    @property
    def gauge_parameter_count(self) -> int:
        return self.gauge_count * self.block_size

    @property
    def equivalent_dense_design_bytes(self) -> int:
        return (
            self.observation_count
            * 3
            * self.gauge_parameter_count
            * np.dtype(np.float64).itemsize
        )


@dataclass(frozen=True, slots=True)
class TreeSparseGaugeDesignV1:
    """Block-local row design with a causal transition/innovation gauge prior."""

    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    parent_indices: np.ndarray
    transition_matrices: np.ndarray
    innovation_scale_tril: np.ndarray
    gauge_ids: tuple[str, ...]
    prior_id: str

    def __post_init__(self) -> None:
        local = _finite_array(
            self.local_gauge_jacobian,
            "local_gauge_jacobian",
            3,
        )
        _require(
            local.shape[1] == 3 and local.shape[2] >= 1,
            "local_gauge_jacobian must have shape (M, 3, G) with G >= 1",
        )
        raw_indices = np.asarray(self.gauge_indices)
        _require(
            raw_indices.ndim == 1
            and np.issubdtype(raw_indices.dtype, np.integer)
            and raw_indices.dtype.kind != "b",
            "gauge_indices must be an integer vector",
        )
        indices = np.asarray(raw_indices, dtype=np.int64)
        _require(
            indices.shape == (len(local),),
            "gauge_indices must contain one value per observation",
        )
        raw_parents = np.asarray(self.parent_indices)
        _require(
            raw_parents.ndim == 1
            and np.issubdtype(raw_parents.dtype, np.integer)
            and raw_parents.dtype.kind != "b",
            "parent_indices must be an integer vector",
        )
        parents = np.asarray(raw_parents, dtype=np.int64)
        transitions = _finite_array(
            self.transition_matrices,
            "transition_matrices",
            3,
        )
        scales = _finite_array(
            self.innovation_scale_tril,
            "innovation_scale_tril",
            3,
        )
        _require(
            isinstance(self.gauge_ids, tuple)
            and all(isinstance(value, str) for value in self.gauge_ids),
            "gauge_ids must be a tuple of strings",
        )
        gauge_ids = tuple(self.gauge_ids)
        _require(bool(gauge_ids) and all(gauge_ids), "gauge_ids must be nonempty")
        _require(
            len(set(gauge_ids)) == len(gauge_ids),
            "gauge_ids must be unique",
        )
        gauge_count = len(gauge_ids)
        block_size = local.shape[2]
        _require(
            parents.shape == (gauge_count,),
            "parent_indices must contain one value per gauge",
        )
        _require(
            transitions.shape == (gauge_count, block_size, block_size),
            "transition_matrices shape changed",
        )
        _require(
            scales.shape == (gauge_count, block_size, block_size),
            "innovation_scale_tril shape changed",
        )
        _require(parents[0] == -1, "the first gauge must be the causal-tree root")
        for index in range(1, gauge_count):
            _require(
                0 <= int(parents[index]) < index,
                "each non-root gauge parent must precede the child",
            )
        _require(
            np.all((indices >= 0) & (indices < gauge_count)),
            "gauge_indices reference an unknown gauge",
        )
        _require(
            np.allclose(scales, np.tril(scales), atol=1e-14, rtol=0.0),
            "innovation_scale_tril must be lower triangular",
        )
        _require(
            np.all(np.diagonal(scales, axis1=1, axis2=2) > 0.0),
            "innovation_scale_tril must have positive diagonal",
        )
        if not isinstance(self.prior_id, str):
            raise TypeError("prior_id must be a string")
        _require(
            len(self.prior_id) == 64
            and all(character in "0123456789abcdef" for character in self.prior_id),
            "prior_id must be a lowercase SHA-256 digest",
        )
        object.__setattr__(self, "local_gauge_jacobian", _readonly(local))
        object.__setattr__(self, "gauge_indices", _readonly(indices, dtype=np.int64))
        object.__setattr__(self, "parent_indices", _readonly(parents, dtype=np.int64))
        object.__setattr__(self, "transition_matrices", _readonly(transitions))
        object.__setattr__(self, "innovation_scale_tril", _readonly(scales))
        object.__setattr__(self, "gauge_ids", gauge_ids)

    @property
    def observation_count(self) -> int:
        return len(self.local_gauge_jacobian)

    @property
    def block_size(self) -> int:
        return self.local_gauge_jacobian.shape[2]

    @property
    def gauge_count(self) -> int:
        return len(self.gauge_ids)

    @property
    def gauge_parameter_count(self) -> int:
        return self.gauge_count * self.block_size

    @property
    def equivalent_dense_design_bytes(self) -> int:
        return (
            self.observation_count
            * 3
            * self.gauge_parameter_count
            * np.dtype(np.float64).itemsize
        )

    @property
    def dense_gauge_prior_avoided_bytes(self) -> int:
        return self.gauge_parameter_count**2 * np.dtype(np.float64).itemsize

    @property
    def tree_factor_storage_nbytes(self) -> int:
        return int(
            self.parent_indices.nbytes
            + self.transition_matrices.nbytes
            + self.innovation_scale_tril.nbytes
        )

    def prior_information_matrix(self) -> np.ndarray:
        """Assemble exact tree precision without constructing prior covariance."""

        dimension = self.gauge_parameter_count
        block_size = self.block_size
        information: np.ndarray = np.zeros((dimension, dimension), dtype=np.float64)
        identity = np.eye(block_size, dtype=np.float64)
        for index in range(self.gauge_count):
            inverse_scale = np.linalg.solve(
                self.innovation_scale_tril[index],
                identity,
            )
            innovation_precision = inverse_scale.T @ inverse_scale
            child = slice(index * block_size, (index + 1) * block_size)
            information[child, child] += innovation_precision
            parent_index = int(self.parent_indices[index])
            if parent_index < 0:
                continue
            parent = slice(
                parent_index * block_size,
                (parent_index + 1) * block_size,
            )
            transition = self.transition_matrices[index]
            information[child, parent] -= innovation_precision @ transition
            information[parent, child] -= transition.T @ innovation_precision
            information[parent, parent] += (
                transition.T @ innovation_precision @ transition
            )
        return 0.5 * (information + information.T)


GaugeDesignV1 = SparseGaugeDesignV1 | TreeSparseGaugeDesignV1


class _LazyPriorCovariance:
    """Materialize a dense prior only when a rejected result requires it."""

    def __init__(
        self,
        state_covariance: np.ndarray,
        nuisance_precision: np.ndarray,
        nuisance_covariance: np.ndarray | None,
    ) -> None:
        self._state_covariance = state_covariance
        self._nuisance_precision = nuisance_precision
        self._nuisance_covariance = nuisance_covariance

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        nuisance_covariance = self._nuisance_covariance
        if nuisance_covariance is None:
            identity = np.eye(len(self._nuisance_precision), dtype=np.float64)
            nuisance_covariance = np.linalg.solve(
                self._nuisance_precision,
                identity,
            )
            nuisance_covariance = 0.5 * (nuisance_covariance + nuisance_covariance.T)
        result = _block_diagonal([self._state_covariance, nuisance_covariance])
        return result if dtype is None else np.asarray(result, dtype=dtype)


@dataclass(frozen=True, slots=True)
class _NuisanceLayout:
    gauge_parameter_count: int
    shared_count: int
    view_count: int
    anchor_bias_count: int

    @property
    def shared_slice(self) -> slice:
        return slice(
            self.gauge_parameter_count,
            self.gauge_parameter_count + self.shared_count,
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
    def nuisance_count(self) -> int:
        return self.anchor_bias_slice.stop


def _sparse_fallback_result(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    reason: str,
    diagnostics: Mapping[str, Any],
    *,
    prior_covariance: np.ndarray | _LazyPriorCovariance,
) -> GaugeAwareBeliefResult:
    state_count = batch.state_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    anchor_count = (
        0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
    )
    return GaugeAwareBeliefResult(
        inference_admissible=False,
        reason=reason,
        state_coefficients=np.zeros(state_count),
        gauge_delta=np.zeros(gauge.gauge_parameter_count),
        shared_bias_coefficients=np.zeros(shared_count),
        view_bias_coefficients=np.zeros(view_count),
        anchor_bias_coefficients=np.zeros(anchor_bias_count),
        posterior_covariance=np.asarray(prior_covariance, dtype=np.float64),
        identifiable_state_transform=np.zeros((state_count, 0)),
        identifiable_fractions=np.zeros(0),
        query_sensitivity_fractions=np.zeros(0),
        robust_weights=np.zeros(len(batch.innovation_m)),
        anchor_robust_weights=np.zeros(anchor_count),
        diagnostics=diagnostics,
        input_lineage=batch.metadata or {},
    )


def _prior_covariances(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    config: PriorAwareGaugeConfigV1,
) -> tuple[np.ndarray, np.ndarray, _LazyPriorCovariance]:
    state_count = batch.state_jacobian.shape[2]
    state = (
        np.eye(state_count) * config.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2)
    )
    dense_covariance_blocks: list[np.ndarray] | None
    if isinstance(gauge, TreeSparseGaugeDesignV1):
        gauge_precision = gauge.prior_information_matrix()
        dense_covariance_blocks = None
    else:
        gauge_covariance = np.asarray(gauge.gauge_prior_covariance)
        gauge_precision = _regularized_precision(
            gauge_covariance,
            "gauge prior covariance",
            eigenvalue_floor=config.prior_eigenvalue_floor,
        )
        dense_covariance_blocks = [gauge_covariance]
    precision_blocks = [gauge_precision]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    if shared_count:
        shared_covariance = np.eye(shared_count) * config.shared_bias_prior_std_m**2
        precision_blocks.append(
            np.eye(shared_count) / config.shared_bias_prior_std_m**2
        )
        if dense_covariance_blocks is not None:
            dense_covariance_blocks.append(shared_covariance)
    if view_count:
        view_covariance = np.eye(view_count) * config.view_bias_prior_std_m**2
        precision_blocks.append(np.eye(view_count) / config.view_bias_prior_std_m**2)
        if dense_covariance_blocks is not None:
            dense_covariance_blocks.append(view_covariance)
    if batch.anchor_bias_prior_covariance is not None:
        anchor_covariance = np.asarray(batch.anchor_bias_prior_covariance)
        precision_blocks.append(
            _regularized_precision(
                anchor_covariance,
                "anchor bias prior covariance",
                eigenvalue_floor=config.prior_eigenvalue_floor,
            )
        )
        if dense_covariance_blocks is not None:
            dense_covariance_blocks.append(anchor_covariance)
    nuisance_precision = _block_diagonal(precision_blocks)
    nuisance_covariance = (
        None
        if dense_covariance_blocks is None
        else _block_diagonal(dense_covariance_blocks)
    )
    return (
        state,
        nuisance_precision,
        _LazyPriorCovariance(state, nuisance_precision, nuisance_covariance),
    )


def _whiten_sparse_observations(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    count = len(batch.innovation_m)
    target: np.ndarray = np.empty((count, 3), dtype=np.float64)
    state = np.empty_like(batch.state_jacobian)
    local_gauge = np.empty_like(gauge.local_gauge_jacobian)
    shared = np.empty_like(batch.shared_bias_jacobian)
    view = np.empty_like(batch.view_bias_jacobian)
    whiteners = np.empty_like(batch.observation_covariance_m2)
    for index in range(count):
        whitener = _positive_definite_whitener(
            batch.observation_covariance_m2[index],
            f"observation covariance {index}",
        )
        whiteners[index] = whitener
        target[index] = whitener @ batch.innovation_m[index]
        state[index] = whitener @ batch.state_jacobian[index]
        local_gauge[index] = whitener @ gauge.local_gauge_jacobian[index]
        shared[index] = whitener @ batch.shared_bias_jacobian[index]
        view[index] = whitener @ batch.view_bias_jacobian[index]
    return target, state, local_gauge, shared, view, whiteners


def _information_blocks(
    weight: np.ndarray,
    state: np.ndarray,
    local_gauge: np.ndarray,
    gauge_indices: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    *,
    gauge_count: int,
    layout: _NuisanceLayout,
    anchor_weight: np.ndarray,
    anchor_state: np.ndarray,
    anchor_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_count = state.shape[2]
    known = np.einsum(
        "m,mci,mcj->ij",
        weight,
        state,
        state,
        optimize=True,
    )
    cross = np.zeros((state_count, layout.nuisance_count), dtype=np.float64)
    nuisance: np.ndarray = np.zeros(
        (layout.nuisance_count, layout.nuisance_count),
        dtype=np.float64,
    )
    if layout.shared_count:
        cross[:, layout.shared_slice] = np.einsum(
            "m,mci,mcj->ij",
            weight,
            state,
            shared,
            optimize=True,
        )
        nuisance[layout.shared_slice, layout.shared_slice] = np.einsum(
            "m,mci,mcj->ij",
            weight,
            shared,
            shared,
            optimize=True,
        )
    if layout.view_count:
        cross[:, layout.view_slice] = np.einsum(
            "m,mci,mcj->ij",
            weight,
            state,
            view,
            optimize=True,
        )
        nuisance[layout.view_slice, layout.view_slice] = np.einsum(
            "m,mci,mcj->ij",
            weight,
            view,
            view,
            optimize=True,
        )
    if layout.shared_count and layout.view_count:
        shared_view = np.einsum(
            "m,mci,mcj->ij",
            weight,
            shared,
            view,
            optimize=True,
        )
        nuisance[layout.shared_slice, layout.view_slice] = shared_view
        nuisance[layout.view_slice, layout.shared_slice] = shared_view.T

    block_size = local_gauge.shape[2]
    for gauge_index in range(gauge_count):
        selected = gauge_indices == gauge_index
        if not np.any(selected):
            continue
        local_weight = weight[selected]
        local_state = state[selected]
        local_design = local_gauge[selected]
        gauge_slice = slice(
            gauge_index * block_size,
            (gauge_index + 1) * block_size,
        )
        cross[:, gauge_slice] = np.einsum(
            "m,mci,mcj->ij",
            local_weight,
            local_state,
            local_design,
            optimize=True,
        )
        nuisance[gauge_slice, gauge_slice] = np.einsum(
            "m,mci,mcj->ij",
            local_weight,
            local_design,
            local_design,
            optimize=True,
        )
        if layout.shared_count:
            gauge_shared = np.einsum(
                "m,mci,mcj->ij",
                local_weight,
                local_design,
                shared[selected],
                optimize=True,
            )
            nuisance[gauge_slice, layout.shared_slice] = gauge_shared
            nuisance[layout.shared_slice, gauge_slice] = gauge_shared.T
        if layout.view_count:
            gauge_view = np.einsum(
                "m,mci,mcj->ij",
                local_weight,
                local_design,
                view[selected],
                optimize=True,
            )
            nuisance[gauge_slice, layout.view_slice] = gauge_view
            nuisance[layout.view_slice, gauge_slice] = gauge_view.T

    if len(anchor_weight):
        known += np.einsum(
            "a,aci,acj->ij",
            anchor_weight,
            anchor_state,
            anchor_state,
            optimize=True,
        )
        if layout.anchor_bias_count:
            cross[:, layout.anchor_bias_slice] = np.einsum(
                "a,aci,acj->ij",
                anchor_weight,
                anchor_state,
                anchor_bias,
                optimize=True,
            )
            nuisance[layout.anchor_bias_slice, layout.anchor_bias_slice] = np.einsum(
                "a,aci,acj->ij",
                anchor_weight,
                anchor_bias,
                anchor_bias,
                optimize=True,
            )
    return known, cross, nuisance


def _prior_aware_basis_from_information(
    known: np.ndarray,
    cross: np.ndarray,
    nuisance_information_from_data: np.ndarray,
    state_prior: np.ndarray,
    nuisance_prior_precision: np.ndarray,
    query: np.ndarray,
    config: PriorAwareGaugeConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    state_square_root = _positive_semidefinite_square_root(
        state_prior,
        "state prior covariance",
        eigenvalue_floor=config.prior_eigenvalue_floor,
    )
    nuisance_information = nuisance_prior_precision + nuisance_information_from_data
    conditional = known - cross @ np.linalg.solve(nuisance_information, cross.T)
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
        else np.zeros((known.shape[0], 0), dtype=np.float64)
    )
    return (
        mapping,
        np.asarray(identifiable_fractions),
        np.asarray(query_fractions),
        {
            "maximum_conditional_information_eigenvalue": maximum_information,
            "maximum_query_sensitivity_norm": maximum_query,
        },
    )


def _right_blocks(
    weight: np.ndarray,
    state: np.ndarray,
    local_gauge: np.ndarray,
    gauge_indices: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    target: np.ndarray,
    *,
    gauge_count: int,
    layout: _NuisanceLayout,
    anchor_weight: np.ndarray,
    anchor_state: np.ndarray,
    anchor_bias: np.ndarray,
    anchor_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_right = np.einsum(
        "m,mci,mc->i",
        weight,
        state,
        target,
        optimize=True,
    )
    nuisance_right: np.ndarray = np.zeros(layout.nuisance_count, dtype=np.float64)
    block_size = local_gauge.shape[2]
    for gauge_index in range(gauge_count):
        selected = gauge_indices == gauge_index
        if not np.any(selected):
            continue
        gauge_slice = slice(
            gauge_index * block_size,
            (gauge_index + 1) * block_size,
        )
        nuisance_right[gauge_slice] = np.einsum(
            "m,mci,mc->i",
            weight[selected],
            local_gauge[selected],
            target[selected],
            optimize=True,
        )
    if layout.shared_count:
        nuisance_right[layout.shared_slice] = np.einsum(
            "m,mci,mc->i",
            weight,
            shared,
            target,
            optimize=True,
        )
    if layout.view_count:
        nuisance_right[layout.view_slice] = np.einsum(
            "m,mci,mc->i",
            weight,
            view,
            target,
            optimize=True,
        )
    if len(anchor_weight):
        state_right += np.einsum(
            "a,aci,ac->i",
            anchor_weight,
            anchor_state,
            anchor_target,
            optimize=True,
        )
        if layout.anchor_bias_count:
            nuisance_right[layout.anchor_bias_slice] = np.einsum(
                "a,aci,ac->i",
                anchor_weight,
                anchor_bias,
                anchor_target,
                optimize=True,
            )
    return state_right, nuisance_right


def _observation_prediction(
    state: np.ndarray,
    local_gauge: np.ndarray,
    gauge_indices: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    solution: np.ndarray,
    *,
    retained: int,
    layout: _NuisanceLayout,
) -> np.ndarray:
    prediction = np.einsum(
        "mcr,r->mc",
        state,
        solution[:retained],
        optimize=True,
    )
    nuisance = solution[retained:]
    block_size = local_gauge.shape[2]
    gauge_coefficients = nuisance[: layout.gauge_parameter_count].reshape(
        -1,
        block_size,
    )
    prediction += np.einsum(
        "mcg,mg->mc",
        local_gauge,
        gauge_coefficients[gauge_indices],
        optimize=True,
    )
    if layout.shared_count:
        prediction += np.einsum(
            "mcb,b->mc",
            shared,
            nuisance[layout.shared_slice],
            optimize=True,
        )
    if layout.view_count:
        prediction += np.einsum(
            "mcv,v->mc",
            view,
            nuisance[layout.view_slice],
            optimize=True,
        )
    return prediction


def _anchor_prediction(
    state: np.ndarray,
    bias: np.ndarray,
    solution: np.ndarray,
    *,
    retained: int,
    layout: _NuisanceLayout,
) -> np.ndarray:
    prediction = np.einsum(
        "acr,r->ac",
        state,
        solution[:retained],
        optimize=True,
    )
    if layout.anchor_bias_count:
        prediction += np.einsum(
            "acb,b->ac",
            bias,
            solution[retained:][layout.anchor_bias_slice],
            optimize=True,
        )
    return prediction


def _score_direction(
    selected: np.ndarray,
    reliability: np.ndarray,
    state: np.ndarray,
    local_gauge: np.ndarray,
    gauge_indices: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    residual: np.ndarray,
    *,
    gauge_count: int,
    layout: _NuisanceLayout,
) -> np.ndarray:
    retained = state.shape[2]
    direction = np.zeros(retained + layout.nuisance_count, dtype=np.float64)
    local_weight = reliability[selected]
    direction[:retained] = np.einsum(
        "m,mci,mc->i",
        local_weight,
        state[selected],
        residual[selected],
        optimize=True,
    )
    selected_gauge_indices = gauge_indices[selected]
    selected_local_gauge = local_gauge[selected]
    selected_residual = residual[selected]
    block_size = local_gauge.shape[2]
    for gauge_index in range(gauge_count):
        local_selected = selected_gauge_indices == gauge_index
        if not np.any(local_selected):
            continue
        gauge_slice = slice(
            retained + gauge_index * block_size,
            retained + (gauge_index + 1) * block_size,
        )
        direction[gauge_slice] = np.einsum(
            "m,mci,mc->i",
            local_weight[local_selected],
            selected_local_gauge[local_selected],
            selected_residual[local_selected],
            optimize=True,
        )
    if layout.shared_count:
        shared_slice = slice(
            retained + layout.shared_slice.start,
            retained + layout.shared_slice.stop,
        )
        direction[shared_slice] = np.einsum(
            "m,mci,mc->i",
            local_weight,
            shared[selected],
            selected_residual,
            optimize=True,
        )
    if layout.view_count:
        view_slice = slice(
            retained + layout.view_slice.start,
            retained + layout.view_slice.stop,
        )
        direction[view_slice] = np.einsum(
            "m,mci,mc->i",
            local_weight,
            view[selected],
            selected_residual,
            optimize=True,
        )
    return direction


def _anchor_score_direction(
    selected: np.ndarray,
    reliability: np.ndarray,
    state: np.ndarray,
    bias: np.ndarray,
    residual: np.ndarray,
    *,
    layout: _NuisanceLayout,
) -> np.ndarray:
    retained = state.shape[2]
    direction = np.zeros(retained + layout.nuisance_count, dtype=np.float64)
    local_weight = reliability[selected]
    direction[:retained] = np.einsum(
        "a,aci,ac->i",
        local_weight,
        state[selected],
        residual[selected],
        optimize=True,
    )
    if layout.anchor_bias_count:
        bias_slice = slice(
            retained + layout.anchor_bias_slice.start,
            retained + layout.anchor_bias_slice.stop,
        )
        direction[bias_slice] = np.einsum(
            "a,aci,ac->i",
            local_weight,
            bias[selected],
            residual[selected],
            optimize=True,
        )
    return direction


def update_sparse_prior_aware_gauge_belief(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> GaugeAwareBeliefResult:
    """Infer a prior-aware state without materializing a dense gauge design."""

    if not isinstance(batch, GaugeAwareObservationBatch):
        raise TypeError("batch must be a GaugeAwareObservationBatch")
    if not isinstance(gauge, (SparseGaugeDesignV1, TreeSparseGaugeDesignV1)):
        raise TypeError(
            "gauge must be a SparseGaugeDesignV1 or TreeSparseGaugeDesignV1"
        )
    _require(
        batch.gauge_jacobian.shape[2] == 0
        and batch.gauge_prior_covariance.shape == (0, 0),
        "batch must leave gauge ownership to SparseGaugeDesignV1",
    )
    _require(
        gauge.observation_count == len(batch.innovation_m),
        "sparse gauge row count differs from the observation batch",
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
    layout = _NuisanceLayout(
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

    state_prior, nuisance_prior_precision, full_prior = _prior_covariances(
        batch, gauge, cfg
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
    known, cross, nuisance_information = _information_blocks(
        identification_weight,
        state_white,
        local_gauge_white,
        gauge.gauge_indices,
        shared_white,
        view_white,
        gauge_count=gauge.gauge_count,
        layout=layout,
        anchor_weight=anchor_identification_weight,
        anchor_state=anchor_state_white,
        anchor_bias=anchor_bias_white,
    )
    state_mapping, identifiable, query_fraction, basis_diagnostics = (
        _prior_aware_basis_from_information(
            known,
            cross,
            nuisance_information,
            state_prior,
            nuisance_prior_precision,
            batch.query_state_jacobian,
            cfg,
        )
    )
    exact_mixture = cfg.minimum_robust_precision == 0.0
    diagnostics: dict[str, Any] = {
        "identifiability_mode": "prior-aware-native-sparse-schur-v1",
        "robust_likelihood": "grouped nominal/outlier Student-t mixture",
        "robust_likelihood_objective": (
            "exact-group-mixture-gradient"
            if exact_mixture
            else "precision-floored-group-mixture-approximation"
        ),
        "posterior_covariance_kind": (
            "working-gauss-newton-irls-not-exact-mixture-hessian"
        ),
        "minimum_robust_precision": cfg.minimum_robust_precision,
        "prior_nominal_probability_used_inside_mixture": True,
        "association_probability_used_as_reliability": False,
        "association_probability_used_as_row_power": True,
        "row_reliability_semantics": "conditional-covariance-precision-scaling",
        "row_association_semantics": "generalized-Bayes-row-power-v1",
        "group_composite_weight_semantics": "generalized-Bayes likelihood power",
        "observation_composite_weight_mode": batch.composite_weight_mode,
        "anchor_composite_weight_mode": batch.anchor_composite_weight_mode,
        "native_sparse_gauge_solver_version": (SPARSE_PRIOR_AWARE_GAUGE_SOLVER_VERSION),
        "native_sparse_gauge_design_materialized": False,
        "native_sparse_gauge_block_size": gauge.block_size,
        "native_sparse_gauge_count": gauge.gauge_count,
        "native_sparse_gauge_parameter_count": gauge.gauge_parameter_count,
        "dense_gauge_design_avoided_bytes": gauge.equivalent_dense_design_bytes,
        "gauge_prior_representation": (
            "tree-transition-innovation-information-v1"
            if isinstance(gauge, TreeSparseGaugeDesignV1)
            else "dense-covariance-v1"
        ),
        "dense_gauge_prior_covariance_materialized": not isinstance(
            gauge, TreeSparseGaugeDesignV1
        ),
        "dense_gauge_prior_avoided_bytes": (
            gauge.dense_gauge_prior_avoided_bytes
            if isinstance(gauge, TreeSparseGaugeDesignV1)
            else 0
        ),
        "tree_factor_storage_nbytes": (
            gauge.tree_factor_storage_nbytes
            if isinstance(gauge, TreeSparseGaugeDesignV1)
            else 0
        ),
        "tree_prior_information_matrix_materialized": isinstance(
            gauge, TreeSparseGaugeDesignV1
        ),
        **basis_diagnostics,
    }
    if not state_mapping.shape[1]:
        return _sparse_fallback_result(
            batch,
            gauge,
            "no-identifiable-query-state",
            diagnostics,
            prior_covariance=full_prior,
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
    prior_precision = _block_diagonal([np.eye(retained), nuisance_prior_precision])
    joint_dimension = retained + layout.nuisance_count
    solution = np.zeros(joint_dimension)
    observation_precision = expected_observation.copy()
    observation_precision_derivative = np.zeros_like(observation_precision)
    anchor_precision = expected_anchor.copy()
    anchor_precision_derivative = np.zeros_like(anchor_precision)
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
    observation_floor_active: np.ndarray = np.zeros(len(observation_groups), dtype=bool)
    anchor_floor_active: np.ndarray = np.zeros(len(anchor_groups), dtype=bool)

    def system() -> tuple[np.ndarray, np.ndarray]:
        row_precision = np.zeros(len(observation_base))
        for position, selected in enumerate(observation_indices):
            row_precision[selected] = observation_precision[position]
        anchor_row_precision = np.zeros(len(anchor_base))
        for position, selected in enumerate(anchor_indices):
            anchor_row_precision[selected] = anchor_precision[position]
        ordinary_weight = observation_base * row_precision
        independent_weight = anchor_base * anchor_row_precision
        known_block, cross_block, nuisance_block = _information_blocks(
            ordinary_weight,
            state_reduced_white,
            local_gauge_white,
            gauge.gauge_indices,
            shared_white,
            view_white,
            gauge_count=gauge.gauge_count,
            layout=layout,
            anchor_weight=independent_weight,
            anchor_state=anchor_state_reduced_white,
            anchor_bias=anchor_bias_white,
        )
        state_right, nuisance_right = _right_blocks(
            ordinary_weight,
            state_reduced_white,
            local_gauge_white,
            gauge.gauge_indices,
            shared_white,
            view_white,
            target,
            gauge_count=gauge.gauge_count,
            layout=layout,
            anchor_weight=independent_weight,
            anchor_state=anchor_state_reduced_white,
            anchor_bias=anchor_bias_white,
            anchor_target=anchor_target,
        )
        normal = prior_precision.copy()
        normal[:retained, :retained] += known_block
        normal[:retained, retained:] += cross_block
        normal[retained:, :retained] += cross_block.T
        normal[retained:, retained:] += nuisance_block
        right = np.concatenate((state_right, nuisance_right))
        return 0.5 * (normal + normal.T), right

    def refresh_mixture_statistics(
        current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = batch.innovation_m - _observation_prediction(
            state_reduced_raw,
            gauge.local_gauge_jacobian,
            gauge.gauge_indices,
            batch.shared_bias_jacobian,
            batch.view_bias_jacobian,
            current,
            retained=retained,
            layout=layout,
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
                observation_precision_derivative[position] = 0.0
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
            observation_precision_derivative[position] = (
                statistics.expected_precision_derivative
            )
            observation_responsibility[position] = (
                statistics.posterior_nominal_probability
            )
            observation_floor_active[position] = statistics.precision_floor_active

        if not anchor_count:
            return white_residual, np.zeros((0, 3))
        anchor_residual = anchor_innovation - _anchor_prediction(
            anchor_state_reduced_raw,
            anchor_bias,
            current,
            retained=retained,
            layout=layout,
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
                anchor_precision_derivative[position] = 0.0
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
            anchor_precision_derivative[position] = (
                statistics.expected_precision_derivative
            )
            anchor_responsibility[position] = statistics.posterior_nominal_probability
            anchor_floor_active[position] = statistics.precision_floor_active
        return white_residual, white_anchor_residual

    refresh_mixture_statistics(solution)
    normal = prior_precision.copy()
    right = np.zeros(joint_dimension)
    condition_number = float("inf")
    stationarity_norm = float("inf")
    solution_delta = float("inf")
    fixed_point_converged = False
    iteration_count = 0
    final_white_residual = np.zeros_like(batch.innovation_m)
    final_white_anchor_residual = np.zeros((anchor_count, 3))
    for iteration in range(cfg.maximum_iterations):
        iteration_count = iteration + 1
        normal, right = system()
        condition_number = float(np.linalg.cond(normal))
        if (
            not np.isfinite(condition_number)
            or condition_number > cfg.maximum_condition_number
        ):
            diagnostics["condition_number"] = condition_number
            return _sparse_fallback_result(
                batch,
                gauge,
                "ill-conditioned-posterior",
                diagnostics,
                prior_covariance=full_prior,
            )
        try:
            candidate = _solve_spd_system(normal, right)
        except np.linalg.LinAlgError:
            return _sparse_fallback_result(
                batch,
                gauge,
                "singular-posterior",
                diagnostics,
                prior_covariance=full_prior,
            )
        solution_delta = float(np.linalg.norm(candidate - solution))
        solution = candidate
        final_white_residual, final_white_anchor_residual = refresh_mixture_statistics(
            solution
        )
        normal, right = system()
        stationarity_norm = float(np.linalg.norm(normal @ solution - right))
        solution_scale = 1.0 + float(np.linalg.norm(solution))
        stationarity_scale = 1.0 + float(np.linalg.norm(right))
        if (
            solution_delta <= cfg.convergence_tolerance * solution_scale
            and stationarity_norm <= cfg.convergence_tolerance * stationarity_scale
        ):
            fixed_point_converged = True
            break

    condition_number = float(np.linalg.cond(normal))
    if (
        not np.isfinite(condition_number)
        or condition_number > cfg.maximum_condition_number
    ):
        diagnostics["condition_number"] = condition_number
        return _sparse_fallback_result(
            batch,
            gauge,
            "ill-conditioned-final-posterior",
            diagnostics,
            prior_covariance=full_prior,
        )
    try:
        reduced_covariance = _spd_covariance(normal)
    except np.linalg.LinAlgError:
        return _sparse_fallback_result(
            batch,
            gauge,
            "singular-final-posterior",
            diagnostics,
            prior_covariance=full_prior,
        )

    exact_hessian = normal.copy()
    for position, selected in enumerate(observation_indices):
        active = selected[observation_row_weight[selected] > 0.0]
        if not len(active):
            continue
        score_direction = _score_direction(
            active,
            observation_row_weight,
            state_reduced_white,
            local_gauge_white,
            gauge.gauge_indices,
            shared_white,
            view_white,
            final_white_residual,
            gauge_count=gauge.gauge_count,
            layout=layout,
        )
        exact_hessian += (
            2.0
            * observation_group_power[position]
            * observation_precision_derivative[position]
            * np.outer(score_direction, score_direction)
        )
    for position, selected in enumerate(anchor_indices):
        active = selected[anchor_reliability[selected] > 0.0]
        if not len(active):
            continue
        score_direction = _anchor_score_direction(
            active,
            anchor_reliability,
            anchor_state_reduced_white,
            anchor_bias_white,
            final_white_anchor_residual,
            layout=layout,
        )
        exact_hessian += (
            2.0
            * anchor_group_power[position]
            * anchor_precision_derivative[position]
            * np.outer(score_direction, score_direction)
        )
    exact_hessian = 0.5 * (exact_hessian + exact_hessian.T)
    exact_hessian_eigenvalues = np.linalg.eigvalsh(exact_hessian)

    state_coefficients = state_mapping @ solution[:retained]
    full_solution = np.concatenate((state_coefficients, solution[retained:]))
    covariance = _full_covariance(
        state_prior,
        state_mapping,
        reduced_covariance,
        layout.nuisance_count,
    )
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
    diagnostics.update(
        {
            "iterations": iteration_count,
            "mixture_fixed_point_converged": fixed_point_converged,
            "mixture_solution_delta": solution_delta,
            "mixture_stationarity_norm": stationarity_norm,
            "condition_number": condition_number,
            "posterior_solver": "native-sparse-normal-equations-cholesky-v1",
            "normal_equation_dimension": joint_dimension,
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
            "anchor_group_precision_floor_active": anchor_floor_active.tolist(),
            "exact_reduced_mixture_hessian_minimum_eigenvalue": float(
                np.min(exact_hessian_eigenvalues)
            ),
            "exact_reduced_mixture_hessian_maximum_eigenvalue": float(
                np.max(exact_hessian_eigenvalues)
            ),
            "exact_reduced_mixture_hessian_positive_definite": bool(
                np.min(exact_hessian_eigenvalues) > 0.0
            ),
        }
    )
    if not np.all(np.isfinite(full_solution)) or maximum_update > update_limit:
        return _sparse_fallback_result(
            batch,
            gauge,
            "implausible-state-update",
            diagnostics,
            prior_covariance=full_prior,
        )

    ordinary_robust = np.zeros(len(observation_base))
    for position, selected in enumerate(observation_indices):
        ordinary_robust[selected] = observation_precision[position]
    anchor_robust = np.zeros(len(anchor_base))
    for position, selected in enumerate(anchor_indices):
        anchor_robust[selected] = anchor_precision[position]
    nuisance = full_solution[state_count:]
    gauge_slice = slice(0, gauge.gauge_parameter_count)
    shared_slice = slice(gauge_slice.stop, gauge_slice.stop + shared_count)
    view_slice = slice(shared_slice.stop, shared_slice.stop + view_count)
    anchor_bias_slice = slice(view_slice.stop, view_slice.stop + anchor_bias_count)
    return GaugeAwareBeliefResult(
        inference_admissible=True,
        reason="inference-admissible",
        state_coefficients=state_coefficients,
        gauge_delta=nuisance[gauge_slice],
        shared_bias_coefficients=nuisance[shared_slice],
        view_bias_coefficients=nuisance[view_slice],
        anchor_bias_coefficients=nuisance[anchor_bias_slice],
        posterior_covariance=covariance,
        identifiable_state_transform=state_mapping,
        identifiable_fractions=identifiable,
        query_sensitivity_fractions=query_fraction,
        robust_weights=ordinary_robust,
        anchor_robust_weights=anchor_robust,
        diagnostics=diagnostics,
        input_lineage={} if batch.metadata is None else batch.metadata,
    )


__all__ = [
    "SPARSE_PRIOR_AWARE_GAUGE_SOLVER_VERSION",
    "SparseGaugeDesignV1",
    "TreeSparseGaugeDesignV1",
    "update_sparse_prior_aware_gauge_belief",
]
