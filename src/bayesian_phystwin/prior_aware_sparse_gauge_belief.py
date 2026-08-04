"""Native block-sparse prior-aware inference for explicit gauge factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    _block_diagonal,
    _readonly,
    _regularized_precision,
    _require,
)
from ._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _full_covariance,
    _group_layout,
    _prior_aware_basis_from_information,
    _solve_spd_system,
    _spd_covariance,
    _student_t_mixture_statistics,
    _whiten,
)


@dataclass(frozen=True)
class SparseGaugeAwareObservationBatch:
    """A validated observation batch with one local gauge block per row."""

    base: GaugeAwareObservationBatch
    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    gauge_prior_covariance: np.ndarray
    gauge_block_size: int = 7

    def __post_init__(self) -> None:
        if not isinstance(self.base, GaugeAwareObservationBatch):
            raise TypeError("base must be a GaugeAwareObservationBatch")
        _require(
            self.base.gauge_jacobian.shape[2] == 0
            and self.base.gauge_prior_covariance.shape == (0, 0),
            "base batch must not contain a dense gauge design",
        )
        local = np.asarray(self.local_gauge_jacobian, dtype=np.float64)
        _require(
            local.shape == (len(self.base.innovation_m), 3, int(self.gauge_block_size)),
            "local_gauge_jacobian must have shape (M, 3, B)",
        )
        _require(
            np.all(np.isfinite(local)),
            "local_gauge_jacobian contains non-finite values",
        )
        raw_indices = np.asarray(self.gauge_indices)
        _require(
            raw_indices.ndim == 1
            and raw_indices.shape == (len(self.base.innovation_m),)
            and raw_indices.dtype.kind in "iu",
            "gauge_indices must be an integer vector with shape (M,)",
        )
        indices = raw_indices.astype(np.int64, copy=True)
        _require(
            isinstance(self.gauge_block_size, int) and self.gauge_block_size >= 1,
            "gauge_block_size must be a positive integer",
        )
        prior = np.asarray(self.gauge_prior_covariance, dtype=np.float64)
        _require(
            prior.ndim == 2 and prior.shape[0] == prior.shape[1],
            "gauge_prior_covariance must be square",
        )
        _require(
            prior.shape[0] > 0 and prior.shape[0] % self.gauge_block_size == 0,
            "gauge prior dimension must contain complete local blocks",
        )
        _require(
            np.all(np.isfinite(prior)),
            "gauge_prior_covariance contains non-finite values",
        )
        _regularized_precision(
            prior,
            "sparse gauge prior covariance",
            eigenvalue_floor=1e-12,
        )
        group_count = prior.shape[0] // self.gauge_block_size
        _require(
            np.all((indices >= 0) & (indices < group_count)),
            "gauge_indices reference an unknown local block",
        )
        object.__setattr__(self, "local_gauge_jacobian", _readonly(local))
        indices.setflags(write=False)
        object.__setattr__(self, "gauge_indices", indices)
        object.__setattr__(self, "gauge_prior_covariance", _readonly(prior))

    @property
    def gauge_parameter_count(self) -> int:
        """Return the complete joint gauge dimension."""

        return int(self.gauge_prior_covariance.shape[0])

    @property
    def gauge_group_count(self) -> int:
        """Return the number of local gauge blocks in the joint prior."""

        return self.gauge_parameter_count // self.gauge_block_size


def _sparse_fallback_result(
    batch: SparseGaugeAwareObservationBatch,
    reason: str,
    diagnostics: dict[str, Any],
    *,
    prior_covariance: np.ndarray,
) -> GaugeAwareBeliefResult:
    base = batch.base
    state_count = base.state_jacobian.shape[2]
    shared_count = base.shared_bias_jacobian.shape[2]
    view_count = base.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if base.anchor_bias_jacobian is None else base.anchor_bias_jacobian.shape[2]
    )
    anchor_count = (
        0 if base.anchor_innovation_m is None else len(base.anchor_innovation_m)
    )
    return GaugeAwareBeliefResult(
        inference_admissible=False,
        reason=reason,
        state_coefficients=np.zeros(state_count),
        gauge_delta=np.zeros(batch.gauge_parameter_count),
        shared_bias_coefficients=np.zeros(shared_count),
        view_bias_coefficients=np.zeros(view_count),
        anchor_bias_coefficients=np.zeros(anchor_bias_count),
        posterior_covariance=prior_covariance,
        identifiable_state_transform=np.zeros((state_count, 0)),
        identifiable_fractions=np.zeros(0),
        query_sensitivity_fractions=np.zeros(0),
        robust_weights=np.zeros(len(base.innovation_m)),
        anchor_robust_weights=np.zeros(anchor_count),
        diagnostics=diagnostics,
        input_lineage=base.metadata or {},
    )


def _nuisance_indices(
    gauge_index: int,
    *,
    gauge_block_size: int,
    gauge_count: int,
    shared_count: int,
    view_count: int,
) -> np.ndarray:
    gauge_start = gauge_index * gauge_block_size
    return np.concatenate(
        (
            np.arange(gauge_start, gauge_start + gauge_block_size),
            np.arange(gauge_count, gauge_count + shared_count + view_count),
        )
    )


def _weighted_information(
    state: np.ndarray,
    local_gauge: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    gauge_indices: np.ndarray,
    weight: np.ndarray,
    *,
    gauge_block_size: int,
    gauge_group_count: int,
    gauge_count: int,
    nuisance_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_count = state.shape[2]
    known = np.einsum("m,mci,mcj->ij", weight, state, state)
    cross = np.zeros((state_count, nuisance_count), dtype=np.float64)
    nuisance: np.ndarray = np.zeros((nuisance_count, nuisance_count), dtype=np.float64)
    for gauge_index in range(gauge_group_count):
        selected = np.flatnonzero((gauge_indices == gauge_index) & (weight > 0.0))
        if not len(selected):
            continue
        compact = np.concatenate(
            (local_gauge[selected], shared[selected], view[selected]), axis=2
        )
        indices = _nuisance_indices(
            gauge_index,
            gauge_block_size=gauge_block_size,
            gauge_count=gauge_count,
            shared_count=shared.shape[2],
            view_count=view.shape[2],
        )
        local_weight = weight[selected]
        cross[:, indices] += np.einsum(
            "m,mci,mcj->ij", local_weight, state[selected], compact
        )
        nuisance[np.ix_(indices, indices)] += np.einsum(
            "m,mci,mcj->ij", local_weight, compact, compact
        )
    return known, cross, nuisance


def _accumulate_observation_system(
    normal: np.ndarray,
    right: np.ndarray,
    state: np.ndarray,
    local_gauge: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    gauge_indices: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    gauge_block_size: int,
    gauge_group_count: int,
    gauge_count: int,
) -> None:
    retained = state.shape[2]
    for gauge_index in range(gauge_group_count):
        selected = np.flatnonzero((gauge_indices == gauge_index) & (weight > 0.0))
        if not len(selected):
            continue
        compact = np.concatenate(
            (state[selected], local_gauge[selected], shared[selected], view[selected]),
            axis=2,
        )
        nuisance_indices = _nuisance_indices(
            gauge_index,
            gauge_block_size=gauge_block_size,
            gauge_count=gauge_count,
            shared_count=shared.shape[2],
            view_count=view.shape[2],
        )
        indices = np.concatenate((np.arange(retained), retained + nuisance_indices))
        local_weight = weight[selected]
        normal[np.ix_(indices, indices)] += np.einsum(
            "m,mci,mcj->ij", local_weight, compact, compact
        )
        right[indices] += np.einsum(
            "m,mci,mc->i", local_weight, compact, target[selected]
        )


def _observation_prediction(
    state: np.ndarray,
    local_gauge: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    gauge_indices: np.ndarray,
    solution: np.ndarray,
    *,
    gauge_block_size: int,
    gauge_group_count: int,
    gauge_count: int,
) -> np.ndarray:
    retained = state.shape[2]
    prediction = np.einsum("mci,i->mc", state, solution[:retained])
    nuisance = solution[retained:]
    for gauge_index in range(gauge_group_count):
        selected = np.flatnonzero(gauge_indices == gauge_index)
        if not len(selected):
            continue
        start = gauge_index * gauge_block_size
        prediction[selected] += np.einsum(
            "mci,i->mc",
            local_gauge[selected],
            nuisance[start : start + gauge_block_size],
        )
    offset = gauge_count
    if shared.shape[2]:
        prediction += np.einsum(
            "mci,i->mc", shared, nuisance[offset : offset + shared.shape[2]]
        )
        offset += shared.shape[2]
    if view.shape[2]:
        prediction += np.einsum(
            "mci,i->mc", view, nuisance[offset : offset + view.shape[2]]
        )
    return prediction


def _observation_score_direction(
    selected: np.ndarray,
    reliability: np.ndarray,
    state: np.ndarray,
    local_gauge: np.ndarray,
    shared: np.ndarray,
    view: np.ndarray,
    gauge_indices: np.ndarray,
    residual: np.ndarray,
    *,
    gauge_block_size: int,
    gauge_group_count: int,
    gauge_count: int,
    joint_dimension: int,
) -> np.ndarray:
    retained = state.shape[2]
    score: np.ndarray = np.zeros(joint_dimension, dtype=np.float64)
    for gauge_index in range(gauge_group_count):
        active = selected[gauge_indices[selected] == gauge_index]
        if not len(active):
            continue
        compact = np.concatenate(
            (state[active], local_gauge[active], shared[active], view[active]), axis=2
        )
        nuisance_indices = _nuisance_indices(
            gauge_index,
            gauge_block_size=gauge_block_size,
            gauge_count=gauge_count,
            shared_count=shared.shape[2],
            view_count=view.shape[2],
        )
        indices = np.concatenate((np.arange(retained), retained + nuisance_indices))
        score[indices] += np.einsum(
            "m,mci,mc->i",
            reliability[active],
            compact,
            residual[active],
        )
    return score


def update_prior_aware_sparse_gauge_belief(
    batch: SparseGaugeAwareObservationBatch,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> GaugeAwareBeliefResult:
    """Infer state and a correlated joint gauge without dense zero blocks."""

    cfg = config or PriorAwareGaugeConfigV1()
    base = batch.base
    if base.prior_nominal_probability is None or base.composite_weight is None:
        raise ValueError("validated observation mixture metadata is missing")
    observation_nominal = np.asarray(base.prior_nominal_probability)
    observation_composite = np.asarray(base.composite_weight)

    state_count = base.state_jacobian.shape[2]
    gauge_count = batch.gauge_parameter_count
    shared_count = base.shared_bias_jacobian.shape[2]
    view_count = base.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if base.anchor_bias_jacobian is None else base.anchor_bias_jacobian.shape[2]
    )
    nuisance_count = gauge_count + shared_count + view_count + anchor_bias_count

    anchor_count = (
        0 if base.anchor_innovation_m is None else len(base.anchor_innovation_m)
    )
    if anchor_count:
        if (
            base.anchor_innovation_m is None
            or base.anchor_covariance_m2 is None
            or base.anchor_state_jacobian is None
            or base.anchor_correlation_group_ids is None
            or base.anchor_prior_reliability is None
            or base.anchor_prior_nominal_probability is None
            or base.anchor_composite_weight is None
        ):
            raise ValueError("validated anchor mixture metadata is missing")
        anchor_innovation = np.asarray(base.anchor_innovation_m)
        anchor_covariance = np.asarray(base.anchor_covariance_m2)
        anchor_state = np.asarray(base.anchor_state_jacobian)
        anchor_groups_input = base.anchor_correlation_group_ids
        anchor_reliability = np.asarray(base.anchor_prior_reliability)
        anchor_nominal = np.asarray(base.anchor_prior_nominal_probability)
        anchor_composite = np.asarray(base.anchor_composite_weight)
        anchor_bias = (
            np.zeros((anchor_count, 3, anchor_bias_count))
            if base.anchor_bias_jacobian is None
            else np.asarray(base.anchor_bias_jacobian)
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
        (state_white, local_gauge_white, shared_white, view_white),
        whiteners,
    ) = _whiten(
        base.innovation_m,
        base.observation_covariance_m2,
        (
            base.state_jacobian,
            batch.local_gauge_jacobian,
            base.shared_bias_jacobian,
            base.view_bias_jacobian,
        ),
        name="observation",
    )
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

    (
        observation_groups,
        observation_indices,
        observation_base,
        observation_prior,
        observation_group_power,
    ) = _group_layout(
        base.correlation_group_ids,
        base.prior_reliability,
        observation_nominal,
        observation_composite,
        cfg.effective_samples_per_correlation_group,
        composite_weight_mode=base.composite_weight_mode,
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
            composite_weight_mode=base.anchor_composite_weight_mode,
        )
    else:
        anchor_groups, anchor_indices = (), ()
        anchor_base = np.zeros(0)
        anchor_prior = np.zeros(0)
        anchor_group_power = np.zeros(0)

    state_prior = (
        np.eye(state_count) * cfg.state_prior_std_m**2
        if base.state_prior_covariance_m2 is None
        else np.asarray(base.state_prior_covariance_m2)
    )
    nuisance_parts = [np.asarray(batch.gauge_prior_covariance)]
    if shared_count:
        nuisance_parts.append(np.eye(shared_count) * cfg.shared_bias_prior_std_m**2)
    if view_count:
        nuisance_parts.append(np.eye(view_count) * cfg.view_bias_prior_std_m**2)
    if base.anchor_bias_prior_covariance is not None:
        nuisance_parts.append(np.asarray(base.anchor_bias_prior_covariance))
    nuisance_prior = _block_diagonal(nuisance_parts)
    full_prior = _block_diagonal([state_prior, nuisance_prior])

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

    known, cross, nuisance_data = _weighted_information(
        state_white,
        local_gauge_white,
        shared_white,
        view_white,
        batch.gauge_indices,
        identification_weight,
        gauge_block_size=batch.gauge_block_size,
        gauge_group_count=batch.gauge_group_count,
        gauge_count=gauge_count,
        nuisance_count=nuisance_count,
    )
    if anchor_count:
        known += np.einsum(
            "a,aci,acj->ij",
            anchor_identification_weight,
            anchor_state_white,
            anchor_state_white,
        )
        anchor_start = gauge_count + shared_count + view_count
        anchor_indices_global = np.arange(
            anchor_start, anchor_start + anchor_bias_count
        )
        if anchor_bias_count:
            cross[:, anchor_indices_global] += np.einsum(
                "a,aci,acj->ij",
                anchor_identification_weight,
                anchor_state_white,
                anchor_bias_white,
            )
            nuisance_data[np.ix_(anchor_indices_global, anchor_indices_global)] += (
                np.einsum(
                    "a,aci,acj->ij",
                    anchor_identification_weight,
                    anchor_bias_white,
                    anchor_bias_white,
                )
            )
    nuisance_information = (
        _regularized_precision(
            nuisance_prior,
            "nuisance prior covariance",
            eigenvalue_floor=cfg.prior_eigenvalue_floor,
        )
        + nuisance_data
    )
    conditional = known - cross @ np.linalg.solve(nuisance_information, cross.T)
    state_mapping, identifiable, query_fraction, basis_diagnostics = (
        _prior_aware_basis_from_information(
            known,
            conditional,
            state_prior,
            base.query_state_jacobian,
            cfg,
        )
    )
    exact_mixture = cfg.minimum_robust_precision == 0.0
    diagnostics: dict[str, Any] = {
        "identifiability_mode": "prior-aware-block-sparse-schur-v1",
        "gauge_storage": "native-block-sparse-v1",
        "gauge_block_size": batch.gauge_block_size,
        "gauge_group_count": batch.gauge_group_count,
        "dense_gauge_design_allocated": False,
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
        "row_reliability_semantics": "conditional-covariance-precision-scaling",
        "group_composite_weight_semantics": "generalized-Bayes likelihood power",
        "observation_composite_weight_mode": base.composite_weight_mode,
        "anchor_composite_weight_mode": base.anchor_composite_weight_mode,
        **basis_diagnostics,
    }
    if not state_mapping.shape[1]:
        return _sparse_fallback_result(
            batch,
            "no-identifiable-query-state",
            diagnostics,
            prior_covariance=full_prior,
        )

    retained = state_mapping.shape[1]
    observation_state = np.einsum("mcs,sr->mcr", state_white, state_mapping)
    raw_observation_state = np.einsum("mcs,sr->mcr", base.state_jacobian, state_mapping)
    anchor_state_reduced = np.einsum("acs,sr->acr", anchor_state_white, state_mapping)
    raw_anchor_state_reduced = np.einsum("acs,sr->acr", anchor_state, state_mapping)
    reduced_prior = _block_diagonal([np.eye(retained), nuisance_prior])
    prior_precision = _regularized_precision(
        reduced_prior,
        "reduced prior covariance",
        eigenvalue_floor=cfg.prior_eigenvalue_floor,
    )
    joint_dimension = retained + nuisance_count
    solution = np.zeros(joint_dimension)
    observation_precision = expected_observation.copy()
    observation_precision_derivative = np.zeros_like(observation_precision)
    anchor_precision = expected_anchor.copy()
    anchor_precision_derivative = np.zeros_like(anchor_precision)
    observation_responsibility = np.clip(
        observation_prior, cfg.probability_floor, 1.0 - cfg.probability_floor
    )
    anchor_responsibility = np.clip(
        anchor_prior, cfg.probability_floor, 1.0 - cfg.probability_floor
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
        normal = prior_precision.copy()
        right = np.zeros_like(solution)
        _accumulate_observation_system(
            normal,
            right,
            observation_state,
            local_gauge_white,
            shared_white,
            view_white,
            batch.gauge_indices,
            target,
            ordinary_weight,
            gauge_block_size=batch.gauge_block_size,
            gauge_group_count=batch.gauge_group_count,
            gauge_count=gauge_count,
        )
        if anchor_count:
            compact = np.concatenate((anchor_state_reduced, anchor_bias_white), axis=2)
            anchor_start = retained + gauge_count + shared_count + view_count
            indices = np.concatenate(
                (
                    np.arange(retained),
                    np.arange(anchor_start, anchor_start + anchor_bias_count),
                )
            )
            normal[np.ix_(indices, indices)] += np.einsum(
                "a,aci,acj->ij", independent_weight, compact, compact
            )
            right[indices] += np.einsum(
                "a,aci,ac->i", independent_weight, compact, anchor_target
            )
        return 0.5 * (normal + normal.T), right

    def refresh_mixture_statistics(
        current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = base.innovation_m - _observation_prediction(
            raw_observation_state,
            batch.local_gauge_jacobian,
            base.shared_bias_jacobian,
            base.view_bias_jacobian,
            batch.gauge_indices,
            current,
            gauge_block_size=batch.gauge_block_size,
            gauge_group_count=batch.gauge_group_count,
            gauge_count=gauge_count,
        )
        white_residual = np.einsum("mij,mj->mi", whiteners, residual)
        for position, selected in enumerate(observation_indices):
            active = selected[base.prior_reliability[selected] > 0.0]
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
                    base.prior_reliability[active]
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
        anchor_prediction = np.einsum(
            "aci,i->ac", raw_anchor_state_reduced, current[:retained]
        )
        if anchor_bias_count:
            anchor_start = retained + gauge_count + shared_count + view_count
            anchor_prediction += np.einsum(
                "aci,i->ac",
                anchor_bias,
                current[anchor_start : anchor_start + anchor_bias_count],
            )
        anchor_residual = anchor_innovation - anchor_prediction
        white_anchor_residual = np.einsum(
            "aij,aj->ai", anchor_whiteners, anchor_residual
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
    final_white_residual = np.zeros_like(base.innovation_m)
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
                "ill-conditioned-posterior",
                diagnostics,
                prior_covariance=full_prior,
            )
        try:
            candidate = _solve_spd_system(normal, right)
        except np.linalg.LinAlgError:
            return _sparse_fallback_result(
                batch,
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
            "ill-conditioned-final-posterior",
            diagnostics,
            prior_covariance=full_prior,
        )
    try:
        reduced_covariance = _spd_covariance(normal)
    except np.linalg.LinAlgError:
        return _sparse_fallback_result(
            batch,
            "singular-final-posterior",
            diagnostics,
            prior_covariance=full_prior,
        )

    exact_hessian = normal.copy()
    for position, selected in enumerate(observation_indices):
        active = selected[base.prior_reliability[selected] > 0.0]
        if not len(active):
            continue
        score_direction = _observation_score_direction(
            active,
            base.prior_reliability,
            observation_state,
            local_gauge_white,
            shared_white,
            view_white,
            batch.gauge_indices,
            final_white_residual,
            gauge_block_size=batch.gauge_block_size,
            gauge_group_count=batch.gauge_group_count,
            gauge_count=gauge_count,
            joint_dimension=joint_dimension,
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
        score_direction = np.zeros(joint_dimension, dtype=np.float64)
        compact = np.concatenate(
            (anchor_state_reduced[active], anchor_bias_white[active]), axis=2
        )
        anchor_start = retained + gauge_count + shared_count + view_count
        indices = np.concatenate(
            (
                np.arange(retained),
                np.arange(anchor_start, anchor_start + anchor_bias_count),
            )
        )
        score_direction[indices] = np.einsum(
            "a,aci,ac->i",
            anchor_reliability[active],
            compact,
            final_white_anchor_residual[active],
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
        state_prior, state_mapping, reduced_covariance, nuisance_count
    )
    query_update = np.einsum("qcs,s->qc", base.query_state_jacobian, state_coefficients)
    maximum_update = float(np.max(np.linalg.norm(query_update, axis=1), initial=0.0))
    relative_limit = (
        cfg.maximum_update_to_physical_response_ratio * base.physical_response_scale_m
    )
    update_limit = min(cfg.maximum_state_update_m, relative_limit)
    diagnostics.update(
        {
            "iterations": iteration_count,
            "mixture_fixed_point_converged": fixed_point_converged,
            "mixture_solution_delta": solution_delta,
            "mixture_stationarity_norm": stationarity_norm,
            "condition_number": condition_number,
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
    gauge_slice = slice(0, gauge_count)
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
        input_lineage=base.metadata or {},
    )


__all__ = [
    "SparseGaugeAwareObservationBatch",
    "update_prior_aware_sparse_gauge_belief",
]
