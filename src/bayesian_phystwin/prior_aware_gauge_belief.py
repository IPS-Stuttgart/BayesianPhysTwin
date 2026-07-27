"""Prior-aware group-mixture inference for gauge- and bias-aware updates."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    _block_diagonal,
    _fallback_result,
    _regularized_precision,
)
from ._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _full_covariance,
    _group_layout,
    _prior_aware_basis,
    _prior_covariances,
    _solve_spd_system,
    _spd_covariance,
    _student_t_mixture_statistics,
    _whiten,
)


def update_prior_aware_gauge_belief(
    batch: GaugeAwareObservationBatch,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> GaugeAwareBeliefResult:
    """Infer state while conditioning identifiability on nuisance priors."""

    cfg = config or PriorAwareGaugeConfigV1()
    if (
        batch.prior_nominal_probability is None
        or batch.composite_weight is None
    ):
        raise ValueError("validated observation mixture metadata is missing")
    observation_nominal = np.asarray(batch.prior_nominal_probability)
    observation_composite = np.asarray(batch.composite_weight)

    state_count = batch.state_jacobian.shape[2]
    gauge_count = batch.gauge_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    nuisance_count = gauge_count + shared_count + view_count + anchor_bias_count
    ordinary_nuisance = np.concatenate(
        (
            batch.gauge_jacobian,
            batch.shared_bias_jacobian,
            batch.view_bias_jacobian,
            np.zeros((len(batch.innovation_m), 3, anchor_bias_count)),
        ),
        axis=2,
    )

    anchor_count = (
        0
        if batch.anchor_innovation_m is None
        else len(batch.anchor_innovation_m)
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
    anchor_nuisance = np.concatenate(
        (
            np.zeros((anchor_count, 3, gauge_count + shared_count + view_count)),
            anchor_bias,
        ),
        axis=2,
    )

    target, (state_white, nuisance_white), whiteners = _whiten(
        batch.innovation_m,
        batch.observation_covariance_m2,
        (batch.state_jacobian, ordinary_nuisance),
        name="observation",
    )
    if anchor_count:
        (
            anchor_target,
            (anchor_state_white, anchor_nuisance_white),
            anchor_whiteners,
        ) = _whiten(
            anchor_innovation,
            anchor_covariance,
            (anchor_state, anchor_nuisance),
            name="anchor",
        )
    else:
        anchor_target = np.zeros((0, 3))
        anchor_state_white = anchor_state
        anchor_nuisance_white = anchor_nuisance
        anchor_whiteners = np.zeros((0, 3, 3))
    (
        observation_groups,
        observation_indices,
        observation_base,
        observation_prior,
        observation_group_power,
    ) = _group_layout(
        batch.correlation_group_ids,
        batch.prior_reliability,
        observation_nominal,
        observation_composite,
        cfg.effective_samples_per_correlation_group,
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
        )
    else:
        anchor_groups, anchor_indices = (), ()
        anchor_base = np.zeros(0)
        anchor_prior = np.zeros(0)
        anchor_group_power = np.zeros(0)
    state_prior, nuisance_prior, full_prior = _prior_covariances(batch, cfg)
    expected_observation = observation_prior + (
        1.0 - observation_prior
    ) / cfg.outlier_covariance_multiplier
    expected_anchor = anchor_prior + (
        1.0 - anchor_prior
    ) / cfg.outlier_covariance_multiplier
    identification_weight = observation_base.copy()
    for position, selected in enumerate(observation_indices):
        identification_weight[selected] *= expected_observation[position]
    anchor_identification_weight = anchor_base.copy()
    for position, selected in enumerate(anchor_indices):
        anchor_identification_weight[selected] *= expected_anchor[position]
    state_mapping, identifiable, query_fraction, basis_diagnostics = (
        _prior_aware_basis(
            state_white,
            nuisance_white,
            anchor_state_white,
            anchor_nuisance_white,
            state_prior,
            nuisance_prior,
            identification_weight,
            anchor_identification_weight,
            batch.query_state_jacobian,
            cfg,
        )
    )
    exact_mixture = cfg.minimum_robust_precision == 0.0
    diagnostics: dict[str, Any] = {
        "identifiability_mode": "prior-aware-schur-v1",
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
        **basis_diagnostics,
    }
    if not state_mapping.shape[1]:
        return _fallback_result(
            batch,
            "no-identifiable-query-state",
            diagnostics,
            prior_covariance=full_prior,
        )
    retained = state_mapping.shape[1]
    observation_design = np.concatenate(
        (
            np.einsum("mcs,sr->mcr", state_white, state_mapping),
            nuisance_white,
        ),
        axis=2,
    )
    anchor_design = np.concatenate(
        (
            np.einsum("acs,sr->acr", anchor_state_white, state_mapping),
            anchor_nuisance_white,
        ),
        axis=2,
    )
    raw_observation_design = np.concatenate(
        (
            np.einsum("mcs,sr->mcr", batch.state_jacobian, state_mapping),
            ordinary_nuisance,
        ),
        axis=2,
    )
    raw_anchor_design = np.concatenate(
        (
            np.einsum("acs,sr->acr", anchor_state, state_mapping),
            anchor_nuisance,
        ),
        axis=2,
    )
    reduced_prior = _block_diagonal(
        [np.eye(retained), nuisance_prior]
        if nuisance_count
        else [np.eye(retained)]
    )
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
        observation_prior,
        cfg.probability_floor,
        1.0 - cfg.probability_floor,
    )
    anchor_responsibility = np.clip(
        anchor_prior,
        cfg.probability_floor,
        1.0 - cfg.probability_floor,
    )
    observation_floor_active = np.zeros(len(observation_groups), dtype=bool)
    anchor_floor_active = np.zeros(len(anchor_groups), dtype=bool)

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
        normal += np.einsum(
            "m,mci,mcj->ij",
            ordinary_weight,
            observation_design,
            observation_design,
        )
        right += np.einsum(
            "m,mci,mc->i", ordinary_weight, observation_design, target
        )
        if anchor_count:
            normal += np.einsum(
                "a,aci,acj->ij",
                independent_weight,
                anchor_design,
                anchor_design,
            )
            right += np.einsum(
                "a,aci,ac->i", independent_weight, anchor_design, anchor_target
            )
        return 0.5 * (normal + normal.T), right

    def refresh_mixture_statistics(
        current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = batch.innovation_m - np.einsum(
            "mci,i->mc", raw_observation_design, current
        )
        white_residual = np.einsum("mij,mj->mi", whiteners, residual)
        for position, selected in enumerate(observation_indices):
            active = selected[batch.prior_reliability[selected] > 0.0]
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
                    batch.prior_reliability[active]
                    * np.sum(np.square(white_residual[active]), axis=1)
                )
            )
            statistics = _student_t_mixture_statistics(
                squared_mahalanobis,
                3 * len(active),
                observation_prior[position],
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
        anchor_residual = anchor_innovation - np.einsum(
            "aci,i->ac", raw_anchor_design, current
        )
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
                anchor_prior[position],
                cfg,
            )
            anchor_precision[position] = statistics.expected_precision
            anchor_precision_derivative[position] = (
                statistics.expected_precision_derivative
            )
            anchor_responsibility[position] = (
                statistics.posterior_nominal_probability
            )
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
            return _fallback_result(
                batch,
                "ill-conditioned-posterior",
                diagnostics,
                prior_covariance=full_prior,
            )
        try:
            candidate = _solve_spd_system(normal, right)
        except np.linalg.LinAlgError:
            return _fallback_result(
                batch,
                "singular-posterior",
                diagnostics,
                prior_covariance=full_prior,
            )
        solution_delta = float(np.linalg.norm(candidate - solution))
        solution = candidate
        final_white_residual, final_white_anchor_residual = (
            refresh_mixture_statistics(solution)
        )
        normal, right = system()
        stationarity_norm = float(np.linalg.norm(normal @ solution - right))
        solution_scale = 1.0 + float(np.linalg.norm(solution))
        stationarity_scale = 1.0 + float(np.linalg.norm(right))
        if (
            solution_delta <= cfg.convergence_tolerance * solution_scale
            and stationarity_norm
            <= cfg.convergence_tolerance * stationarity_scale
        ):
            fixed_point_converged = True
            break

    condition_number = float(np.linalg.cond(normal))
    if (
        not np.isfinite(condition_number)
        or condition_number > cfg.maximum_condition_number
    ):
        diagnostics["condition_number"] = condition_number
        return _fallback_result(
            batch,
            "ill-conditioned-final-posterior",
            diagnostics,
            prior_covariance=full_prior,
        )
    try:
        reduced_covariance = _spd_covariance(normal)
    except np.linalg.LinAlgError:
        return _fallback_result(
            batch,
            "singular-final-posterior",
            diagnostics,
            prior_covariance=full_prior,
        )

    exact_hessian = normal.copy()
    for position, selected in enumerate(observation_indices):
        active = selected[batch.prior_reliability[selected] > 0.0]
        if not len(active):
            continue
        score_direction = np.einsum(
            "m,mci,mc->i",
            batch.prior_reliability[active],
            observation_design[active],
            final_white_residual[active],
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
        score_direction = np.einsum(
            "a,aci,ac->i",
            anchor_reliability[active],
            anchor_design[active],
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
    query_update = np.einsum(
        "qcs,s->qc", batch.query_state_jacobian, state_coefficients
    )
    maximum_update = float(
        np.max(np.linalg.norm(query_update, axis=1), initial=0.0)
    )
    relative_limit = (
        cfg.maximum_update_to_physical_response_ratio
        * batch.physical_response_scale_m
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
        return _fallback_result(
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
        input_lineage={} if batch.metadata is None else batch.metadata,
    )


__all__ = ["PriorAwareGaugeConfigV1", "update_prior_aware_gauge_belief"]
