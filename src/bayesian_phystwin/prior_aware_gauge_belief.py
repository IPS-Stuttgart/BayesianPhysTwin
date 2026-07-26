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
    _mixture_precision,
    _prior_aware_basis,
    _prior_covariances,
    _whiten,
)


def update_prior_aware_gauge_belief(
    batch: GaugeAwareObservationBatch,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> GaugeAwareBeliefResult:
    """Infer state while conditioning identifiability on nuisance priors."""

    cfg = config or PriorAwareGaugeConfigV1()
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
    anchor_state = (
        np.zeros((0, 3, state_count))
        if batch.anchor_state_jacobian is None
        else np.asarray(batch.anchor_state_jacobian)
    )
    anchor_bias = (
        np.zeros((anchor_count, 3, anchor_bias_count))
        if batch.anchor_bias_jacobian is None
        else np.asarray(batch.anchor_bias_jacobian)
    )
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
            batch.anchor_innovation_m,
            batch.anchor_covariance_m2,
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
    ) = _group_layout(
        batch.correlation_group_ids,
        batch.prior_reliability,
        batch.prior_nominal_probability,
        batch.composite_weight,
        cfg.effective_samples_per_correlation_group,
    )
    if anchor_count:
        anchor_groups, anchor_indices, anchor_base, anchor_prior = _group_layout(
            batch.anchor_correlation_group_ids,
            batch.anchor_prior_reliability,
            batch.anchor_prior_nominal_probability,
            batch.anchor_composite_weight,
            cfg.effective_samples_per_anchor_correlation_group,
        )
    else:
        anchor_groups, anchor_indices = (), ()
        anchor_base = np.zeros(0)
        anchor_prior = np.zeros(0)
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
    state_mapping, identifiable, query_fraction, basis_diagnostics = _prior_aware_basis(
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
    diagnostics: dict[str, Any] = {
        "identifiability_mode": "prior-aware-schur-v1",
        "robust_likelihood": "grouped nominal/outlier Student-t mixture",
        "posterior_covariance_kind": "working Laplace/IRLS covariance",
        "prior_nominal_probability_used_inside_mixture": True,
        "association_probability_used_as_reliability": False,
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
    solution = np.zeros(retained + nuisance_count)
    observation_precision = expected_observation.copy()
    anchor_precision = expected_anchor.copy()
    observation_responsibility = observation_prior.copy()
    anchor_responsibility = anchor_prior.copy()

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
        return normal, right

    for iteration in range(cfg.maximum_iterations):
        previous = solution.copy()
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
            solution = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            return _fallback_result(
                batch,
                "singular-posterior",
                diagnostics,
                prior_covariance=full_prior,
            )
        residual = batch.innovation_m - np.einsum(
            "mci,i->mc", raw_observation_design, solution
        )
        white_residual = np.einsum("mij,mj->mi", whiteners, residual)
        for position, selected in enumerate(observation_indices):
            (
                observation_precision[position],
                observation_responsibility[position],
            ) = _mixture_precision(
                float(np.sum(np.square(white_residual[selected]))),
                3 * len(selected),
                observation_prior[position],
                cfg,
            )
        if anchor_count:
            residual = batch.anchor_innovation_m - np.einsum(
                "aci,i->ac", raw_anchor_design, solution
            )
            white_residual = np.einsum(
                "aij,aj->ai", anchor_whiteners, residual
            )
            for position, selected in enumerate(anchor_indices):
                (
                    anchor_precision[position],
                    anchor_responsibility[position],
                ) = _mixture_precision(
                    float(np.sum(np.square(white_residual[selected]))),
                    3 * len(selected),
                    anchor_prior[position],
                    cfg,
                )
        if np.linalg.norm(solution - previous) <= cfg.convergence_tolerance:
            break
    normal, right = system()
    try:
        solution = np.linalg.solve(normal, right)
        reduced_covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return _fallback_result(
            batch,
            "singular-final-posterior",
            diagnostics,
            prior_covariance=full_prior,
        )
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
            "iterations": iteration + 1,
            "condition_number": float(np.linalg.cond(normal)),
            "maximum_query_state_update_m": maximum_update,
            "active_state_update_limit_m": update_limit,
            "observation_group_ids": list(observation_groups),
            "observation_group_posterior_nominal_probability": (
                observation_responsibility.tolist()
            ),
            "anchor_group_ids": list(anchor_groups),
            "anchor_group_posterior_nominal_probability": (
                anchor_responsibility.tolist()
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
        input_lineage=batch.metadata,
    )


__all__ = ["PriorAwareGaugeConfigV1", "update_prior_aware_gauge_belief"]
