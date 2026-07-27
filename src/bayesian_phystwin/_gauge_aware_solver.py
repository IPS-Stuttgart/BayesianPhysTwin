"""Inference and selection for gauge-aware Bayesian updates."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareBeliefConfig,
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    GaugeAwareSelection,
    _block_diagonal,
    _fallback_result,
    _finite_array,
    _orthonormal_column_space,
    _positive_definite_whitener,
    _positive_semidefinite_square_root,
    _regularized_precision,
    _require,
    _student_t_weights,
    _subspace_overlap,
)


class _RegretDecision(Protocol):
    candidate_accepted: bool
    selected_value: np.ndarray
    reason: str


def _correlation_group_weights(
    group_ids: tuple[str, ...],
    reliability: np.ndarray,
    prior_nominal_probability: np.ndarray,
    composite_weight: np.ndarray,
    effective_samples_per_group: float,
    *,
    composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
) -> tuple[np.ndarray, dict[str, int]]:
    weights = np.zeros(len(group_ids), dtype=np.float64)
    counts: dict[str, int] = {}
    for group_id in group_ids:
        counts[group_id] = counts.get(group_id, 0) + 1
    for index, group_id in enumerate(group_ids):
        count = counts[group_id]
        group_scale = (
            1.0
            if composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
            else min(effective_samples_per_group, float(count)) / count
        )
        weights[index] = (
            reliability[index]
            * prior_nominal_probability[index]
            * composite_weight[index]
            * group_scale
        )
    return weights, counts


def _whiten_observations(
    target: np.ndarray,
    covariance: np.ndarray,
    designs: tuple[np.ndarray, ...],
    base_weight: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    count = len(target)
    whitened_target = np.empty((count, 3), dtype=np.float64)
    whitened_designs = tuple(
        np.empty_like(design, dtype=np.float64) for design in designs
    )
    whiteners = np.empty((count, 3, 3), dtype=np.float64)
    for index in range(count):
        whitener = _positive_definite_whitener(
            covariance[index], f"{name} covariance {index}"
        )
        whiteners[index] = whitener
        scale = np.sqrt(base_weight[index])
        whitened_target[index] = scale * (whitener @ target[index])
        for destination, design in zip(whitened_designs, designs, strict=True):
            destination[index] = scale * (whitener @ design[index])
    return whitened_target, whitened_designs, whiteners


def _query_identifiable_transform(
    state_design: np.ndarray,
    nuisance_design: np.ndarray,
    query_design: np.ndarray,
    *,
    minimum_identifiable_fraction: float,
    minimum_query_sensitivity_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    state_flat = state_design.reshape(-1, state_design.shape[2])
    nuisance_flat = (
        np.zeros((state_flat.shape[0], 0), dtype=np.float64)
        if nuisance_design.shape[2] == 0
        else nuisance_design.reshape(-1, nuisance_design.shape[2])
    )
    query_flat = query_design.reshape(-1, query_design.shape[2])
    nuisance_space = _orthonormal_column_space(nuisance_flat)
    projected = state_flat.copy()
    if nuisance_space.shape[1]:
        projected -= nuisance_space @ (nuisance_space.T @ state_flat)
    projected_norm = float(np.linalg.norm(projected))
    floor = (
        max(projected.shape)
        * np.finfo(np.float64).eps
        * max(1.0, float(np.linalg.norm(state_flat)))
    )
    if projected_norm <= floor:
        return (
            np.zeros((state_design.shape[2], 0)),
            np.zeros(0),
            np.zeros(0),
            {
                "state_nuisance_subspace_cosine": _subspace_overlap(
                    state_flat, nuisance_flat
                ),
                "projected_state_design_norm": projected_norm,
            },
        )
    _, singular_values, right_transpose = np.linalg.svd(projected, full_matrices=False)
    tolerance = max(projected.shape) * np.finfo(np.float64).eps * singular_values[0]
    candidate_count = int(np.sum(singular_values > tolerance))
    candidates = right_transpose[:candidate_count].T
    query_norms = np.asarray(
        [
            float(np.linalg.norm(query_flat @ candidates[:, index]))
            for index in range(candidate_count)
        ]
    )
    maximum_query_norm = float(np.max(query_norms, initial=0.0))
    transforms: list[np.ndarray] = []
    fractions: list[float] = []
    query_fractions: list[float] = []
    for index in range(candidate_count):
        direction = candidates[:, index]
        total_norm = float(np.linalg.norm(state_flat @ direction))
        if total_norm <= tolerance:
            continue
        identifiable_fraction = float(
            np.linalg.norm(projected @ direction) / total_norm
        )
        query_fraction = (
            query_norms[index] / maximum_query_norm if maximum_query_norm > 0.0 else 0.0
        )
        if (
            identifiable_fraction >= minimum_identifiable_fraction
            and query_fraction >= minimum_query_sensitivity_fraction
        ):
            transforms.append(direction)
            fractions.append(min(1.0, identifiable_fraction))
            query_fractions.append(min(1.0, query_fraction))
    return (
        np.column_stack(transforms)
        if transforms
        else np.zeros((state_design.shape[2], 0)),
        np.asarray(fractions),
        np.asarray(query_fractions),
        {
            "state_nuisance_subspace_cosine": _subspace_overlap(
                state_flat, nuisance_flat
            ),
            "projected_state_design_norm": projected_norm,
            "maximum_query_sensitivity_norm": maximum_query_norm,
        },
    )


def _state_prior_covariance(
    batch: GaugeAwareObservationBatch,
    cfg: GaugeAwareBeliefConfig,
) -> np.ndarray:
    state_count = batch.state_jacobian.shape[2]
    return (
        np.eye(state_count, dtype=np.float64) * cfg.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2, dtype=np.float64)
    )


def _nuisance_prior_covariance(
    batch: GaugeAwareObservationBatch,
    cfg: GaugeAwareBeliefConfig,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    gauge_count = batch.gauge_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    if gauge_count:
        blocks.append(np.asarray(batch.gauge_prior_covariance))
    if shared_count:
        blocks.append(np.eye(shared_count) * cfg.shared_bias_prior_std_m**2)
    if view_count:
        blocks.append(np.eye(view_count) * cfg.view_bias_prior_std_m**2)
    if anchor_bias_count:
        if batch.anchor_bias_prior_covariance is None:
            raise ValueError("anchor bias prior covariance is missing")
        blocks.append(np.asarray(batch.anchor_bias_prior_covariance))
    return _block_diagonal(blocks)


def _joint_prior_covariance(
    state_prior: np.ndarray,
    nuisance_prior: np.ndarray,
) -> np.ndarray:
    return _block_diagonal([state_prior, nuisance_prior])


def _assemble_full_posterior(
    *,
    state_prior_covariance: np.ndarray,
    state_prior_square_root: np.ndarray,
    state_transform: np.ndarray,
    reduced_covariance: np.ndarray,
    reduced_state_count: int,
    nuisance_count: int,
) -> np.ndarray:
    """Restore untouched prior state variance outside the inferred subspace."""

    state_count = len(state_prior_covariance)
    reduced_state_covariance = reduced_covariance[
        :reduced_state_count, :reduced_state_count
    ]
    reduced_state_nuisance = reduced_covariance[
        :reduced_state_count,
        reduced_state_count : reduced_state_count + nuisance_count,
    ]
    nuisance_covariance = reduced_covariance[
        reduced_state_count : reduced_state_count + nuisance_count,
        reduced_state_count : reduced_state_count + nuisance_count,
    ]

    whitened_state_covariance = np.eye(state_count, dtype=np.float64)
    whitened_state_covariance += (
        state_transform
        @ (reduced_state_covariance - np.eye(reduced_state_count))
        @ state_transform.T
    )
    state_covariance = (
        state_prior_square_root @ whitened_state_covariance @ state_prior_square_root.T
    )
    state_covariance = 0.5 * (state_covariance + state_covariance.T)

    full_covariance = np.zeros(
        (state_count + nuisance_count, state_count + nuisance_count),
        dtype=np.float64,
    )
    full_covariance[:state_count, :state_count] = state_covariance
    if nuisance_count:
        state_nuisance = (
            state_prior_square_root @ state_transform @ reduced_state_nuisance
        )
        full_covariance[:state_count, state_count:] = state_nuisance
        full_covariance[state_count:, :state_count] = state_nuisance.T
        full_covariance[state_count:, state_count:] = nuisance_covariance
    return 0.5 * (full_covariance + full_covariance.T)


def update_gauge_aware_belief(
    batch: GaugeAwareObservationBatch,
    *,
    config: GaugeAwareBeliefConfig | None = None,
) -> GaugeAwareBeliefResult:
    """Infer query-relevant state while preserving unsupported prior uncertainty."""

    cfg = config or GaugeAwareBeliefConfig()
    state_count = batch.state_jacobian.shape[2]
    gauge_count = batch.gauge_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    ordinary_nuisance_count = gauge_count + shared_count + view_count
    nuisance_count = ordinary_nuisance_count + anchor_bias_count

    state_prior = _state_prior_covariance(batch, cfg)
    state_prior_square_root = _positive_semidefinite_square_root(
        state_prior,
        "state prior covariance",
        eigenvalue_floor=cfg.prior_eigenvalue_floor,
    )
    nuisance_prior_covariance = _nuisance_prior_covariance(batch, cfg)
    full_prior_covariance = _joint_prior_covariance(
        state_prior, nuisance_prior_covariance
    )

    base_weight, group_counts = _correlation_group_weights(
        batch.correlation_group_ids,
        batch.prior_reliability,
        batch.prior_nominal_probability,
        batch.composite_weight,
        cfg.effective_samples_per_correlation_group,
        composite_weight_mode=batch.composite_weight_mode,
    )
    if batch.anchor_innovation_m is None:
        anchor_base_weight = np.zeros(0, dtype=np.float64)
        anchor_group_counts: dict[str, int] = {}
    else:
        if (
            batch.anchor_correlation_group_ids is None
            or batch.anchor_prior_reliability is None
            or batch.anchor_prior_nominal_probability is None
            or batch.anchor_composite_weight is None
        ):
            raise ValueError("validated anchor dependence metadata is missing")
        anchor_base_weight, anchor_group_counts = _correlation_group_weights(
            batch.anchor_correlation_group_ids,
            batch.anchor_prior_reliability,
            batch.anchor_prior_nominal_probability,
            batch.anchor_composite_weight,
            cfg.effective_samples_per_anchor_correlation_group,
            composite_weight_mode=batch.anchor_composite_weight_mode,
        )

    diagnostics: dict[str, Any] = {
        "observation_count": len(batch.innovation_m),
        "active_observation_count": int(np.sum(base_weight > 0.0)),
        "correlation_group_count": len(group_counts),
        "correlation_group_sizes": group_counts,
        "effective_observation_information_mass": float(np.sum(base_weight)),
        "anchor_count": len(anchor_base_weight),
        "active_anchor_count": int(np.sum(anchor_base_weight > 0.0)),
        "anchor_correlation_group_count": len(anchor_group_counts),
        "anchor_correlation_group_sizes": anchor_group_counts,
        "effective_anchor_information_mass": float(np.sum(anchor_base_weight)),
        "state_mode_count": state_count,
        "state_prior_rank": int(
            np.linalg.matrix_rank(
                state_prior,
                tol=cfg.prior_eigenvalue_floor,
            )
        ),
        "state_prior_trace_m2": float(np.trace(state_prior)),
        "gauge_parameter_count": gauge_count,
        "shared_bias_parameter_count": shared_count,
        "view_bias_parameter_count": view_count,
        "anchor_bias_parameter_count": anchor_bias_count,
        "query_point_count": len(batch.query_state_jacobian),
        "prior_reliability_uses_innovation": False,
        "prior_nominal_probability_uses_innovation": False,
        "association_probability_used_as_reliability": False,
        "observation_composite_weight_mode": batch.composite_weight_mode,
        "anchor_composite_weight_mode": batch.anchor_composite_weight_mode,
        "correlation_treatment": (
            "provider-final per-row observation power; no consumer recap"
            if batch.composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
            else "consumer effective-sample cap after composite weighting"
        ),
        "state_subspace_coordinates": "prior-whitened",
        "unsupported_state_prior_preserved": True,
    }
    if not np.any(base_weight > 0.0) and not np.any(anchor_base_weight > 0.0):
        return _fallback_result(
            batch,
            "no-observation-support",
            diagnostics,
            prior_covariance=full_prior_covariance,
        )

    ordinary_nuisance = np.concatenate(
        (
            batch.gauge_jacobian,
            batch.shared_bias_jacobian,
            batch.view_bias_jacobian,
            np.zeros((len(batch.innovation_m), 3, anchor_bias_count)),
        ),
        axis=2,
    )
    target_w, (state_w_original, nuisance_w), whiteners = _whiten_observations(
        batch.innovation_m,
        batch.observation_covariance_m2,
        (batch.state_jacobian, ordinary_nuisance),
        base_weight,
        name="observation",
    )
    state_w = np.einsum(
        "mcs,sk->mck",
        state_w_original,
        state_prior_square_root,
        optimize=True,
    )

    if batch.anchor_innovation_m is None:
        anchor_target_w = np.zeros((0, 3))
        anchor_state_w_original = np.zeros((0, 3, state_count))
        anchor_state_w = np.zeros((0, 3, state_count))
        anchor_nuisance_w = np.zeros((0, 3, nuisance_count))
        anchor_whiteners = np.zeros((0, 3, 3))
        raw_anchor_nuisance = np.zeros((0, 3, nuisance_count))
    else:
        anchor_bias = (
            np.zeros((len(batch.anchor_innovation_m), 3, 0))
            if batch.anchor_bias_jacobian is None
            else batch.anchor_bias_jacobian
        )
        raw_anchor_nuisance = np.concatenate(
            (
                np.zeros(
                    (
                        len(batch.anchor_innovation_m),
                        3,
                        ordinary_nuisance_count,
                    )
                ),
                anchor_bias,
            ),
            axis=2,
        )
        (
            anchor_target_w,
            (anchor_state_w_original, anchor_nuisance_w),
            anchor_whiteners,
        ) = _whiten_observations(
            batch.anchor_innovation_m,
            batch.anchor_covariance_m2,
            (batch.anchor_state_jacobian, raw_anchor_nuisance),
            anchor_base_weight,
            name="anchor",
        )
        anchor_state_w = np.einsum(
            "acs,sk->ack",
            anchor_state_w_original,
            state_prior_square_root,
            optimize=True,
        )

    nuisance_prior_square_root = _positive_semidefinite_square_root(
        nuisance_prior_covariance,
        "nuisance prior covariance",
        eigenvalue_floor=cfg.prior_eigenvalue_floor,
    )
    scaled_nuisance_w = np.einsum(
        "mcn,nk->mck", nuisance_w, nuisance_prior_square_root, optimize=True
    )
    scaled_anchor_nuisance_w = np.einsum(
        "acn,nk->ack",
        anchor_nuisance_w,
        nuisance_prior_square_root,
        optimize=True,
    )
    query_state_whitened = np.einsum(
        "qcs,sk->qck",
        batch.query_state_jacobian,
        state_prior_square_root,
        optimize=True,
    )
    identifiability_state = np.concatenate((state_w, anchor_state_w), axis=0)
    identifiability_nuisance = np.concatenate(
        (scaled_nuisance_w, scaled_anchor_nuisance_w),
        axis=0,
    )
    transform, fractions, query_fractions, identifiability = (
        _query_identifiable_transform(
            identifiability_state,
            identifiability_nuisance,
            query_state_whitened,
            minimum_identifiable_fraction=cfg.minimum_identifiable_fraction,
            minimum_query_sensitivity_fraction=(cfg.minimum_query_sensitivity_fraction),
        )
    )
    diagnostics.update(identifiability)
    diagnostics["identifiable_query_state_mode_count"] = transform.shape[1]
    if transform.shape[1] == 0:
        return _fallback_result(
            batch,
            "no-identifiable-query-state",
            diagnostics,
            prior_covariance=full_prior_covariance,
        )

    reduced_state_count = transform.shape[1]
    observation_design = np.concatenate(
        (
            np.einsum("mcs,sr->mcr", state_w, transform, optimize=True),
            nuisance_w,
        ),
        axis=2,
    )
    anchor_design = np.concatenate(
        (
            np.einsum(
                "acs,sr->acr",
                anchor_state_w,
                transform,
                optimize=True,
            ),
            anchor_nuisance_w,
        ),
        axis=2,
    )
    joint_dimension = reduced_state_count + nuisance_count
    prior_blocks = [np.eye(reduced_state_count)]
    if nuisance_count:
        prior_blocks.append(nuisance_prior_covariance)
    prior_covariance = _block_diagonal(prior_blocks)
    prior_precision = _regularized_precision(
        prior_covariance,
        "joint prior covariance",
        eigenvalue_floor=cfg.prior_eigenvalue_floor,
    )
    _require(
        prior_precision.shape == (joint_dimension, joint_dimension),
        "joint prior precision has changed shape",
    )

    active_observation = base_weight > 0.0
    active_anchor = anchor_base_weight > 0.0
    robust = active_observation.astype(np.float64)
    anchor_robust = active_anchor.astype(np.float64)
    solution = np.zeros(joint_dimension, dtype=np.float64)

    def posterior_system() -> tuple[np.ndarray, np.ndarray]:
        normal = prior_precision.copy()
        right = np.zeros(joint_dimension, dtype=np.float64)
        normal += np.einsum(
            "m,mci,mcj->ij",
            robust,
            observation_design,
            observation_design,
            optimize=True,
        )
        right += np.einsum(
            "m,mci,mc->i",
            robust,
            observation_design,
            target_w,
            optimize=True,
        )
        if len(anchor_target_w):
            normal += np.einsum(
                "a,aci,acj->ij",
                anchor_robust,
                anchor_design,
                anchor_design,
                optimize=True,
            )
            right += np.einsum(
                "a,aci,ac->i",
                anchor_robust,
                anchor_design,
                anchor_target_w,
                optimize=True,
            )
        return normal, right

    state_mapping = state_prior_square_root @ transform
    raw_state_reduced = np.einsum(
        "mcs,sr->mcr",
        batch.state_jacobian,
        state_mapping,
        optimize=True,
    )
    raw_observation_design = np.concatenate(
        (raw_state_reduced, ordinary_nuisance), axis=2
    )
    raw_anchor_design = (
        np.zeros((0, 3, joint_dimension))
        if batch.anchor_state_jacobian is None
        else np.concatenate(
            (
                np.einsum(
                    "acs,sr->acr",
                    batch.anchor_state_jacobian,
                    state_mapping,
                    optimize=True,
                ),
                raw_anchor_nuisance,
            ),
            axis=2,
        )
    )

    normal = prior_precision.copy()
    iteration = 0
    for iteration in range(cfg.maximum_iterations):
        previous = solution.copy()
        normal, right = posterior_system()
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
                prior_covariance=full_prior_covariance,
            )
        try:
            solution = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            return _fallback_result(
                batch,
                "singular-posterior",
                diagnostics,
                prior_covariance=full_prior_covariance,
            )
        residual = batch.innovation_m - np.einsum(
            "mci,i->mc", raw_observation_design, solution, optimize=True
        )
        whitened_residual = np.einsum("mij,mj->mi", whiteners, residual, optimize=True)
        robust = _student_t_weights(
            np.sum(np.square(whitened_residual), axis=1),
            dimension=3,
            degrees_of_freedom=cfg.degrees_of_freedom,
            minimum=cfg.minimum_robust_weight,
        )
        robust[~active_observation] = 0.0
        if batch.anchor_innovation_m is not None:
            anchor_residual = batch.anchor_innovation_m - np.einsum(
                "aci,i->ac", raw_anchor_design, solution, optimize=True
            )
            whitened_anchor_residual = np.einsum(
                "aij,aj->ai", anchor_whiteners, anchor_residual, optimize=True
            )
            anchor_robust = _student_t_weights(
                np.sum(np.square(whitened_anchor_residual), axis=1),
                dimension=3,
                degrees_of_freedom=cfg.degrees_of_freedom,
                minimum=cfg.minimum_robust_weight,
            )
            anchor_robust[~active_anchor] = 0.0
        if np.linalg.norm(solution - previous) <= cfg.convergence_tolerance:
            break

    normal, right = posterior_system()
    try:
        solution = np.linalg.solve(normal, right)
        reduced_covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return _fallback_result(
            batch,
            "singular-posterior",
            diagnostics,
            prior_covariance=full_prior_covariance,
        )

    reduced_state_solution = solution[:reduced_state_count]
    whitened_state_solution = transform @ reduced_state_solution
    state_coefficients = state_prior_square_root @ whitened_state_solution
    nuisance_solution = solution[reduced_state_count:]
    full_covariance = _assemble_full_posterior(
        state_prior_covariance=state_prior,
        state_prior_square_root=state_prior_square_root,
        state_transform=transform,
        reduced_covariance=reduced_covariance,
        reduced_state_count=reduced_state_count,
        nuisance_count=nuisance_count,
    )

    gauge_slice = slice(0, gauge_count)
    shared_slice = slice(gauge_slice.stop, gauge_slice.stop + shared_count)
    view_slice = slice(shared_slice.stop, shared_slice.stop + view_count)
    anchor_bias_slice = slice(view_slice.stop, view_slice.stop + anchor_bias_count)
    query_update = np.einsum(
        "qcs,s->qc",
        batch.query_state_jacobian,
        state_coefficients,
        optimize=True,
    )
    query_update_norm = np.linalg.norm(query_update, axis=1)
    maximum_query_update = float(np.max(query_update_norm, initial=0.0))
    relative_limit = (
        cfg.maximum_update_to_physical_response_ratio * batch.physical_response_scale_m
    )
    update_limit = min(cfg.maximum_state_update_m, relative_limit)
    diagnostics.update(
        {
            "iterations": iteration + 1,
            "condition_number": float(np.linalg.cond(normal)),
            "minimum_robust_weight": (
                float(np.min(robust[active_observation]))
                if np.any(active_observation)
                else 1.0
            ),
            "downweighted_observation_fraction": (
                float(np.mean(robust[active_observation] < 1.0))
                if np.any(active_observation)
                else 0.0
            ),
            "minimum_anchor_robust_weight": (
                float(np.min(anchor_robust[active_anchor]))
                if np.any(active_anchor)
                else 1.0
            ),
            "downweighted_anchor_fraction": (
                float(np.mean(anchor_robust[active_anchor] < 1.0))
                if np.any(active_anchor)
                else 0.0
            ),
            "maximum_query_state_update_m": maximum_query_update,
            "absolute_state_update_limit_m": cfg.maximum_state_update_m,
            "physical_response_relative_limit_m": relative_limit,
            "active_state_update_limit_m": update_limit,
            "state_posterior_trace_m2": float(
                np.trace(full_covariance[:state_count, :state_count])
            ),
            "retained_state_prior_trace_m2": float(
                np.trace(
                    state_prior_square_root
                    @ transform
                    @ transform.T
                    @ state_prior_square_root.T
                )
            ),
        }
    )
    if (
        not np.all(np.isfinite(solution))
        or not np.all(np.isfinite(full_covariance))
        or maximum_query_update > update_limit
    ):
        return _fallback_result(
            batch,
            "implausible-state-update",
            diagnostics,
            prior_covariance=full_prior_covariance,
        )

    if nuisance_count:
        cross = full_covariance[:state_count, state_count:]
        state_variance = np.diag(full_covariance[:state_count, :state_count])[:, None]
        nuisance_variance = np.diag(full_covariance[state_count:, state_count:])[None]
        denominator = np.sqrt(np.maximum(state_variance * nuisance_variance, 1e-30))
        diagnostics["maximum_state_nuisance_posterior_correlation"] = float(
            np.max(np.abs(cross / denominator), initial=0.0)
        )
    else:
        diagnostics["maximum_state_nuisance_posterior_correlation"] = 0.0

    return GaugeAwareBeliefResult(
        inference_admissible=True,
        reason="inference-admissible",
        state_coefficients=state_coefficients,
        gauge_delta=nuisance_solution[gauge_slice],
        shared_bias_coefficients=nuisance_solution[shared_slice],
        view_bias_coefficients=nuisance_solution[view_slice],
        anchor_bias_coefficients=nuisance_solution[anchor_bias_slice],
        posterior_covariance=full_covariance,
        identifiable_state_transform=state_mapping,
        identifiable_fractions=fractions,
        query_sensitivity_fractions=query_fractions,
        robust_weights=robust,
        anchor_robust_weights=anchor_robust,
        diagnostics=diagnostics,
        input_lineage=batch.metadata,
    )


def decode_gauge_aware_query(
    result: GaugeAwareBeliefResult, query_state_jacobian: np.ndarray
) -> np.ndarray:
    """Decode the numerically admissible state correction in query space."""

    query = _finite_array(query_state_jacobian, "query_state_jacobian", 3)
    _require(
        query.shape[1:] == (3, len(result.state_coefficients)),
        "query state Jacobian has changed shape",
    )
    if not result.inference_admissible:
        return np.zeros(query.shape[:2], dtype=np.float64)
    return np.einsum("qcs,s->qc", query, result.state_coefficients, optimize=True)


def _same_array_bytes(first: np.ndarray, second: np.ndarray) -> bool:
    left = np.asarray(first)
    right = np.asarray(second)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def select_gauge_aware_candidate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    result: GaugeAwareBeliefResult,
    *,
    regret_decision: _RegretDecision | None = None,
) -> GaugeAwareSelection:
    """Require both inference admissibility and a baseline-relative regret guard."""

    baseline_input = np.asarray(baseline)
    candidate_input = np.asarray(candidate)
    _require(candidate_input.shape == baseline_input.shape, "candidate shape changed")
    _require(
        np.all(np.isfinite(baseline_input)) and np.all(np.isfinite(candidate_input)),
        "candidate values must be finite",
    )

    guard_present = regret_decision is not None
    guard_accepted = bool(
        guard_present
        and regret_decision is not None
        and regret_decision.candidate_accepted
    )
    if regret_decision is not None:
        expected = candidate_input if guard_accepted else baseline_input
        _require(
            _same_array_bytes(regret_decision.selected_value, expected),
            "regret decision is not bound to the supplied baseline and candidate",
        )

    accepted = result.inference_admissible and guard_accepted
    if not result.inference_admissible:
        reason = f"inference-{result.reason}-exact-baseline-fallback"
    elif not guard_present:
        reason = "missing-regret-guard-exact-baseline-fallback"
    elif not guard_accepted:
        reason = "regret-guard-exact-baseline-fallback"
    else:
        reason = "candidate-accepted"
    selected = candidate_input.copy() if accepted else baseline_input.copy()
    if not accepted and not _same_array_bytes(selected, baseline_input):
        raise AssertionError("gauge-aware fallback changed baseline bytes")
    return GaugeAwareSelection(
        candidate_accepted=accepted,
        inference_admissible=result.inference_admissible,
        regret_guard_present=guard_present,
        regret_guard_accepted=guard_accepted,
        reason=reason,
        selected_value=selected,
    )
