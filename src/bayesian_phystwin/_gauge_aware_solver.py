"""Inference and selection for gauge-aware Bayesian updates."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._gauge_aware_contracts import (
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


def _correlation_group_weights(
    group_ids: tuple[str, ...],
    reliability: np.ndarray,
    effective_samples_per_group: float,
) -> tuple[np.ndarray, dict[str, int]]:
    weights = np.zeros(len(group_ids), dtype=np.float64)
    counts: dict[str, int] = {}
    for group_id in group_ids:
        counts[group_id] = counts.get(group_id, 0) + 1
    for index, group_id in enumerate(group_ids):
        count = counts[group_id]
        group_scale = min(effective_samples_per_group, float(count)) / count
        weights[index] = reliability[index] * group_scale
    return weights, counts


def _whiten_observations(
    target: np.ndarray,
    covariance: np.ndarray,
    designs: tuple[np.ndarray, ...],
    base_weight: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    count = len(target)
    whitened_target = np.empty((count, 3), dtype=np.float64)
    whitened_designs = tuple(
        np.empty_like(design, dtype=np.float64) for design in designs
    )
    whiteners = np.empty((count, 3, 3), dtype=np.float64)
    for index in range(count):
        whitener = _positive_definite_whitener(
            covariance[index], f"observation covariance {index}"
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
    _, singular_values, right_transpose = np.linalg.svd(
        projected, full_matrices=False
    )
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
        identifiable_fraction = float(np.linalg.norm(projected @ direction) / total_norm)
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


def update_gauge_aware_belief(
    batch: GaugeAwareObservationBatch,
    *,
    config: GaugeAwareBeliefConfig | None = None,
) -> GaugeAwareBeliefResult:
    """Infer query-relevant physical state separately from gauge and camera bias."""

    cfg = config or GaugeAwareBeliefConfig()
    state_count = batch.state_jacobian.shape[2]
    gauge_count = batch.gauge_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    nuisance_count = gauge_count + shared_count + view_count
    base_weight, group_counts = _correlation_group_weights(
        batch.correlation_group_ids,
        batch.prior_reliability,
        cfg.effective_samples_per_correlation_group,
    )
    diagnostics: dict[str, Any] = {
        "observation_count": len(batch.innovation_m),
        "active_observation_count": int(np.sum(base_weight > 0.0)),
        "correlation_group_count": len(group_counts),
        "correlation_group_sizes": group_counts,
        "effective_observation_information_mass": float(np.sum(base_weight)),
        "state_mode_count": state_count,
        "gauge_parameter_count": gauge_count,
        "shared_bias_parameter_count": shared_count,
        "view_bias_parameter_count": view_count,
        "query_point_count": len(batch.query_state_jacobian),
        "independent_anchor_count": (
            0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
        ),
        "prior_reliability_uses_innovation": False,
        "correlation_treatment": "effective-sample cap within declared groups",
    }
    if not np.any(base_weight > 0.0):
        return _fallback_result(batch, "no-observation-support", diagnostics)

    nuisance = np.concatenate(
        (
            batch.gauge_jacobian,
            batch.shared_bias_jacobian,
            batch.view_bias_jacobian,
        ),
        axis=2,
    )
    target_w, (state_w, nuisance_w), whiteners = _whiten_observations(
        batch.innovation_m,
        batch.observation_covariance_m2,
        (batch.state_jacobian, nuisance),
        base_weight,
    )

    if batch.anchor_innovation_m is None:
        anchor_target_w = np.zeros((0, 3))
        anchor_state_w = np.zeros((0, 3, state_count))
        anchor_whiteners = np.zeros((0, 3, 3))
    else:
        anchor_weight = np.ones(len(batch.anchor_innovation_m))
        zero_nuisance = np.zeros(
            (len(batch.anchor_innovation_m), 3, nuisance_count)
        )
        anchor_target_w, (anchor_state_w, _), anchor_whiteners = _whiten_observations(
            batch.anchor_innovation_m,
            batch.anchor_covariance_m2,
            (batch.anchor_state_jacobian, zero_nuisance),
            anchor_weight,
        )

    nuisance_prior_blocks: list[np.ndarray] = []
    if gauge_count:
        nuisance_prior_blocks.append(np.asarray(batch.gauge_prior_covariance))
    if shared_count:
        nuisance_prior_blocks.append(
            np.eye(shared_count) * cfg.shared_bias_prior_std_m**2
        )
    if view_count:
        nuisance_prior_blocks.append(
            np.eye(view_count) * cfg.view_bias_prior_std_m**2
        )
    nuisance_prior_covariance = _block_diagonal(nuisance_prior_blocks)
    nuisance_prior_sqrt = _positive_semidefinite_square_root(
        nuisance_prior_covariance,
        "nuisance prior covariance",
        eigenvalue_floor=cfg.prior_eigenvalue_floor,
    )
    scaled_nuisance_w = np.einsum(
        "mcn,nk->mck", nuisance_w, nuisance_prior_sqrt, optimize=True
    )
    identifiability_state = np.concatenate((state_w, anchor_state_w), axis=0)
    identifiability_nuisance = np.concatenate(
        (
            scaled_nuisance_w,
            np.zeros((len(anchor_state_w), 3, nuisance_count)),
        ),
        axis=0,
    )
    transform, fractions, query_fractions, identifiability = (
        _query_identifiable_transform(
            identifiability_state,
            identifiability_nuisance,
            batch.query_state_jacobian,
            minimum_identifiable_fraction=cfg.minimum_identifiable_fraction,
            minimum_query_sensitivity_fraction=(
                cfg.minimum_query_sensitivity_fraction
            ),
        )
    )
    diagnostics.update(identifiability)
    diagnostics["identifiable_query_state_mode_count"] = transform.shape[1]
    if transform.shape[1] == 0:
        return _fallback_result(
            batch, "no-identifiable-query-state", diagnostics
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
            np.einsum("acs,sr->acr", anchor_state_w, transform, optimize=True),
            np.zeros((len(anchor_state_w), 3, nuisance_count)),
        ),
        axis=2,
    )
    joint_dimension = reduced_state_count + nuisance_count

    state_prior = (
        np.eye(state_count) * cfg.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2)
    )
    reduced_state_prior = transform.T @ state_prior @ transform
    prior_blocks = [reduced_state_prior]
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
    robust = active_observation.astype(np.float64)
    anchor_robust = np.ones(len(anchor_target_w), dtype=np.float64)
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

    raw_state_reduced = np.einsum(
        "mcs,sr->mcr", batch.state_jacobian, transform, optimize=True
    )
    raw_observation_design = np.concatenate(
        (raw_state_reduced, nuisance), axis=2
    )
    raw_anchor_design = (
        np.zeros((0, 3, joint_dimension))
        if batch.anchor_state_jacobian is None
        else np.concatenate(
            (
                np.einsum(
                    "acs,sr->acr",
                    batch.anchor_state_jacobian,
                    transform,
                    optimize=True,
                ),
                np.zeros((len(batch.anchor_state_jacobian), 3, nuisance_count)),
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
        if not np.isfinite(condition_number) or condition_number > cfg.maximum_condition_number:
            diagnostics["condition_number"] = condition_number
            return _fallback_result(batch, "ill-conditioned-posterior", diagnostics)
        try:
            solution = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            return _fallback_result(batch, "singular-posterior", diagnostics)
        residual = batch.innovation_m - np.einsum(
            "mci,i->mc", raw_observation_design, solution, optimize=True
        )
        whitened_residual = np.einsum(
            "mij,mj->mi", whiteners, residual, optimize=True
        )
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
        if np.linalg.norm(solution - previous) <= cfg.convergence_tolerance:
            break

    normal, right = posterior_system()
    solution = np.linalg.solve(normal, right)
    reduced_covariance = np.linalg.inv(normal)
    full_dimension = state_count + nuisance_count
    mapping = np.zeros((full_dimension, joint_dimension), dtype=np.float64)
    mapping[:state_count, :reduced_state_count] = transform
    if nuisance_count:
        mapping[state_count:, reduced_state_count:] = np.eye(nuisance_count)
    full_solution = mapping @ solution
    full_covariance = mapping @ reduced_covariance @ mapping.T
    state_coefficients = full_solution[:state_count]
    gauge_slice = slice(state_count, state_count + gauge_count)
    shared_slice = slice(gauge_slice.stop, gauge_slice.stop + shared_count)
    view_slice = slice(shared_slice.stop, shared_slice.stop + view_count)
    query_update = np.einsum(
        "qcs,s->qc",
        batch.query_state_jacobian,
        state_coefficients,
        optimize=True,
    )
    query_update_norm = np.linalg.norm(query_update, axis=1)
    maximum_query_update = float(np.max(query_update_norm, initial=0.0))
    relative_limit = (
        cfg.maximum_update_to_physical_response_ratio
        * batch.physical_response_scale_m
    )
    update_limit = min(cfg.maximum_state_update_m, relative_limit)
    diagnostics.update(
        {
            "iterations": iteration + 1,
            "condition_number": float(np.linalg.cond(normal)),
            "minimum_robust_weight": float(np.min(robust)),
            "downweighted_observation_fraction": float(np.mean(robust < 1.0)),
            "maximum_query_state_update_m": maximum_query_update,
            "absolute_state_update_limit_m": cfg.maximum_state_update_m,
            "physical_response_relative_limit_m": relative_limit,
            "active_state_update_limit_m": update_limit,
        }
    )
    if not np.all(np.isfinite(full_solution)) or maximum_query_update > update_limit:
        return _fallback_result(batch, "implausible-state-update", diagnostics)

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
        accepted=True,
        reason="accepted",
        state_coefficients=state_coefficients,
        gauge_delta=full_solution[gauge_slice],
        shared_bias_coefficients=full_solution[shared_slice],
        view_bias_coefficients=full_solution[view_slice],
        posterior_covariance=full_covariance,
        identifiable_state_transform=transform,
        identifiable_fractions=fractions,
        query_sensitivity_fractions=query_fractions,
        robust_weights=robust,
        anchor_robust_weights=anchor_robust,
        diagnostics=diagnostics,
    )


def decode_gauge_aware_query(
    result: GaugeAwareBeliefResult, query_state_jacobian: np.ndarray
) -> np.ndarray:
    """Decode the accepted state correction in a requested query space."""

    query = _finite_array(query_state_jacobian, "query_state_jacobian", 3)
    _require(
        query.shape[1:] == (3, len(result.state_coefficients)),
        "query state Jacobian has changed shape",
    )
    if not result.accepted:
        return np.zeros(query.shape[:2], dtype=np.float64)
    return np.einsum(
        "qcs,s->qc", query, result.state_coefficients, optimize=True
    )


def select_gauge_aware_candidate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    result: GaugeAwareBeliefResult,
) -> GaugeAwareSelection:
    """Select the candidate or return the baseline byte for byte on rejection."""

    baseline_input = np.asarray(baseline)
    candidate_input = np.asarray(candidate)
    _require(candidate_input.shape == baseline_input.shape, "candidate shape changed")
    selected = candidate_input.copy() if result.accepted else baseline_input.copy()
    if not result.accepted and selected.tobytes() != baseline_input.tobytes():
        raise AssertionError("gauge-aware fallback changed baseline bytes")
    return GaugeAwareSelection(
        candidate_accepted=result.accepted,
        reason="candidate-accepted" if result.accepted else "exact-baseline-fallback",
        selected_value=selected,
    )
