"""Robust recursive filtering for graph-modal discrepancy dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._graph_dynamic_discrepancy_common import (
    GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
    GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
    GRAPH_DYNAMIC_DISCREPANCY_VERSION,
    GraphDynamicDiscrepancyConfigV1,
    _covariance_from_precision,
    _positive_definite_precision,
    _positive_semidefinite_root,
    _real,
    _require,
    _state_array,
    _symmetric,
    _transition_and_noise,
    _validate_graph_basis,
)
from ._graph_dynamic_discrepancy_contract import (
    GraphDynamicDiscrepancyBeliefV1,
)


def _covariance_array(
    supplied: np.ndarray | None,
    *,
    frame_count: int,
    node_count: int,
    default_std_m: float,
) -> np.ndarray:
    if supplied is None:
        covariance = np.broadcast_to(
            np.eye(3, dtype=np.float64) * default_std_m**2,
            (frame_count, node_count, 3, 3),
        ).copy()
    else:
        raw = np.asarray(supplied)
        _require(
            raw.dtype.kind in {"i", "u", "f"},
            "observation covariance must be real numeric",
        )
        value = np.asarray(raw, dtype=np.float64)
        if value.shape == (3, 3):
            covariance = np.broadcast_to(
                value,
                (frame_count, node_count, 3, 3),
            ).copy()
        elif value.shape == (node_count, 3, 3):
            covariance = np.broadcast_to(
                value[None],
                (frame_count, node_count, 3, 3),
            ).copy()
        elif value.shape == (frame_count, node_count, 3, 3):
            covariance = value.copy()
        else:
            raise ValueError(
                "observation_covariance_m2 must have shape (3, 3), "
                "(node, 3, 3), or (frame, node, 3, 3)"
            )
    _require(
        np.all(np.isfinite(covariance)),
        "observation covariance must be finite",
    )
    _require(
        np.allclose(
            covariance,
            np.swapaxes(covariance, -1, -2),
            atol=1e-12,
            rtol=1e-12,
        ),
        "observation covariance must be symmetric",
    )
    minimum = np.linalg.eigvalsh(covariance)[..., 0]
    _require(
        np.all(minimum > 0.0),
        "observation covariance must be positive definite",
    )
    return covariance


def _frame_group_labels(
    supplied: np.ndarray | None,
    *,
    frame_count: int,
    node_count: int,
) -> np.ndarray:
    if supplied is None:
        result = np.empty((frame_count, node_count), dtype=object)
        for frame in range(frame_count):
            result[frame] = f"frame-{frame}"
        return result
    value = np.asarray(supplied, dtype=object)
    _require(
        value.shape == (frame_count, node_count),
        "correlation_group_ids shape changed",
    )
    result = np.empty_like(value)
    for index in np.ndindex(value.shape):
        label = str(value[index])
        _require(label, "correlation group labels must be nonempty")
        result[index] = label
    return result


def _frame_update(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    residual_m: np.ndarray,
    usable: np.ndarray,
    graph_basis: np.ndarray,
    observation_covariance_m2: np.ndarray,
    reliability: np.ndarray,
    group_labels: np.ndarray,
    config: GraphDynamicDiscrepancyConfigV1,
) -> tuple[np.ndarray, np.ndarray, bool, str, dict[str, Any]]:
    selected = np.flatnonzero(usable & (reliability > 0.0))
    diagnostics: dict[str, Any] = {
        "usable_node_count": int(len(selected)),
        "mixture_fixed_point_converged": False,
        "posterior_covariance_kind": (
            "working-gauss-newton-irls-not-exact-mixture-hessian"
        ),
    }
    if not len(selected):
        return (
            prior_mean.copy(),
            prior_covariance.copy(),
            False,
            "no-observation-support",
            diagnostics,
        )
    dimension = len(prior_mean)
    position_dimension = dimension // 2
    identity3 = np.eye(3, dtype=np.float64)
    designs: dict[int, np.ndarray] = {}
    precisions: dict[int, np.ndarray] = {}
    for node in selected:
        design = np.zeros((3, dimension), dtype=np.float64)
        design[:, :position_dimension] = np.kron(
            graph_basis[node : node + 1],
            identity3,
        )
        designs[int(node)] = design
        precisions[int(node)] = _positive_definite_precision(
            observation_covariance_m2[node],
            name=f"observation covariance for node {node}",
        )
    prior_root = _positive_semidefinite_root(
        prior_covariance,
        name="predicted state covariance",
    )
    diagnostics["predicted_covariance_rank"] = int(prior_root.shape[1])
    if not prior_root.shape[1]:
        return (
            prior_mean.copy(),
            prior_covariance.copy(),
            False,
            "deterministic-predicted-belief",
            diagnostics,
        )
    reduced_designs = {
        node: design @ prior_root for node, design in designs.items()
    }
    centered_targets = {
        node: residual_m[node] - designs[node] @ prior_mean
        for node in designs
    }
    ordered_groups = tuple(
        dict.fromkeys(str(group_labels[node]) for node in selected)
    )
    group_nodes = tuple(
        np.asarray(
            [node for node in selected if str(group_labels[node]) == label],
            dtype=np.int64,
        )
        for label in ordered_groups
    )
    group_power = np.asarray(
        [
            min(
                config.effective_samples_per_correlation_group,
                float(len(nodes)),
            )
            / len(nodes)
            for nodes in group_nodes
        ],
        dtype=np.float64,
    )

    def expanded_state(reduced: np.ndarray) -> np.ndarray:
        return prior_mean + prior_root @ reduced

    def robust_weights(reduced: np.ndarray) -> np.ndarray:
        current = expanded_state(reduced)
        weights = np.empty(len(group_nodes), dtype=np.float64)
        for position, nodes in enumerate(group_nodes):
            squared_mahalanobis = 0.0
            for raw_node in nodes:
                node = int(raw_node)
                innovation = residual_m[node] - designs[node] @ current
                row_reliability = float(reliability[node])
                squared_mahalanobis += row_reliability * float(
                    innovation @ precisions[node] @ innovation
                )
            effective_dimension = 3.0 * len(nodes)
            covariance_degrees = config.degrees_of_freedom - 2.0
            weight = (
                covariance_degrees + effective_dimension
            ) / (covariance_degrees + squared_mahalanobis)
            weights[position] = np.clip(
                weight,
                config.minimum_robust_weight,
                1.0,
            )
        return weights

    def system(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reduced_dimension = prior_root.shape[1]
        normal = np.eye(reduced_dimension, dtype=np.float64)
        right = np.zeros(reduced_dimension, dtype=np.float64)
        for position, nodes in enumerate(group_nodes):
            for raw_node in nodes:
                node = int(raw_node)
                row_weight = (
                    float(reliability[node])
                    * group_power[position]
                    * weights[position]
                )
                design = reduced_designs[node]
                precision = precisions[node]
                normal += row_weight * design.T @ precision @ design
                right += (
                    row_weight
                    * design.T
                    @ precision
                    @ centered_targets[node]
                )
        return _symmetric(normal), right

    current = np.zeros(prior_root.shape[1], dtype=np.float64)
    weights = robust_weights(current)
    final_normal = np.eye(prior_root.shape[1], dtype=np.float64)
    condition_number = float("inf")
    weight_delta = float("inf")
    solution_delta = float("inf")
    stationarity_norm = float("inf")
    converged = False
    iteration_count = 0
    for iteration in range(config.maximum_iterations):
        iteration_count = iteration + 1
        normal, right = system(weights)
        condition_number = float(np.linalg.cond(normal))
        if (
            not np.isfinite(condition_number)
            or condition_number > config.maximum_condition_number
        ):
            diagnostics["condition_number"] = condition_number
            return (
                prior_mean.copy(),
                prior_covariance.copy(),
                False,
                "ill-conditioned-posterior",
                diagnostics,
            )
        try:
            factor = np.linalg.cholesky(normal)
            candidate = np.linalg.solve(
                factor.T,
                np.linalg.solve(factor, right),
            )
        except np.linalg.LinAlgError:
            return (
                prior_mean.copy(),
                prior_covariance.copy(),
                False,
                "singular-posterior",
                diagnostics,
            )
        refreshed = robust_weights(candidate)
        refreshed_normal, refreshed_right = system(refreshed)
        solution_delta = float(np.linalg.norm(candidate - current))
        weight_delta = float(
            np.max(np.abs(refreshed - weights), initial=0.0)
        )
        stationarity_norm = float(
            np.linalg.norm(refreshed_normal @ candidate - refreshed_right)
        )
        stationarity_scale = 1.0 + float(np.linalg.norm(refreshed_right))
        current = candidate
        weights = refreshed
        final_normal = refreshed_normal
        if (
            weight_delta <= config.convergence_tolerance
            and stationarity_norm
            <= config.convergence_tolerance * stationarity_scale
        ):
            converged = True
            break
    diagnostics.update(
        {
            "iterations": iteration_count,
            "mixture_fixed_point_converged": converged,
            "mixture_solution_delta": solution_delta,
            "mixture_weight_delta": weight_delta,
            "mixture_stationarity_norm": stationarity_norm,
            "condition_number": condition_number,
            "correlation_group_ids": list(ordered_groups),
            "correlation_group_power": group_power.tolist(),
            "correlation_group_robust_weight": weights.tolist(),
            "student_t_covariance_parameterization": True,
            "group_power_enters_likelihood_once": True,
        }
    )
    if not converged:
        return (
            prior_mean.copy(),
            prior_covariance.copy(),
            False,
            "robust-fixed-point-not-converged",
            diagnostics,
        )
    final_condition_number = float(np.linalg.cond(final_normal))
    diagnostics["final_condition_number"] = final_condition_number
    if (
        not np.isfinite(final_condition_number)
        or final_condition_number > config.maximum_condition_number
    ):
        return (
            prior_mean.copy(),
            prior_covariance.copy(),
            False,
            "ill-conditioned-final-posterior",
            diagnostics,
        )
    try:
        reduced_covariance = _covariance_from_precision(
            final_normal,
            name="posterior precision",
        )
    except ValueError:
        return (
            prior_mean.copy(),
            prior_covariance.copy(),
            False,
            "singular-final-posterior",
            diagnostics,
        )
    posterior_mean = expanded_state(current)
    posterior_covariance = _symmetric(
        prior_root @ reduced_covariance @ prior_root.T
    )
    rank = graph_basis.shape[1]
    state = _state_array(posterior_mean, rank)
    position_field = graph_basis @ state[0]
    velocity_field = graph_basis @ state[1]
    maximum_position = float(
        np.max(np.linalg.norm(position_field, axis=1), initial=0.0)
    )
    maximum_velocity = float(
        np.max(np.linalg.norm(velocity_field, axis=1), initial=0.0)
    )
    diagnostics.update(
        {
            "maximum_node_position_m": maximum_position,
            "maximum_node_velocity_mps": maximum_velocity,
        }
    )
    if (
        not np.all(np.isfinite(posterior_mean))
        or maximum_position > config.maximum_node_position_m
        or maximum_velocity > config.maximum_node_velocity_mps
    ):
        return (
            prior_mean.copy(),
            prior_covariance.copy(),
            False,
            "implausible-discrepancy-update",
            diagnostics,
        )
    return (
        posterior_mean,
        posterior_covariance,
        True,
        "inference-admissible",
        diagnostics,
    )


def fit_graph_dynamic_discrepancy(
    residual_m: np.ndarray,
    valid: np.ndarray,
    graph_basis: np.ndarray,
    *,
    frame_dt_s: float,
    observation_covariance_m2: np.ndarray | None = None,
    prior_reliability: np.ndarray | None = None,
    correlation_group_ids: np.ndarray | None = None,
    config: GraphDynamicDiscrepancyConfigV1 | None = None,
) -> GraphDynamicDiscrepancyBeliefV1:
    """Filter a causal residual prefix into one graph-modal endpoint belief."""

    if config is None:
        cfg = GraphDynamicDiscrepancyConfigV1()
    elif isinstance(config, GraphDynamicDiscrepancyConfigV1):
        cfg = config
    else:
        raise TypeError("config must be a GraphDynamicDiscrepancyConfigV1")
    frame_dt = _real(frame_dt_s, name="frame_dt_s")
    _require(frame_dt > 0.0, "frame_dt_s must be positive")
    raw_residual = np.asarray(residual_m)
    _require(
        raw_residual.dtype.kind in {"i", "u", "f"},
        "residual_m must be real numeric",
    )
    residual = np.asarray(raw_residual, dtype=np.float64)
    raw_availability = np.asarray(valid)
    _require(
        raw_availability.dtype.kind == "b",
        "valid must be a Boolean array",
    )
    availability = np.asarray(raw_availability, dtype=bool)
    basis = _validate_graph_basis(graph_basis)
    _require(
        residual.ndim == 3 and residual.shape[2] == 3,
        "residual_m must have shape (frame, node, 3)",
    )
    frame_count, node_count, _ = residual.shape
    _require(frame_count >= 1, "residual_m must contain at least one frame")
    _require(
        availability.shape == (frame_count, node_count),
        "valid shape changed",
    )
    _require(
        basis.shape[0] == node_count,
        "graph_basis does not cover every residual node",
    )
    finite = np.all(np.isfinite(residual), axis=2)
    usable = availability & finite
    if prior_reliability is None:
        reliability = np.ones((frame_count, node_count), dtype=np.float64)
    else:
        raw_reliability = np.asarray(prior_reliability)
        _require(
            raw_reliability.dtype.kind in {"i", "u", "f"},
            "prior_reliability must be real numeric",
        )
        reliability = np.asarray(raw_reliability, dtype=np.float64).copy()
        _require(
            reliability.shape == (frame_count, node_count),
            "prior_reliability shape changed",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior_reliability must lie in [0, 1]",
        )
    reliability[~usable] = 0.0
    covariance = _covariance_array(
        observation_covariance_m2,
        frame_count=frame_count,
        node_count=node_count,
        default_std_m=cfg.observation_std_m,
    )
    group_labels = _frame_group_labels(
        correlation_group_ids,
        frame_count=frame_count,
        node_count=node_count,
    )
    rank = basis.shape[1]
    position_dimension = 3 * rank
    state_dimension = 6 * rank
    state_mean = np.zeros(state_dimension, dtype=np.float64)
    state_covariance = np.zeros(
        (state_dimension, state_dimension),
        dtype=np.float64,
    )
    state_covariance[:position_dimension, :position_dimension] = (
        np.eye(position_dimension) * cfg.initial_position_std_m**2
    )
    state_covariance[position_dimension:, position_dimension:] = (
        np.eye(position_dimension) * cfg.initial_velocity_std_mps**2
    )
    transition, process_noise, _ = _transition_and_noise(
        rank,
        frame_dt_s=frame_dt,
        velocity_retention=cfg.velocity_retention,
        process_position_std_m=cfg.process_position_std_m,
        process_acceleration_std_mps2=cfg.process_acceleration_std_mps2,
    )
    accepted = np.zeros(frame_count, dtype=bool)
    reasons: list[str] = []
    frame_diagnostics: list[dict[str, Any]] = []
    for frame in range(frame_count):
        if frame:
            state_mean = transition @ state_mean
            state_covariance = _symmetric(
                transition @ state_covariance @ transition.T + process_noise
            )
        (
            state_mean,
            state_covariance,
            accepted[frame],
            reason,
            diagnostics,
        ) = _frame_update(
            state_mean,
            state_covariance,
            residual[frame],
            usable[frame],
            basis,
            covariance[frame],
            reliability[frame],
            group_labels[frame],
            cfg,
        )
        reasons.append(reason)
        frame_diagnostics.append(
            {"frame_index": frame, "reason": reason, **diagnostics}
        )
    state = _state_array(state_mean, rank)
    summary = {
        "schema": GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
        "schema_version": GRAPH_DYNAMIC_DISCREPANCY_VERSION,
        "scientific_boundary": GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
        "frame_count": frame_count,
        "node_count": node_count,
        "graph_rank": rank,
        "accepted_update_count": int(np.sum(accepted)),
        "rejected_update_count": int(np.sum(~accepted & np.any(usable, axis=1))),
        "unsupported_frame_count": int(np.sum(~np.any(usable, axis=1))),
        "observation_likelihood": "grouped Student-t IRLS",
        "prior_reliability_uses_innovation": False,
        "frame_diagnostics": frame_diagnostics,
    }
    return GraphDynamicDiscrepancyBeliefV1(
        graph_basis=basis,
        state_mean=state,
        state_covariance=state_covariance,
        frame_dt_s=frame_dt,
        velocity_retention=cfg.velocity_retention,
        process_position_std_m=cfg.process_position_std_m,
        process_acceleration_std_mps2=cfg.process_acceleration_std_mps2,
        last_frame_index=frame_count - 1,
        update_accepted=accepted,
        update_reasons=tuple(reasons),
        diagnostics=summary,
    )


__all__ = ["fit_graph_dynamic_discrepancy"]
