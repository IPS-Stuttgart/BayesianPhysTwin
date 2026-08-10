"""Observation update for graph-modal discrepancy dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._graph_dynamic_discrepancy_common import (
    GraphDynamicDiscrepancyConfigV1,
    _covariance_from_precision,
    _positive_definite_precision,
    _positive_semidefinite_root,
    _require,
    _state_array,
    _symmetric,
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
    result: np.ndarray
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
        label = value[index]
        _require(
            type(label) is str and bool(label),
            "correlation group labels must be literal nonempty strings",
        )
        result[index] = label
    return result


def _student_t_group_weight(
    squared_mahalanobis: float,
    dimension: int,
    config: GraphDynamicDiscrepancyConfigV1,
) -> float:
    _require(
        np.isfinite(squared_mahalanobis) and squared_mahalanobis >= 0.0,
        "squared Mahalanobis distance must be finite and nonnegative",
    )
    _require(dimension >= 1, "Student-t group dimension must be positive")
    covariance_degrees = config.degrees_of_freedom - 2.0
    weight = (config.degrees_of_freedom + dimension) / (
        covariance_degrees + squared_mahalanobis
    )
    return float(np.clip(weight, config.minimum_robust_weight, 1.0))


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
        design: np.ndarray = np.zeros((3, dimension), dtype=np.float64)
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
    reduced_designs = {node: design @ prior_root for node, design in designs.items()}
    centered_targets = {
        node: residual_m[node] - designs[node] @ prior_mean for node in designs
    }
    ordered_groups = tuple(dict.fromkeys(group_labels[node] for node in selected))
    group_nodes = tuple(
        np.asarray(
            [node for node in selected if group_labels[node] == label],
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
    base_information = np.zeros(
        (prior_root.shape[1], prior_root.shape[1]),
        dtype=np.float64,
    )
    for position, nodes in enumerate(group_nodes):
        for raw_node in nodes:
            node = int(raw_node)
            design = reduced_designs[node]
            base_information += (
                float(reliability[node])
                * group_power[position]
                * design.T
                @ precisions[node]
                @ design
            )
    information_eigenvalues = np.linalg.eigvalsh(_symmetric(base_information))
    information_scale = float(np.max(information_eigenvalues, initial=0.0))
    diagnostics["base_information_maximum_eigenvalue"] = information_scale
    if information_scale <= np.finfo(np.float64).eps:
        return (
            prior_mean.copy(),
            prior_covariance.copy(),
            False,
            "no-identifiable-graph-support",
            diagnostics,
        )

    def expanded_state(reduced: np.ndarray) -> np.ndarray:
        return prior_mean + prior_root @ reduced

    def robust_weights(reduced: np.ndarray) -> np.ndarray:
        current = expanded_state(reduced)
        weights: np.ndarray = np.empty(len(group_nodes), dtype=np.float64)
        for position, nodes in enumerate(group_nodes):
            squared_mahalanobis = 0.0
            for raw_node in nodes:
                node = int(raw_node)
                innovation = residual_m[node] - designs[node] @ current
                row_reliability = float(reliability[node])
                squared_mahalanobis += row_reliability * float(
                    innovation @ precisions[node] @ innovation
                )
            effective_dimension = 3 * len(nodes)
            weights[position] = _student_t_group_weight(
                squared_mahalanobis,
                effective_dimension,
                config,
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
                    float(reliability[node]) * group_power[position] * weights[position]
                )
                design = reduced_designs[node]
                precision = precisions[node]
                normal += row_weight * design.T @ precision @ design
                right += row_weight * design.T @ precision @ centered_targets[node]
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
        weight_delta = float(np.max(np.abs(refreshed - weights), initial=0.0))
        stationarity_norm = float(
            np.linalg.norm(refreshed_normal @ candidate - refreshed_right)
        )
        stationarity_scale = 1.0 + float(np.linalg.norm(refreshed_right))
        current = candidate
        weights = refreshed
        final_normal = refreshed_normal
        if (
            weight_delta <= config.convergence_tolerance
            and stationarity_norm <= config.convergence_tolerance * stationarity_scale
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
    posterior_covariance = _symmetric(prior_root @ reduced_covariance @ prior_root.T)
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
