"""Partial-observation state completion for one reusable Deform360 graph."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.spatial.distance import cdist

from causal4d_public.deform360_reusable_graph import (
    CanonicalDeform360Graph,
    deterministic_farthest_point_indices,
)


@dataclass(frozen=True)
class PartialGraphStateConfig:
    """Frozen controls for robust graph-constrained state completion."""

    start_count: int = 12
    anchor_count: int = 24
    iterations: int = 1200
    learning_rate: float = 0.002
    observation_scale_m: float = 0.01
    hidden_node_distance_cap_m: float = 0.025
    hidden_node_fit_weight: float = 0.35
    edge_strain_weight: float = 10.0
    bridge_strain_weight: float = 3.0
    contact_anchor_weight: float = 10.0
    controller_group_size: int = 768
    contact_clearance_m: float = 0.002
    readout_neighbour_count: int = 4
    readout_geometry_scale_m: float = 0.01
    readout_color_scale: float = 1.0
    readout_color_weight: float = 0.10
    measurement_variance_m2: float = 4e-6
    maximum_supported_distance_m: float = 0.02
    minimum_observed_target_fraction: float = 0.95
    minimum_effective_target_reliability: float = 0.70
    maximum_p99_relative_edge_strain: float = 0.50
    maximum_bridge_relative_edge_strain: float = 0.50
    maximum_contact_anchor_error_m: float = 0.015

    def validate(self) -> None:
        if self.start_count < 1 or self.anchor_count < 1:
            raise ValueError("state completion needs positive start and anchor counts")
        if self.iterations < 1 or self.learning_rate <= 0.0:
            raise ValueError("state completion optimizer settings are invalid")
        for name, value in (
            ("observation_scale_m", self.observation_scale_m),
            ("hidden_node_distance_cap_m", self.hidden_node_distance_cap_m),
            ("readout_geometry_scale_m", self.readout_geometry_scale_m),
            ("readout_color_scale", self.readout_color_scale),
            ("maximum_supported_distance_m", self.maximum_supported_distance_m),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            self.hidden_node_fit_weight < 0.0
            or self.edge_strain_weight <= 0.0
            or self.bridge_strain_weight <= 0.0
            or self.contact_anchor_weight <= 0.0
        ):
            raise ValueError("state completion loss weights are invalid")
        if self.readout_color_weight < 0.0:
            raise ValueError("readout_color_weight must be non-negative")
        if self.measurement_variance_m2 < 0.0:
            raise ValueError("measurement_variance_m2 must be non-negative")
        if self.readout_neighbour_count < 1:
            raise ValueError("readout_neighbour_count must be positive")
        if self.controller_group_size < 1:
            raise ValueError("controller_group_size must be positive")
        if self.contact_clearance_m <= 1e-4:
            raise ValueError("contact_clearance_m must exceed 1e-4")
        if self.maximum_contact_anchor_error_m <= 0.0:
            raise ValueError("maximum_contact_anchor_error_m must be positive")
        for name, value in (
            ("minimum_observed_target_fraction", self.minimum_observed_target_fraction),
            (
                "minimum_effective_target_reliability",
                self.minimum_effective_target_reliability,
            ),
            (
                "maximum_p99_relative_edge_strain",
                self.maximum_p99_relative_edge_strain,
            ),
            (
                "maximum_bridge_relative_edge_strain",
                self.maximum_bridge_relative_edge_strain,
            ),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")


@dataclass(frozen=True)
class PartialGraphStateResult:
    """Completed state, uncertain target readout, and physical diagnostics."""

    vertices: np.ndarray
    readout_weights: np.ndarray
    readout_covariance_m2: np.ndarray
    target_prior_reliability: np.ndarray
    state_covariance_m2: np.ndarray
    source_to_target_distance_m: np.ndarray
    target_to_source_distance_m: np.ndarray
    relative_edge_strain: np.ndarray
    metrics: dict[str, float | bool]


def _points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(points) < 1 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be non-empty and finite")
    return points


def _colors(value: np.ndarray, *, count: int, name: str) -> np.ndarray:
    colors = np.asarray(value, dtype=np.float64)
    if colors.shape != (count, 3) or not np.all(np.isfinite(colors)):
        raise ValueError(f"{name} must have finite shape ({count}, 3)")
    return colors


def _principal_axes(points: np.ndarray) -> np.ndarray:
    _, _, right = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
    axes = right.T
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def _proper_pca_rotations(source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    source_axes = _principal_axes(source)
    target_axes = _principal_axes(target)
    rotations = [np.eye(3)]
    for order in permutations(range(3)):
        permutation = np.eye(3)[:, order]
        for signs in product((-1.0, 1.0), repeat=3):
            signed = permutation @ np.diag(signs)
            rotation = target_axes @ signed @ source_axes.T
            if np.linalg.det(rotation) > 0.0:
                rotations.append(rotation)
    unique: list[np.ndarray] = []
    for rotation in rotations:
        if not any(np.allclose(rotation, seen, atol=1e-10) for seen in unique):
            unique.append(rotation)
    return unique


def _initial_states(
    source: np.ndarray,
    target: np.ndarray,
    *,
    config: PartialGraphStateConfig,
) -> np.ndarray:
    anchor_count = min(config.anchor_count, len(source))
    anchors = deterministic_farthest_point_indices(source, anchor_count)
    target_center = np.mean(target, axis=0)
    candidates: list[tuple[float, np.ndarray]] = []
    cap_sq = config.hidden_node_distance_cap_m**2
    for rotation in _proper_pca_rotations(source, target):
        rotated = source @ rotation.T
        for anchor in anchors:
            state = rotated + (target_center - rotated[anchor])
            distance = cdist(state, target)
            source_distance = np.min(distance, axis=1)
            target_distance = np.min(distance, axis=0)
            objective = float(
                np.mean(target_distance**2)
                + config.hidden_node_fit_weight
                * np.mean(np.minimum(source_distance**2, cap_sq))
            )
            candidates.append((objective, state))
    candidates.sort(key=lambda item: item[0])
    return np.stack([state for _, state in candidates[: config.start_count]])


def _robust_color_descriptor(colors: np.ndarray) -> np.ndarray:
    center = np.median(colors, axis=0)
    scale = 1.4826 * np.median(np.abs(colors - center), axis=0)
    scale = np.maximum(scale, 0.02)
    return np.clip((colors - center) / scale, -4.0, 4.0)


def _target_readout(
    vertices: np.ndarray,
    canonical_colors: np.ndarray,
    target: np.ndarray,
    target_colors: np.ndarray,
    target_reliability: np.ndarray,
    *,
    config: PartialGraphStateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry_sq = cdist(target, vertices, metric="sqeuclidean")
    canonical_descriptor = _robust_color_descriptor(canonical_colors)
    target_descriptor = _robust_color_descriptor(target_colors)
    color_sq = cdist(target_descriptor, canonical_descriptor, metric="sqeuclidean")
    cost = geometry_sq / (config.readout_geometry_scale_m**2)
    cost += config.readout_color_weight * color_sq / (config.readout_color_scale**2)
    neighbour_count = min(config.readout_neighbour_count, len(vertices))
    selected = np.argpartition(cost, neighbour_count - 1, axis=1)[:, :neighbour_count]
    selected_cost = np.take_along_axis(cost, selected, axis=1)
    logits = -0.5 * (selected_cost - np.min(selected_cost, axis=1, keepdims=True))
    probability = np.exp(np.clip(logits, -700.0, 0.0))
    probability /= np.sum(probability, axis=1, keepdims=True)
    weights = np.zeros((len(target), len(vertices)), dtype=np.float64)
    np.put_along_axis(weights, selected, probability, axis=1)
    readout = weights @ vertices
    covariance = np.empty((len(target), 3, 3), dtype=np.float64)
    for point in range(len(target)):
        residual = vertices - readout[point]
        covariance[point] = (
            residual.T * weights[point]
        ) @ residual + config.measurement_variance_m2 * np.eye(3)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-300)), axis=1)
    entropy /= np.log(max(2, neighbour_count))
    reconstruction_error = np.linalg.norm(readout - target, axis=1)
    geometry_reliability = np.exp(
        -0.5 * (reconstruction_error / config.maximum_supported_distance_m) ** 2
    )
    ambiguity_reliability = np.clip(1.0 - 0.25 * entropy, 0.0, 1.0)
    reliability = target_reliability * geometry_reliability * ambiguity_reliability
    return weights, covariance, reliability


def _contact_targets(
    canonical: CanonicalDeform360Graph,
    controller_points: np.ndarray | None,
    episode_points: np.ndarray,
    *,
    controller_group_size: int,
    contact_clearance_m: float,
) -> np.ndarray:
    anchors = np.asarray(canonical.contact_anchor_indices, dtype=np.int64)
    if len(anchors) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if controller_points is None:
        raise ValueError("canonical contact anchors require controller points")
    controls = _points(controller_points, name="controller_points")
    if len(controls) % controller_group_size:
        raise ValueError("controller points do not form locked groups")
    group_count = len(controls) // controller_group_size
    if group_count != len(anchors):
        raise ValueError("controller group count differs from canonical contacts")
    targets = []
    for start in range(0, len(controls), controller_group_size):
        group = controls[start : start + controller_group_size]
        distance = cdist(group, episode_points)
        controller_index, object_index = np.unravel_index(
            np.argmin(distance),
            distance.shape,
        )
        controller_position = group[int(controller_index)]
        controller_to_object = episode_points[int(object_index)] - controller_position
        norm = float(np.linalg.norm(controller_to_object))
        if norm <= contact_clearance_m:
            targets.append(episode_points[int(object_index)])
        else:
            targets.append(
                controller_position + contact_clearance_m * controller_to_object / norm
            )
    return np.asarray(targets, dtype=np.float64)


def evaluate_partial_graph_state(
    canonical: CanonicalDeform360Graph,
    vertices: np.ndarray,
    episode_points: np.ndarray,
    episode_colors: np.ndarray,
    *,
    config: PartialGraphStateConfig,
    candidate_reliability: np.ndarray | None = None,
    controller_points: np.ndarray | None = None,
) -> PartialGraphStateResult:
    """Evaluate a completed state with the same uncertainty and physical gates."""

    config.validate()
    state = _points(vertices, name="vertices")
    if state.shape != canonical.vertices.shape:
        raise ValueError("completed state must match the canonical graph")
    target = _points(episode_points, name="episode_points")
    target_colors = _colors(
        episode_colors,
        count=len(target),
        name="episode_colors",
    )
    canonical_colors = _colors(
        canonical.colors,
        count=len(state),
        name="canonical.colors",
    )
    if candidate_reliability is None:
        target_reliability = np.ones(len(target), dtype=np.float64)
    else:
        target_reliability = np.asarray(candidate_reliability, dtype=np.float64)
        if target_reliability.shape != (len(target),) or not np.all(
            np.isfinite(target_reliability)
        ):
            raise ValueError("candidate_reliability must be finite per target point")
        if np.any(target_reliability < 0.0) or np.any(target_reliability > 1.0):
            raise ValueError("candidate_reliability must lie in [0, 1]")
    distance = cdist(state, target)
    source_distance = np.min(distance, axis=1)
    target_distance = np.min(distance, axis=0)
    edge_length = np.linalg.norm(
        state[canonical.springs[:, 0]] - state[canonical.springs[:, 1]],
        axis=1,
    )
    relative_strain = np.abs(edge_length / canonical.rest_lengths - 1.0)
    contact_targets = _contact_targets(
        canonical,
        controller_points,
        target,
        controller_group_size=config.controller_group_size,
        contact_clearance_m=config.contact_clearance_m,
    )
    contact_error = (
        np.linalg.norm(
            state[canonical.contact_anchor_indices] - contact_targets,
            axis=1,
        )
        if len(contact_targets)
        else np.empty(0, dtype=np.float64)
    )
    weights, readout_covariance, prior_reliability = _target_readout(
        state,
        canonical_colors,
        target,
        target_colors,
        target_reliability,
        config=config,
    )
    readout = weights @ state
    state_variance = config.measurement_variance_m2 + source_distance**2
    state_covariance = state_variance[:, None, None] * np.eye(3)[None]
    observed_fraction = float(
        np.mean(target_distance <= config.maximum_supported_distance_m)
    )
    canonical_supported_fraction = float(
        np.mean(source_distance <= config.maximum_supported_distance_m)
    )
    p99_strain = float(np.quantile(relative_strain, 0.99))
    bridge_strain = (
        relative_strain[-canonical.bridge_spring_count :]
        if canonical.bridge_spring_count
        else np.empty(0, dtype=np.float64)
    )
    maximum_bridge_strain = float(np.max(bridge_strain, initial=0.0))
    maximum_contact_error = float(np.max(contact_error, initial=0.0))
    effective_reliability = float(np.mean(prior_reliability))
    finite = bool(
        np.all(np.isfinite(state))
        and np.all(np.isfinite(readout_covariance))
        and np.all(np.isfinite(state_covariance))
    )
    passed = bool(
        finite
        and observed_fraction >= config.minimum_observed_target_fraction
        and effective_reliability >= config.minimum_effective_target_reliability
        and p99_strain <= config.maximum_p99_relative_edge_strain
        and maximum_bridge_strain <= config.maximum_bridge_relative_edge_strain
        and maximum_contact_error <= config.maximum_contact_anchor_error_m
    )
    metrics: dict[str, float | bool] = {
        "passed": passed,
        "finite": finite,
        "symmetric_chamfer_m": float(
            0.5 * (np.mean(source_distance) + np.mean(target_distance))
        ),
        "source_to_target_p95_m": float(np.quantile(source_distance, 0.95)),
        "target_to_source_p95_m": float(np.quantile(target_distance, 0.95)),
        "observed_target_fraction": observed_fraction,
        "canonical_supported_fraction": canonical_supported_fraction,
        "effective_target_reliability": effective_reliability,
        "initial_readout_rmse_m": float(np.sqrt(np.mean((readout - target) ** 2))),
        "p99_absolute_relative_edge_strain": p99_strain,
        "maximum_absolute_relative_edge_strain": float(np.max(relative_strain)),
        "maximum_bridge_absolute_relative_edge_strain": maximum_bridge_strain,
        "maximum_contact_anchor_error_m": maximum_contact_error,
    }
    return PartialGraphStateResult(
        vertices=state,
        readout_weights=weights,
        readout_covariance_m2=readout_covariance,
        target_prior_reliability=prior_reliability,
        state_covariance_m2=state_covariance,
        source_to_target_distance_m=source_distance,
        target_to_source_distance_m=target_distance,
        relative_edge_strain=relative_strain,
        metrics=metrics,
    )


def fit_partial_graph_state(
    canonical: CanonicalDeform360Graph,
    episode_points: np.ndarray,
    episode_colors: np.ndarray,
    *,
    config: PartialGraphStateConfig,
    candidate_reliability: np.ndarray | None = None,
    controller_points: np.ndarray | None = None,
    device: str = "cpu",
) -> PartialGraphStateResult:
    """Fit current graph state to partial observations while preserving material lengths."""

    import torch

    config.validate()
    source = _points(canonical.vertices, name="canonical.vertices")
    target = _points(episode_points, name="episode_points")
    target_colors = _colors(
        episode_colors,
        count=len(target),
        name="episode_colors",
    )
    contact_targets = _contact_targets(
        canonical,
        controller_points,
        target,
        controller_group_size=config.controller_group_size,
        contact_clearance_m=config.contact_clearance_m,
    )
    if candidate_reliability is None:
        target_reliability = np.ones(len(target), dtype=np.float64)
    else:
        target_reliability = np.asarray(candidate_reliability, dtype=np.float64)
        if target_reliability.shape != (len(target),) or not np.all(
            np.isfinite(target_reliability)
        ):
            raise ValueError("candidate_reliability must be finite per target point")
        if np.any(target_reliability < 0.0) or np.any(target_reliability > 1.0):
            raise ValueError("candidate_reliability must lie in [0, 1]")

    starts = _initial_states(source, target, config=config).astype(np.float32)
    state = torch.tensor(starts, dtype=torch.float32, device=device, requires_grad=True)
    target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    edges = torch.tensor(canonical.springs, dtype=torch.long, device=device)
    rest_lengths = torch.tensor(
        canonical.rest_lengths,
        dtype=torch.float32,
        device=device,
    )
    contact_anchor_indices = torch.tensor(
        canonical.contact_anchor_indices,
        dtype=torch.long,
        device=device,
    )
    contact_target_tensor = torch.tensor(
        contact_targets,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam([state], lr=config.learning_rate)
    cap = torch.tensor(config.hidden_node_distance_cap_m, device=device)
    observation_variance = config.observation_scale_m**2
    for _ in range(config.iterations):
        optimizer.zero_grad()
        distance = torch.cdist(state, target_tensor)
        source_distance = distance.min(dim=2).values
        target_distance = distance.min(dim=1).values
        data_loss = target_distance.square().mean(dim=1) / observation_variance
        data_loss += config.hidden_node_fit_weight * (
            torch.minimum(source_distance, cap).square().mean(dim=1)
            / observation_variance
        )
        edge_length = torch.linalg.vector_norm(
            state[:, edges[:, 0]] - state[:, edges[:, 1]],
            dim=-1,
        )
        strain = edge_length / rest_lengths - 1.0
        loss = data_loss + config.edge_strain_weight * strain.square().mean(dim=1)
        if canonical.bridge_spring_count:
            loss += config.bridge_strain_weight * strain[
                :, -canonical.bridge_spring_count :
            ].square().mean(dim=1)
        if len(contact_targets):
            contact_error = state[:, contact_anchor_indices] - contact_target_tensor
            loss += config.contact_anchor_weight * (
                contact_error.square().sum(dim=2).mean(dim=1) / observation_variance
            )
        loss.sum().backward()
        optimizer.step()

    with torch.no_grad():
        distance = torch.cdist(state, target_tensor)
        source_distance_all = distance.min(dim=2).values
        target_distance_all = distance.min(dim=1).values
        edge_length = torch.linalg.vector_norm(
            state[:, edges[:, 0]] - state[:, edges[:, 1]],
            dim=-1,
        )
        strain_all = torch.abs(edge_length / rest_lengths - 1.0)
        objective = target_distance_all.square().mean(dim=1) / observation_variance
        objective += config.hidden_node_fit_weight * (
            torch.minimum(source_distance_all, cap).square().mean(dim=1)
            / observation_variance
        )
        objective += config.edge_strain_weight * strain_all.square().mean(dim=1)
        if canonical.bridge_spring_count:
            objective += config.bridge_strain_weight * strain_all[
                :, -canonical.bridge_spring_count :
            ].square().mean(dim=1)
        if len(contact_targets):
            contact_error = state[:, contact_anchor_indices] - contact_target_tensor
            objective += config.contact_anchor_weight * (
                contact_error.square().sum(dim=2).mean(dim=1) / observation_variance
            )
        selected = int(torch.argmin(objective).item())
        vertices = state[selected].detach().cpu().numpy().astype(np.float64)
    return evaluate_partial_graph_state(
        canonical,
        vertices,
        target,
        target_colors,
        config=config,
        candidate_reliability=target_reliability,
        controller_points=controller_points,
    )


__all__ = [
    "PartialGraphStateConfig",
    "PartialGraphStateResult",
    "evaluate_partial_graph_state",
    "fit_partial_graph_state",
]
