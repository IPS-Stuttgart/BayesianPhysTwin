"""Physics-guided assimilation of anonymous MotionCrafter scene flow."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_graph import PhysTwinSpringGraphConfig, build_phystwin_spring_graph
from .phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)
from .phystwin_motioncrafter_association import (
    MOTIONCRAFTER_REPOSITORY,
    MOTIONCRAFTER_REVISION,
    MotionCrafterPrediction,
    _distribution,
    _load_pickle,
    _mean_on_frames,
    _nearest_neighbors,
    _sha256,
    align_motioncrafter_prediction,
    dense_graph_error_by_frame,
    load_motioncrafter_prediction,
    manual_track_association_audit,
    resample_cover_grid,
    robust_similarity_transform,
)


@dataclass(frozen=True)
class AnonymousSceneFlowConfig:
    """Frozen settings for physics-guided anonymous scene-flow assimilation."""

    camera_index: int = 0
    process_stride: int = 1
    measurement_stride_pixels: int = 4
    alignment_stride_pixels: int = 4
    alignment_trim_fraction: float = 0.8
    alignment_iterations: int = 5
    candidate_count: int = 4
    position_scale_m: float = 0.01
    flow_scale_m: float = 0.02
    flow_strength: float = 1.0
    maximum_position_error_m: float = 0.04
    maximum_flow_endpoint_error_m: float = 0.06
    entropy_strength: float = 0.5
    minimum_observation_mass: float = 0.2
    multiview_consistency_scale_m: float = 0.015
    minimum_multiview_reliability: float = 0.05
    graph_prior_strength: float = 0.3
    graph_zero_prior_strength: float = 0.0
    graph_ridge: float = 1e-8
    graph_solver_relative_tolerance: float = 1e-5
    graph_solver_maximum_iterations: int = 5000
    maximum_graph_correction_m: float = 0.01


@dataclass(frozen=True)
class FramewiseGraphObservations:
    """Anonymous MotionCrafter measurements assigned to persistent graph IDs."""

    positions: np.ndarray
    flow_endpoints: np.ndarray
    valid: np.ndarray
    flow_valid: np.ndarray
    reliability: np.ndarray
    flow_reliability: np.ndarray
    measurement_mass: np.ndarray
    flow_measurement_mass: np.ndarray
    normalized_entropy: np.ndarray
    position_error_m: np.ndarray
    flow_endpoint_error_m: np.ndarray
    sampled_measurement_count: np.ndarray
    accepted_measurement_count: np.ndarray


@dataclass(frozen=True)
class GraphRegularizedObservations:
    """Graph-coherent state observations and their direct-evidence boundary."""

    positions: np.ndarray
    correction: np.ndarray
    valid: np.ndarray
    direct_valid: np.ndarray
    direct_reliability: np.ndarray
    solve_iterations: np.ndarray
    solve_relative_residual: np.ndarray


def _validate_config(config: AnonymousSceneFlowConfig) -> None:
    if config.camera_index < 0 or config.process_stride < 1:
        raise ValueError("camera_index must be nonnegative and stride positive")
    if (
        min(
            config.measurement_stride_pixels,
            config.alignment_stride_pixels,
            config.alignment_iterations,
            config.candidate_count,
        )
        < 1
    ):
        raise ValueError("stride, iteration, and candidate settings must be positive")
    if not 0.5 <= config.alignment_trim_fraction <= 1.0:
        raise ValueError("alignment_trim_fraction must lie in [0.5, 1]")
    if (
        min(
            config.position_scale_m,
            config.flow_scale_m,
            config.maximum_position_error_m,
            config.maximum_flow_endpoint_error_m,
            config.minimum_observation_mass,
            config.multiview_consistency_scale_m,
            config.graph_prior_strength,
            config.graph_ridge,
            config.graph_solver_relative_tolerance,
            config.maximum_graph_correction_m,
        )
        <= 0.0
    ):
        raise ValueError("likelihood scales and graph settings must be positive")
    if (
        min(
            config.flow_strength,
            config.entropy_strength,
            config.graph_zero_prior_strength,
        )
        < 0.0
    ):
        raise ValueError("likelihood strengths and zero prior must be nonnegative")
    if not 0.0 < config.minimum_multiview_reliability <= 1.0:
        raise ValueError("minimum_multiview_reliability must lie in (0, 1]")
    if config.graph_solver_maximum_iterations < 1:
        raise ValueError("graph_solver_maximum_iterations must be positive")


def _masked_softmax_negative_cost(
    cost: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_cost = np.asarray(cost, dtype=float)
    candidate_valid = np.asarray(valid, dtype=bool)
    if candidate_cost.shape != candidate_valid.shape or candidate_cost.ndim != 2:
        raise ValueError("cost and valid must have matching shape (M, K)")
    row_valid = np.any(candidate_valid, axis=1)
    weights = np.zeros_like(candidate_cost)
    if not np.any(row_valid):
        return weights, row_valid
    selected_cost = candidate_cost[row_valid]
    selected_valid = candidate_valid[row_valid]
    minimum = np.min(np.where(selected_valid, selected_cost, np.inf), axis=1)
    score = np.where(
        selected_valid,
        np.exp(np.clip(-(selected_cost - minimum[:, None]), -700.0, 0.0)),
        0.0,
    )
    weights[row_valid] = score / np.sum(score, axis=1, keepdims=True)
    return weights, row_valid


def _accumulate_vertex_measurements(
    vertex_count: int,
    candidate_indices: np.ndarray,
    candidate_weights: np.ndarray,
    measurement_confidence: np.ndarray,
    measurement_values: np.ndarray,
    measurement_entropy: np.ndarray,
    position_error: np.ndarray,
    flow_error: np.ndarray,
) -> tuple[np.ndarray, ...]:
    indices = np.asarray(candidate_indices, dtype=np.int64).ravel()
    posterior = np.asarray(candidate_weights, dtype=float).ravel()
    confidence = np.repeat(
        np.asarray(measurement_confidence, dtype=float),
        candidate_indices.shape[1],
    )
    evidence = posterior * confidence
    values = np.repeat(
        np.asarray(measurement_values, dtype=float),
        candidate_indices.shape[1],
        axis=0,
    )
    entropy = np.repeat(
        np.asarray(measurement_entropy, dtype=float),
        candidate_indices.shape[1],
    )
    position = np.repeat(
        np.asarray(position_error, dtype=float),
        candidate_indices.shape[1],
    )
    flow = np.repeat(
        np.asarray(flow_error, dtype=float),
        candidate_indices.shape[1],
    )

    mass = np.zeros(vertex_count, dtype=float)
    evidence_mass = np.zeros(vertex_count, dtype=float)
    numerator = np.zeros((vertex_count, 3), dtype=float)
    entropy_numerator = np.zeros(vertex_count, dtype=float)
    position_numerator = np.zeros(vertex_count, dtype=float)
    flow_numerator = np.zeros(vertex_count, dtype=float)
    np.add.at(mass, indices, posterior)
    np.add.at(evidence_mass, indices, evidence)
    np.add.at(numerator, indices, evidence[:, None] * values)
    np.add.at(entropy_numerator, indices, evidence * entropy)
    np.add.at(position_numerator, indices, evidence * position)
    finite_flow = np.isfinite(flow)
    np.add.at(
        flow_numerator, indices[finite_flow], evidence[finite_flow] * flow[finite_flow]
    )
    return (
        mass,
        evidence_mass,
        numerator,
        entropy_numerator,
        position_numerator,
        flow_numerator,
    )


def associate_anonymous_scene_flow(
    prediction: MotionCrafterPrediction,
    object_masks: np.ndarray,
    graph_trajectory: np.ndarray,
    *,
    config: AnonymousSceneFlowConfig,
) -> FramewiseGraphObservations:
    """Re-associate anonymous points/flow to PhysTwin graph IDs every frame.

    Candidate identities come from one fixed PhysTwin trajectory. MotionCrafter
    identities are never propagated across time; its flow only scores whether a
    current measurement is compatible with a graph vertex's next state.
    """

    _validate_config(config)
    masks = np.asarray(object_masks, dtype=bool)
    graph = np.asarray(graph_trajectory, dtype=float)
    if masks.shape != prediction.valid_mask.shape:
        raise ValueError("object_masks must match MotionCrafter frame/image shape")
    if graph.ndim != 3 or graph.shape[2] != 3:
        raise ValueError("graph_trajectory must have shape (T, N, 3)")
    if len(graph) != len(prediction.point_map):
        raise ValueError("graph and MotionCrafter trajectories must share frames")
    if not np.all(np.isfinite(graph)):
        raise ValueError("graph_trajectory must be finite")
    frame_count, height, width = masks.shape
    vertex_count = graph.shape[1]
    if vertex_count < 1:
        raise ValueError("graph_trajectory must contain vertices")
    candidate_count = min(config.candidate_count, vertex_count)

    positions = np.full((frame_count, vertex_count, 3), np.nan, dtype=np.float32)
    endpoints = np.full_like(positions, np.nan)
    valid = np.zeros((frame_count, vertex_count), dtype=bool)
    flow_valid = np.zeros_like(valid)
    reliability = np.zeros((frame_count, vertex_count), dtype=np.float32)
    flow_reliability = np.zeros_like(reliability)
    mass = np.zeros((frame_count, vertex_count), dtype=np.float32)
    flow_mass = np.zeros_like(mass)
    entropy = np.full((frame_count, vertex_count), np.nan, dtype=np.float32)
    position_error = np.full_like(entropy, np.nan)
    flow_error = np.full_like(entropy, np.nan)
    sampled_count = np.zeros(frame_count, dtype=np.int32)
    accepted_count = np.zeros(frame_count, dtype=np.int32)
    grid_y, grid_x = np.indices((height, width))
    stride_mask = (grid_y % config.measurement_stride_pixels == 0) & (
        grid_x % config.measurement_stride_pixels == 0
    )

    for frame in range(frame_count):
        sample_mask = (
            masks[frame]
            & prediction.valid_mask[frame]
            & stride_mask
            & np.all(np.isfinite(prediction.point_map[frame]), axis=2)
        )
        pixels = np.column_stack(np.nonzero(sample_mask))
        sampled_count[frame] = len(pixels)
        if len(pixels) == 0:
            continue
        points = prediction.point_map[frame, pixels[:, 0], pixels[:, 1]].astype(float)
        candidate_distance, candidate_indices = _nearest_neighbors(
            graph[frame], points, k=candidate_count
        )
        candidate_valid = candidate_distance <= config.maximum_position_error_m
        cost = 0.5 * np.square(candidate_distance / config.position_scale_m)
        flow_is_usable = np.zeros(len(points), dtype=bool)
        measurement_endpoints = np.full_like(points, np.nan)
        candidate_flow_distance = np.full_like(candidate_distance, np.nan)
        if frame + 1 < frame_count:
            selected_flow = prediction.scene_flow[
                frame, pixels[:, 0], pixels[:, 1]
            ].astype(float)
            flow_is_usable = prediction.deform_mask[
                frame, pixels[:, 0], pixels[:, 1]
            ] & np.all(np.isfinite(selected_flow), axis=1)
            measurement_endpoints[flow_is_usable] = (
                points[flow_is_usable] + selected_flow[flow_is_usable]
            )
            if np.any(flow_is_usable):
                next_candidates = graph[frame + 1][candidate_indices[flow_is_usable]]
                candidate_flow_distance[flow_is_usable] = np.linalg.norm(
                    measurement_endpoints[flow_is_usable, None] - next_candidates,
                    axis=2,
                )
                if config.flow_strength > 0.0:
                    candidate_valid[flow_is_usable] &= (
                        candidate_flow_distance[flow_is_usable]
                        <= config.maximum_flow_endpoint_error_m
                    )
                    cost[flow_is_usable] += (
                        config.flow_strength
                        * 0.5
                        * np.square(
                            candidate_flow_distance[flow_is_usable]
                            / config.flow_scale_m
                        )
                    )

        candidate_weights, accepted = _masked_softmax_negative_cost(
            cost, candidate_valid
        )
        accepted_count[frame] = int(np.sum(accepted))
        if not np.any(accepted):
            continue
        candidate_indices = candidate_indices[accepted]
        candidate_weights = candidate_weights[accepted]
        points = points[accepted]
        measurement_endpoints = measurement_endpoints[accepted]
        flow_is_usable = flow_is_usable[accepted]
        candidate_distance = candidate_distance[accepted]
        candidate_flow_distance = candidate_flow_distance[accepted]
        valid_candidate_count = np.sum(candidate_weights > 0.0, axis=1)
        measurement_entropy = -np.sum(
            np.where(
                candidate_weights > 0.0,
                candidate_weights * np.log(np.maximum(candidate_weights, 1e-300)),
                0.0,
            ),
            axis=1,
        )
        multiple = valid_candidate_count > 1
        measurement_entropy[multiple] /= np.log(valid_candidate_count[multiple])
        expected_cost = np.sum(candidate_weights * cost[accepted], axis=1)
        measurement_confidence = np.exp(
            -expected_cost - config.entropy_strength * measurement_entropy
        )
        expected_position_error = np.sum(candidate_weights * candidate_distance, axis=1)
        expected_flow_error = np.full(len(points), np.nan, dtype=float)
        expected_flow_error[flow_is_usable] = np.sum(
            candidate_weights[flow_is_usable] * candidate_flow_distance[flow_is_usable],
            axis=1,
        )
        (
            frame_mass,
            evidence_mass,
            numerator,
            entropy_numerator,
            position_numerator,
            _,
        ) = _accumulate_vertex_measurements(
            vertex_count,
            candidate_indices,
            candidate_weights,
            measurement_confidence,
            points,
            measurement_entropy,
            expected_position_error,
            expected_flow_error,
        )
        frame_valid = evidence_mass >= config.minimum_observation_mass
        positions[frame, frame_valid] = (
            numerator[frame_valid] / evidence_mass[frame_valid, None]
        ).astype(np.float32)
        valid[frame] = frame_valid
        mass[frame] = frame_mass.astype(np.float32)
        reliability[frame] = (1.0 - np.exp(-evidence_mass)).astype(np.float32)
        reliability[frame, ~frame_valid] = 0.0
        entropy[frame, frame_valid] = (
            entropy_numerator[frame_valid] / evidence_mass[frame_valid]
        ).astype(np.float32)
        position_error[frame, frame_valid] = (
            position_numerator[frame_valid] / evidence_mass[frame_valid]
        ).astype(np.float32)

        if np.any(flow_is_usable):
            flow_candidate_indices = candidate_indices[flow_is_usable]
            flow_candidate_weights = candidate_weights[flow_is_usable]
            flow_confidence = measurement_confidence[flow_is_usable]
            (
                frame_flow_mass,
                flow_evidence_mass,
                flow_numerator,
                _,
                _,
                flow_error_numerator,
            ) = _accumulate_vertex_measurements(
                vertex_count,
                flow_candidate_indices,
                flow_candidate_weights,
                flow_confidence,
                measurement_endpoints[flow_is_usable],
                measurement_entropy[flow_is_usable],
                expected_position_error[flow_is_usable],
                expected_flow_error[flow_is_usable],
            )
            frame_flow_valid = flow_evidence_mass >= config.minimum_observation_mass
            endpoints[frame, frame_flow_valid] = (
                flow_numerator[frame_flow_valid]
                / flow_evidence_mass[frame_flow_valid, None]
            ).astype(np.float32)
            flow_valid[frame] = frame_flow_valid
            flow_mass[frame] = frame_flow_mass.astype(np.float32)
            flow_reliability[frame] = (1.0 - np.exp(-flow_evidence_mass)).astype(
                np.float32
            )
            flow_reliability[frame, ~frame_flow_valid] = 0.0
            flow_error[frame, frame_flow_valid] = (
                flow_error_numerator[frame_flow_valid]
                / flow_evidence_mass[frame_flow_valid]
            ).astype(np.float32)

    return FramewiseGraphObservations(
        positions=positions,
        flow_endpoints=endpoints,
        valid=valid,
        flow_valid=flow_valid,
        reliability=reliability,
        flow_reliability=flow_reliability,
        measurement_mass=mass,
        flow_measurement_mass=flow_mass,
        normalized_entropy=entropy,
        position_error_m=position_error,
        flow_endpoint_error_m=flow_error,
        sampled_measurement_count=sampled_count,
        accepted_measurement_count=accepted_count,
    )


def _fuse_multiview_values(
    values: np.ndarray,
    valid: np.ndarray,
    reliability: np.ndarray,
    *,
    consistency_scale_m: float,
    minimum_reliability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measurements = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    confidence = np.asarray(reliability, dtype=float)
    if measurements.ndim != 4 or measurements.shape[-1] != 3:
        raise ValueError("multiview values must have shape (V, T, N, 3)")
    if mask.shape != measurements.shape[:3] or confidence.shape != mask.shape:
        raise ValueError("multiview validity and reliability shapes disagree")
    base_weight = np.where(mask, np.clip(confidence, 0.0, 1.0), 0.0)
    weighted_values = np.where(mask[..., None], measurements, 0.0)
    denominator = np.sum(base_weight, axis=0)
    center = np.zeros(measurements.shape[1:], dtype=float)
    available = denominator > 0.0
    center[available] = (
        np.sum(base_weight[..., None] * weighted_values, axis=0)[available]
        / denominator[available, None]
    )
    effective_weight = base_weight.copy()
    for _ in range(2):
        residual = np.linalg.norm(measurements - center[None], axis=3)
        consistency = np.exp(
            np.clip(
                -0.5 * np.square(residual / consistency_scale_m),
                -50.0,
                0.0,
            )
        )
        effective_weight = np.where(mask, base_weight * consistency, 0.0)
        denominator = np.sum(effective_weight, axis=0)
        available = denominator > 0.0
        center[available] = (
            np.sum(effective_weight[..., None] * weighted_values, axis=0)[available]
            / denominator[available, None]
        )
    combined_reliability = 1.0 - np.prod(1.0 - effective_weight, axis=0)
    combined_valid = available & (combined_reliability >= minimum_reliability)
    combined = np.full(measurements.shape[1:], np.nan, dtype=np.float32)
    combined[combined_valid] = center[combined_valid].astype(np.float32)
    combined_reliability[~combined_valid] = 0.0
    return (
        combined,
        combined_valid,
        combined_reliability.astype(np.float32),
        effective_weight,
    )


def _fuse_scalar_diagnostic(
    values: np.ndarray,
    effective_weight: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    diagnostics = np.asarray(values, dtype=float)
    usable = np.isfinite(diagnostics) & (effective_weight > 0.0)
    weight = np.where(usable, effective_weight, 0.0)
    denominator = np.sum(weight, axis=0)
    output = np.full(diagnostics.shape[1:], np.nan, dtype=np.float32)
    available = np.asarray(valid, dtype=bool) & (denominator > 0.0)
    output[available] = (
        np.sum(np.where(usable, weight * diagnostics, 0.0), axis=0)[available]
        / denominator[available]
    ).astype(np.float32)
    return output


def combine_framewise_graph_observations(
    observations_by_view: dict[int, FramewiseGraphObservations],
    *,
    consistency_scale_m: float = 0.015,
    minimum_reliability: float = 0.05,
) -> FramewiseGraphObservations:
    """Reliability-fuse calibrated views and suppress cross-view outliers."""

    if not observations_by_view:
        raise ValueError("at least one view is required")
    if consistency_scale_m <= 0.0 or not 0.0 < minimum_reliability <= 1.0:
        raise ValueError("invalid multiview fusion settings")
    ordered = [observations_by_view[key] for key in sorted(observations_by_view)]
    shapes = {observation.positions.shape for observation in ordered}
    if len(shapes) != 1:
        raise ValueError("all views must share graph observation shape")
    if len(ordered) == 1:
        return ordered[0]

    positions, valid, reliability, position_weight = _fuse_multiview_values(
        np.stack([observation.positions for observation in ordered]),
        np.stack([observation.valid for observation in ordered]),
        np.stack([observation.reliability for observation in ordered]),
        consistency_scale_m=consistency_scale_m,
        minimum_reliability=minimum_reliability,
    )
    endpoints, flow_valid, flow_reliability, flow_weight = _fuse_multiview_values(
        np.stack([observation.flow_endpoints for observation in ordered]),
        np.stack([observation.flow_valid for observation in ordered]),
        np.stack([observation.flow_reliability for observation in ordered]),
        consistency_scale_m=consistency_scale_m,
        minimum_reliability=minimum_reliability,
    )
    return FramewiseGraphObservations(
        positions=positions,
        flow_endpoints=endpoints,
        valid=valid,
        flow_valid=flow_valid,
        reliability=reliability,
        flow_reliability=flow_reliability,
        measurement_mass=np.sum(
            np.stack([observation.measurement_mass for observation in ordered]),
            axis=0,
        ).astype(np.float32),
        flow_measurement_mass=np.sum(
            np.stack([observation.flow_measurement_mass for observation in ordered]),
            axis=0,
        ).astype(np.float32),
        normalized_entropy=_fuse_scalar_diagnostic(
            np.stack([observation.normalized_entropy for observation in ordered]),
            position_weight,
            valid,
        ),
        position_error_m=_fuse_scalar_diagnostic(
            np.stack([observation.position_error_m for observation in ordered]),
            position_weight,
            valid,
        ),
        flow_endpoint_error_m=_fuse_scalar_diagnostic(
            np.stack([observation.flow_endpoint_error_m for observation in ordered]),
            flow_weight,
            flow_valid,
        ),
        sampled_measurement_count=np.sum(
            np.stack(
                [observation.sampled_measurement_count for observation in ordered]
            ),
            axis=0,
        ).astype(np.int32),
        accepted_measurement_count=np.sum(
            np.stack(
                [observation.accepted_measurement_count for observation in ordered]
            ),
            axis=0,
        ).astype(np.int32),
    )


def _spring_components(node_count: int, springs: np.ndarray) -> np.ndarray:
    parent = np.arange(node_count, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in np.asarray(springs, dtype=np.int64):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            parent[second_root] = first_root
    return np.asarray([find(node) for node in range(node_count)], dtype=np.int64)


def graph_regularized_state_observations(
    graph_trajectory: np.ndarray,
    observations: FramewiseGraphObservations,
    springs: np.ndarray,
    *,
    prior_strength: float,
    zero_prior_strength: float = 0.0,
    ridge: float = 1e-8,
    relative_tolerance: float = 1e-5,
    maximum_iterations: int = 5000,
    maximum_correction_m: float = 0.03,
) -> GraphRegularizedObservations:
    """Interpolate direct innovations under the PhysTwin spring Laplacian."""

    graph = np.asarray(graph_trajectory, dtype=float)
    edges = np.asarray(springs, dtype=np.int64)
    if graph.shape != observations.positions.shape:
        raise ValueError("graph_trajectory must match observation positions")
    if (
        prior_strength <= 0.0
        or zero_prior_strength < 0.0
        or ridge <= 0.0
        or relative_tolerance <= 0.0
        or maximum_iterations < 1
        or maximum_correction_m <= 0.0
    ):
        raise ValueError("graph smoothing settings must be positive")
    node_count = graph.shape[1]
    laplacian = normalized_spring_laplacian(node_count, edges)
    components = _spring_components(node_count, edges)
    positions = np.full_like(graph, np.nan, dtype=np.float32)
    correction = np.full_like(graph, np.nan, dtype=np.float32)
    valid = np.zeros(graph.shape[:2], dtype=bool)
    iterations = np.zeros((len(graph), 3), dtype=np.int32)
    residual = np.full((len(graph), 3), np.nan, dtype=np.float64)

    for frame in range(len(graph)):
        direct = observations.valid[frame]
        if not np.any(direct):
            continue
        innovation = np.zeros((node_count, 3), dtype=float)
        innovation[direct] = (
            observations.positions[frame, direct] - graph[frame, direct]
        )
        variance = np.ones(node_count, dtype=float)
        variance[direct] = 1.0 / np.maximum(
            observations.reliability[frame, direct], 1e-6
        )
        posterior = graph_smoothed_discrepancy_posterior(
            innovation,
            variance,
            direct,
            laplacian,
            prior_strength=prior_strength,
            ridge=zero_prior_strength + ridge,
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
        )
        frame_correction = posterior.mean
        norm = np.linalg.norm(frame_correction, axis=1)
        over_limit = norm > maximum_correction_m
        frame_correction[over_limit] *= (maximum_correction_m / norm[over_limit])[
            :, None
        ]
        supported_components = np.unique(components[direct])
        frame_valid = np.isin(components, supported_components)
        positions[frame, frame_valid] = (
            graph[frame, frame_valid] + frame_correction[frame_valid]
        ).astype(np.float32)
        correction[frame, frame_valid] = frame_correction[frame_valid].astype(
            np.float32
        )
        valid[frame] = frame_valid
        iterations[frame] = np.asarray(posterior.solve_iterations, dtype=np.int32)
        residual[frame] = np.asarray(posterior.solve_relative_residuals, dtype=float)

    return GraphRegularizedObservations(
        positions=positions,
        correction=correction,
        valid=valid,
        direct_valid=observations.valid.copy(),
        direct_reliability=observations.reliability.copy(),
        solve_iterations=iterations,
        solve_relative_residual=residual,
    )


def _variant_summary(
    graph_initial: np.ndarray,
    graph_trajectory: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    reliability: np.ndarray,
    manual_tracks: np.ndarray | None,
    frame_indices: np.ndarray,
    train_selection: np.ndarray,
    future_selection: np.ndarray,
) -> tuple[dict[str, object], dict[str, object] | None]:
    discrepancy = dense_graph_error_by_frame(
        graph_trajectory, positions, valid, reliability
    )
    audit = None
    manual_error = None
    if manual_tracks is not None:
        audit = manual_track_association_audit(
            graph_initial,
            positions,
            valid,
            manual_tracks,
            frame_indices,
        )
        manual_error = np.asarray(audit["error_by_sampled_frame_m"], dtype=float)
    summary: dict[str, object] = {
        "direct_or_supported_vertex_fraction_by_sampled_frame": np.mean(
            valid, axis=1
        ).tolist(),
        "training_vertex_fraction": float(np.mean(valid[train_selection])),
        "future_vertex_fraction": float(np.mean(valid[future_selection])),
        "released_phystwin_discrepancy_by_sampled_frame_m": discrepancy.tolist(),
        "released_phystwin_training_discrepancy_mean_m": _mean_on_frames(
            discrepancy, train_selection
        ),
        "released_phystwin_future_discrepancy_mean_m": _mean_on_frames(
            discrepancy, future_selection
        ),
    }
    if audit is not None and manual_error is not None:
        summary["manual_identity_audit"] = {
            "available": True,
            **audit,
            "training_mean_m": _mean_on_frames(manual_error, train_selection),
            "future_mean_m": _mean_on_frames(manual_error, future_selection),
        }
    else:
        summary["manual_identity_audit"] = {
            "available": False,
            "reason": "gt_track_3d.pkl is absent",
        }
    return summary, audit


def assimilate_motioncrafter_case(
    case_dir: str | Path,
    raw_case_dir: str | Path,
    motioncrafter_npz_path: str | Path,
    output_dir: str | Path,
    *,
    config: AnonymousSceneFlowConfig,
    train_end_frame: int | None = None,
    additional_views: dict[int, str | Path] | None = None,
) -> dict[str, object]:
    """Run position/flow and graph-prior controls for one released case."""

    _validate_config(config)
    case_path = Path(case_dir)
    raw_path = Path(raw_case_dir)
    output = Path(output_dir)
    final_path = case_path / "final_data.pkl"
    baseline_path = case_path / "inference.pkl"
    optimal_path = case_path / "optimal_params.pkl"
    track_path = case_path / "gt_track_3d.pkl"
    split_path = case_path / "split.json"
    view_paths = {config.camera_index: Path(motioncrafter_npz_path)}
    for camera, path in (additional_views or {}).items():
        camera_index = int(camera)
        if camera_index < 0:
            raise ValueError("additional camera indices must be nonnegative")
        if camera_index in view_paths:
            raise ValueError(f"duplicate MotionCrafter camera {camera_index}")
        view_paths[camera_index] = Path(path)
    data = _load_pickle(final_path)
    baseline = np.asarray(_load_pickle(baseline_path), dtype=float)
    optimal = _load_pickle(optimal_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_end = int(split["test"][0]) if train_end_frame is None else train_end_frame
    frame_count = int(split["frame_len"])
    if not 1 < train_end < frame_count:
        raise ValueError("train_end_frame must leave a future interval")
    predictions = {
        camera: load_motioncrafter_prediction(path)
        for camera, path in sorted(view_paths.items())
    }
    prediction = predictions[config.camera_index]
    frame_indices = np.arange(0, frame_count, config.process_stride, dtype=np.int64)[
        : len(prediction.point_map)
    ]
    if len(frame_indices) != len(prediction.point_map):
        raise ValueError("process_stride does not explain MotionCrafter frame count")
    for camera, candidate in predictions.items():
        if candidate.point_map.shape != prediction.point_map.shape:
            raise ValueError(
                f"MotionCrafter camera {camera} shape does not match the primary view"
            )

    with np.load(raw_path / "pcd" / "0.npz") as pcd_archive:
        camera_points = np.asarray(pcd_archive["points"], dtype=float)
    if any(camera >= len(camera_points) for camera in view_paths):
        raise ValueError("camera_index exceeds raw point-cloud cameras")
    with (raw_path / "mask" / "processed_masks.pkl").open("rb") as handle:
        processed_masks = pickle.load(handle)
    transforms: dict[int, dict[str, object]] = {}
    aligned_predictions: dict[int, MotionCrafterPrediction] = {}
    object_masks_by_view: dict[int, np.ndarray] = {}
    for camera, camera_prediction in predictions.items():
        target_shape = camera_prediction.point_map.shape[1:3]
        object_masks = np.stack(
            [
                resample_cover_grid(
                    np.asarray(processed_masks[int(frame)][camera]["object"]),
                    target_shape,
                ).astype(bool)
                for frame in frame_indices
            ]
        )
        initial_world = resample_cover_grid(camera_points[camera], target_shape).astype(
            float
        )
        grid_y, grid_x = np.indices(target_shape)
        alignment_mask = (
            object_masks[0]
            & camera_prediction.valid_mask[0]
            & np.all(np.isfinite(camera_prediction.point_map[0]), axis=2)
            & np.all(np.isfinite(initial_world), axis=2)
            & (np.linalg.norm(initial_world, axis=2) > 1e-6)
            & (grid_y % config.alignment_stride_pixels == 0)
            & (grid_x % config.alignment_stride_pixels == 0)
        )
        transform = robust_similarity_transform(
            camera_prediction.point_map[0, alignment_mask],
            initial_world[alignment_mask],
            trim_fraction=config.alignment_trim_fraction,
            iterations=config.alignment_iterations,
        )
        transforms[camera] = transform
        aligned_predictions[camera] = align_motioncrafter_prediction(
            camera_prediction, transform
        )
        object_masks_by_view[camera] = object_masks

    observed = np.asarray(data["object_points"], dtype=float)
    surface_points = np.asarray(data["surface_points"], dtype=float)
    interior_points = np.asarray(data["interior_points"], dtype=float)
    structure_points = np.concatenate(
        (observed[0], surface_points, interior_points), axis=0
    )
    if baseline.shape[1] != len(structure_points):
        raise ValueError("released graph and baseline trajectory disagree")
    surface_count = len(observed[0]) + len(surface_points)
    graph = build_phystwin_spring_graph(
        structure_points,
        None,
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    surface_springs = graph.springs[np.all(graph.springs < surface_count, axis=1)]
    graph_initial = structure_points[:surface_count]
    sampled_baseline = baseline[frame_indices, :surface_count]
    position_config = replace(config, flow_strength=0.0)
    position_observations_by_view = {
        camera: associate_anonymous_scene_flow(
            aligned_predictions[camera],
            object_masks_by_view[camera],
            sampled_baseline,
            config=position_config,
        )
        for camera in sorted(aligned_predictions)
    }
    flow_observations_by_view = {
        camera: associate_anonymous_scene_flow(
            aligned_predictions[camera],
            object_masks_by_view[camera],
            sampled_baseline,
            config=config,
        )
        for camera in sorted(aligned_predictions)
    }
    position_observations = combine_framewise_graph_observations(
        position_observations_by_view,
        consistency_scale_m=config.multiview_consistency_scale_m,
        minimum_reliability=config.minimum_multiview_reliability,
    )
    flow_observations = combine_framewise_graph_observations(
        flow_observations_by_view,
        consistency_scale_m=config.multiview_consistency_scale_m,
        minimum_reliability=config.minimum_multiview_reliability,
    )
    position_graph = graph_regularized_state_observations(
        sampled_baseline,
        position_observations,
        surface_springs,
        prior_strength=config.graph_prior_strength,
        zero_prior_strength=config.graph_zero_prior_strength,
        ridge=config.graph_ridge,
        relative_tolerance=config.graph_solver_relative_tolerance,
        maximum_iterations=config.graph_solver_maximum_iterations,
        maximum_correction_m=config.maximum_graph_correction_m,
    )
    flow_graph = graph_regularized_state_observations(
        sampled_baseline,
        flow_observations,
        surface_springs,
        prior_strength=config.graph_prior_strength,
        zero_prior_strength=config.graph_zero_prior_strength,
        ridge=config.graph_ridge,
        relative_tolerance=config.graph_solver_relative_tolerance,
        maximum_iterations=config.graph_solver_maximum_iterations,
        maximum_correction_m=config.maximum_graph_correction_m,
    )
    manual_tracks = (
        None
        if not track_path.is_file()
        else np.asarray(_load_pickle(track_path), dtype=float)
    )
    train_selection = frame_indices < train_end
    future_selection = frame_indices >= train_end

    variants: dict[str, dict[str, object]] = {}
    for name, positions_array, valid_array, reliability_array in (
        (
            "released_phystwin",
            sampled_baseline,
            np.ones(sampled_baseline.shape[:2], dtype=bool),
            np.ones(sampled_baseline.shape[:2], dtype=float),
        ),
        (
            "position_only_direct",
            position_observations.positions,
            position_observations.valid,
            position_observations.reliability,
        ),
        (
            "position_flow_direct",
            flow_observations.positions,
            flow_observations.valid,
            flow_observations.reliability,
        ),
        (
            "position_only_graph",
            position_graph.positions,
            position_graph.valid,
            np.where(position_graph.valid, 1.0, 0.0),
        ),
        (
            "position_flow_graph",
            flow_graph.positions,
            flow_graph.valid,
            np.where(flow_graph.valid, 1.0, 0.0),
        ),
    ):
        variants[name], _ = _variant_summary(
            graph_initial,
            sampled_baseline,
            positions_array,
            valid_array,
            reliability_array,
            manual_tracks,
            frame_indices,
            train_selection,
            future_selection,
        )
    for name, direct_valid in (
        ("position_only_graph", position_observations.valid),
        ("position_flow_graph", flow_observations.valid),
    ):
        variants[name]["direct_vertex_fraction_by_sampled_frame"] = np.mean(
            direct_valid, axis=1
        ).tolist()
        variants[name]["training_direct_vertex_fraction"] = float(
            np.mean(direct_valid[train_selection])
        )
        variants[name]["future_direct_vertex_fraction"] = float(
            np.mean(direct_valid[future_selection])
        )

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "assimilation.npz"
    np.savez_compressed(
        archive_path,
        frame_indices=frame_indices.astype(np.int32),
        position_only_positions=position_observations.positions,
        position_only_valid=position_observations.valid,
        position_only_reliability=position_observations.reliability,
        position_only_measurement_mass=position_observations.measurement_mass,
        position_flow_positions=flow_observations.positions,
        position_flow_endpoints=flow_observations.flow_endpoints,
        position_flow_valid=flow_observations.valid,
        position_flow_endpoint_valid=flow_observations.flow_valid,
        position_flow_reliability=flow_observations.reliability,
        position_flow_endpoint_reliability=flow_observations.flow_reliability,
        position_flow_measurement_mass=flow_observations.measurement_mass,
        position_flow_entropy=flow_observations.normalized_entropy,
        position_flow_position_error_m=flow_observations.position_error_m,
        position_flow_endpoint_error_m=flow_observations.flow_endpoint_error_m,
        position_only_graph_positions=position_graph.positions,
        position_only_graph_valid=position_graph.valid,
        position_only_graph_correction=position_graph.correction,
        position_flow_graph_positions=flow_graph.positions,
        position_flow_graph_valid=flow_graph.valid,
        position_flow_graph_correction=flow_graph.correction,
    )
    transform_summaries: dict[str, object] = {}
    for camera, transform in transforms.items():
        transform_summary = {
            key: value
            for key, value in transform.items()
            if key not in {"inlier_mask", "all_pair_residual_m"}
        }
        transform_summary.update(
            {
                "linear": np.asarray(transform["linear"]).tolist(),
                "translation": np.asarray(transform["translation"]).tolist(),
                "rotation": np.asarray(transform["rotation"]).tolist(),
                "all_pair_residual_m": _distribution(
                    np.asarray(transform["all_pair_residual_m"])
                ),
            }
        )
        transform_summaries[str(camera)] = transform_summary
    result: dict[str, Any] = {
        "schema_version": 2,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case_path.name,
        "config": asdict(config),
        "contract": {
            "persistent_identity": "PhysTwin graph vertex IDs",
            "motioncrafter_role": "anonymous per-frame 3D position and forward-flow likelihood",
            "association_prior": "one fixed released PhysTwin trajectory for all variants",
            "reassociation": "independent at every frame; no MotionCrafter identity transport",
            "graph_regularization": "Laplacian innovation prior; direct and imputed support are reported separately",
            "future_use": "future MotionCrafter frames are reconstruction-only and never valid future-prediction inputs",
            "manual_tracks": "post-lock audit only; never used by alignment, assignment, or smoothing",
        },
        "software": {
            "motioncrafter_repository": MOTIONCRAFTER_REPOSITORY,
            "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        },
        "frame_indices": frame_indices.astype(int).tolist(),
        "train_end_frame": train_end,
        "alignment": {
            "view_count": len(transforms),
            "inlier_rmse_m": _distribution(
                np.asarray(
                    [transform["inlier_rmse_m"] for transform in transforms.values()]
                )
            ),
            "by_camera": transform_summaries,
        },
        "graph": {
            "surface_vertex_count": surface_count,
            "surface_spring_count": int(len(surface_springs)),
        },
        "measurements": {
            "position_only_sampled_count": position_observations.sampled_measurement_count.tolist(),
            "position_only_accepted_count": position_observations.accepted_measurement_count.tolist(),
            "position_flow_sampled_count": flow_observations.sampled_measurement_count.tolist(),
            "position_flow_accepted_count": flow_observations.accepted_measurement_count.tolist(),
            "position_flow_entropy": _distribution(
                flow_observations.normalized_entropy
            ),
            "position_flow_position_error_m": _distribution(
                flow_observations.position_error_m
            ),
            "position_flow_endpoint_error_m": _distribution(
                flow_observations.flow_endpoint_error_m
            ),
            "by_camera": {
                str(camera): {
                    "position_only_direct_vertex_fraction": float(
                        np.mean(position_observations_by_view[camera].valid)
                    ),
                    "position_flow_direct_vertex_fraction": float(
                        np.mean(flow_observations_by_view[camera].valid)
                    ),
                    "position_flow_position_error_m": _distribution(
                        flow_observations_by_view[camera].position_error_m
                    ),
                    "position_flow_endpoint_error_m": _distribution(
                        flow_observations_by_view[camera].flow_endpoint_error_m
                    ),
                }
                for camera in sorted(flow_observations_by_view)
            },
        },
        "variants": variants,
        "inputs": {
            "final_data": {
                "path": str(final_path.resolve()),
                "sha256": _sha256(final_path),
            },
            "baseline": {
                "path": str(baseline_path.resolve()),
                "sha256": _sha256(baseline_path),
            },
            "optimal_params": {
                "path": str(optimal_path.resolve()),
                "sha256": _sha256(optimal_path),
            },
            "manual_tracks": (
                None
                if not track_path.is_file()
                else {"path": str(track_path.resolve()), "sha256": _sha256(track_path)}
            ),
            "split": {"path": str(split_path.resolve()), "sha256": _sha256(split_path)},
            "motioncrafter_views": {
                str(camera): {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
                for camera, path in sorted(view_paths.items())
            },
            "raw_case_dir": str(raw_path.resolve()),
        },
        "outputs": {
            "assimilation_npz": str(archive_path.resolve()),
            "assimilation_npz_sha256": _sha256(archive_path),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(summary_path.resolve())
    return result
