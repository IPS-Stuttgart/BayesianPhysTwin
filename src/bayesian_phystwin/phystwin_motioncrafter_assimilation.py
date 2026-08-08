"""Physics-guided assimilation of anonymous MotionCrafter scene flow."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .mask_distance import interior_mask_distance
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
    load_phystwin_world_point_grid,
    manual_track_association_audit,
    resample_cover_grid,
    robust_similarity_transform,
)
from .phystwin_official_evaluation import _nearest_distances
from .pseudo_measurements import PseudoMeasurementBatch
from .robust_likelihood import RobustLikelihoodConfig, robust_mixture_likelihood


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
    graph_covariance_probes: int = 0
    graph_covariance_manual_track_audit: bool = False
    maximum_graph_correction_m: float = 0.01
    reliability_mode: str = "legacy"
    multiview_fusion_mode: str = "legacy_independent"
    correlation_block_pixels: int = 16
    boundary_reliability_scale_pixels: float = 8.0
    boundary_reliability_floor: float = 0.25
    observation_variance_floor_m2: float = 4e-6
    robust_outlier_variance_multiplier: float = 100.0
    robust_model_discrepancy_variance_m2: float = 0.0
    robust_probability_floor: float = 1e-6


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
    prior_reliability: np.ndarray | None = None
    flow_prior_reliability: np.ndarray | None = None
    observation_covariance_m2: np.ndarray | None = None
    flow_observation_covariance_m2: np.ndarray | None = None
    effective_sample_size: np.ndarray | None = None
    flow_effective_sample_size: np.ndarray | None = None
    contributor_count: np.ndarray | None = None


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
    marginal_variance_m2: np.ndarray | None = None


def _validate_config(config: AnonymousSceneFlowConfig) -> None:
    if config.camera_index < 0 or config.process_stride < 1:
        raise ValueError("camera_index must be nonnegative and stride positive")
    if (
        min(
            config.measurement_stride_pixels,
            config.alignment_stride_pixels,
            config.alignment_iterations,
            config.candidate_count,
            config.correlation_block_pixels,
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
            config.boundary_reliability_scale_pixels,
            config.observation_variance_floor_m2,
            config.robust_outlier_variance_multiplier,
            config.robust_probability_floor,
        )
        <= 0.0
    ):
        raise ValueError("likelihood scales and graph settings must be positive")
    if (
        min(
            config.flow_strength,
            config.entropy_strength,
            config.graph_zero_prior_strength,
            config.robust_model_discrepancy_variance_m2,
        )
        < 0.0
    ):
        raise ValueError("likelihood strengths and zero prior must be nonnegative")
    if not 0.0 < config.minimum_multiview_reliability <= 1.0:
        raise ValueError("minimum_multiview_reliability must lie in (0, 1]")
    if config.reliability_mode not in {"legacy", "decoupled_robust"}:
        raise ValueError("reliability_mode must be legacy or decoupled_robust")
    if config.multiview_fusion_mode not in {
        "legacy_independent",
        "covariance_intersection",
    }:
        raise ValueError(
            "multiview_fusion_mode must be legacy_independent or "
            "covariance_intersection"
        )
    if not 0.0 <= config.boundary_reliability_floor <= 1.0:
        raise ValueError("boundary_reliability_floor must lie in [0, 1]")
    if config.robust_outlier_variance_multiplier <= 1.0:
        raise ValueError("robust_outlier_variance_multiplier must exceed one")
    if not 0.0 < config.robust_probability_floor < 0.5:
        raise ValueError("robust_probability_floor must lie in (0, 0.5)")
    if config.graph_solver_maximum_iterations < 1:
        raise ValueError("graph_solver_maximum_iterations must be positive")
    if config.graph_covariance_probes < 0:
        raise ValueError("graph_covariance_probes must be nonnegative")
    if config.graph_covariance_probes and config.graph_covariance_manual_track_audit:
        raise ValueError("choose graph covariance probes or manual-track audit")


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

    mass: np.ndarray = np.zeros(vertex_count, dtype=float)
    evidence_mass: np.ndarray = np.zeros(vertex_count, dtype=float)
    numerator: np.ndarray = np.zeros((vertex_count, 3), dtype=float)
    entropy_numerator: np.ndarray = np.zeros(vertex_count, dtype=float)
    position_numerator: np.ndarray = np.zeros(vertex_count, dtype=float)
    flow_numerator: np.ndarray = np.zeros(vertex_count, dtype=float)
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


def _mask_boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Return the canonical border-aware Euclidean interior distance."""

    return interior_mask_distance(mask)


def _measurement_covariance(
    covariance: np.ndarray | None,
    frame: int,
    pixels: np.ndarray,
    *,
    fallback_variance_m2: float,
) -> np.ndarray:
    if covariance is None:
        return np.broadcast_to(
            fallback_variance_m2 * np.eye(3),
            (len(pixels), 3, 3),
        ).copy()
    selected = np.asarray(covariance[frame, pixels[:, 0], pixels[:, 1]], dtype=float)
    return 0.5 * (selected + np.swapaxes(selected, -1, -2))


def _candidate_mixture_covariance(
    candidate_points: np.ndarray,
    candidate_weights: np.ndarray,
) -> np.ndarray:
    expected = np.sum(candidate_weights[..., None] * candidate_points, axis=1)
    offset = candidate_points - expected[:, None]
    return np.einsum("mk,mki,mkj->mij", candidate_weights, offset, offset)


def _clustered_vertex_measurements(
    *,
    vertex_count: int,
    candidate_indices: np.ndarray,
    candidate_weights: np.ndarray,
    measurement_values: np.ndarray,
    measurement_covariance_m2: np.ndarray,
    assignment_covariance_m2: np.ndarray,
    measurement_prior_reliability: np.ndarray,
    measurement_entropy: np.ndarray,
    position_error_m: np.ndarray,
    flow_error_m: np.ndarray,
    pixels: np.ndarray,
    image_width: int,
    correlation_block_pixels: int,
    minimum_observation_mass: float,
    observation_variance_floor_m2: float,
    predicted_values: np.ndarray,
    robust_config: RobustLikelihoodConfig,
) -> tuple[np.ndarray, ...]:
    """Aggregate dense pixels through capped spatial clusters and one robust update."""

    indices = np.asarray(candidate_indices, dtype=np.int64)
    assignment = np.asarray(candidate_weights, dtype=float)
    values = np.asarray(measurement_values, dtype=float)
    source_covariance = np.asarray(measurement_covariance_m2, dtype=float)
    assignment_covariance = np.asarray(assignment_covariance_m2, dtype=float)
    prior = np.asarray(measurement_prior_reliability, dtype=float)
    if indices.shape != assignment.shape or indices.ndim != 2:
        raise ValueError("candidate indices and weights must have shape (M, K)")
    measurement_count, candidate_count = indices.shape
    if values.shape != (measurement_count, 3):
        raise ValueError("measurement_values must have shape (M, 3)")
    if source_covariance.shape != (measurement_count, 3, 3):
        raise ValueError("measurement covariance must have shape (M, 3, 3)")
    if assignment_covariance.shape != source_covariance.shape:
        raise ValueError("assignment covariance must match source covariance")
    if prior.shape != (measurement_count,):
        raise ValueError("measurement prior reliability must have shape (M,)")

    flat_vertex = indices.ravel()
    flat_assignment = assignment.ravel()
    active = flat_assignment > 0.0
    flat_vertex = flat_vertex[active]
    flat_assignment = flat_assignment[active]
    repeated_prior = np.repeat(prior, candidate_count)[active]
    flat_evidence = flat_assignment * repeated_prior
    repeated_values = np.repeat(values, candidate_count, axis=0)[active]
    repeated_source_covariance = np.repeat(source_covariance, candidate_count, axis=0)[
        active
    ]
    repeated_assignment_covariance = np.repeat(
        assignment_covariance, candidate_count, axis=0
    )[active]

    blocks_x = int(np.ceil(image_width / correlation_block_pixels))
    block = (pixels[:, 0] // correlation_block_pixels) * blocks_x + pixels[
        :, 1
    ] // correlation_block_pixels
    block_count = int(np.max(block)) + 1
    repeated_block = np.repeat(block, candidate_count)[active]
    group_key = flat_vertex * block_count + repeated_block
    unique_key, inverse = np.unique(group_key, return_inverse=True)
    group_count = len(unique_key)
    group_assignment_mass = np.bincount(
        inverse, weights=flat_assignment, minlength=group_count
    )
    group_evidence_mass = np.bincount(
        inverse, weights=flat_evidence, minlength=group_count
    )
    group_mean_numerator: np.ndarray = np.zeros((group_count, 3), dtype=float)
    np.add.at(
        group_mean_numerator,
        inverse,
        flat_evidence[:, None] * repeated_values,
    )
    group_mean = group_mean_numerator / np.maximum(group_evidence_mass[:, None], 1e-15)
    source_second = repeated_source_covariance + np.einsum(
        "mi,mj->mij", repeated_values, repeated_values
    )
    group_source_second: np.ndarray = np.zeros((group_count, 3, 3), dtype=float)
    np.add.at(
        group_source_second,
        inverse,
        flat_evidence[:, None, None] * source_second,
    )
    group_source_second /= np.maximum(group_evidence_mass[:, None, None], 1e-15)
    group_assignment_covariance: np.ndarray = np.zeros((group_count, 3, 3), dtype=float)
    np.add.at(
        group_assignment_covariance,
        inverse,
        flat_evidence[:, None, None] * repeated_assignment_covariance,
    )
    group_assignment_covariance /= np.maximum(group_evidence_mass[:, None, None], 1e-15)

    group_vertex = unique_key // block_count
    cluster_weight = np.minimum(group_evidence_mass, 1.0)
    cluster_assignment_weight = np.minimum(group_assignment_mass, 1.0)
    effective_mass = np.bincount(
        group_vertex, weights=cluster_weight, minlength=vertex_count
    )
    cluster_weight_squared = np.bincount(
        group_vertex, weights=np.square(cluster_weight), minlength=vertex_count
    )
    effective_sample_size = np.divide(
        np.square(effective_mass),
        cluster_weight_squared,
        out=np.zeros(vertex_count, dtype=float),
        where=cluster_weight_squared > 0.0,
    )
    mass = np.bincount(flat_vertex, weights=flat_assignment, minlength=vertex_count)
    mean_numerator: np.ndarray = np.zeros((vertex_count, 3), dtype=float)
    np.add.at(mean_numerator, group_vertex, cluster_weight[:, None] * group_mean)
    mean = mean_numerator / np.maximum(effective_mass[:, None], 1e-15)
    second_numerator: np.ndarray = np.zeros((vertex_count, 3, 3), dtype=float)
    np.add.at(
        second_numerator,
        group_vertex,
        cluster_weight[:, None, None] * group_source_second,
    )
    source_covariance = second_numerator / np.maximum(
        effective_mass[:, None, None], 1e-15
    ) - np.einsum("ni,nj->nij", mean, mean)
    source_covariance /= np.maximum(effective_sample_size[:, None, None], 1.0)
    assignment_numerator: np.ndarray = np.zeros((vertex_count, 3, 3), dtype=float)
    np.add.at(
        assignment_numerator,
        group_vertex,
        cluster_weight[:, None, None] * group_assignment_covariance,
    )
    vertex_assignment_covariance = assignment_numerator / np.maximum(
        effective_mass[:, None, None], 1e-15
    )
    covariance = source_covariance + vertex_assignment_covariance
    covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    covariance += observation_variance_floor_m2 * np.eye(3)[None]

    group_prior = np.divide(
        group_evidence_mass,
        group_assignment_mass,
        out=np.zeros(group_count, dtype=float),
        where=group_assignment_mass > 0.0,
    )
    prior_denominator = np.bincount(
        group_vertex,
        weights=cluster_assignment_weight,
        minlength=vertex_count,
    )
    prior_numerator = np.bincount(
        group_vertex,
        weights=cluster_assignment_weight * group_prior,
        minlength=vertex_count,
    )
    vertex_prior = np.divide(
        prior_numerator,
        prior_denominator,
        out=np.zeros(vertex_count, dtype=float),
        where=prior_denominator > 0.0,
    )
    valid = effective_mass >= minimum_observation_mass
    posterior_reliability: np.ndarray = np.zeros(vertex_count, dtype=float)
    if np.any(valid):
        likelihood = robust_mixture_likelihood(
            PseudoMeasurementBatch(
                observed=mean[valid],
                predicted=np.asarray(predicted_values, dtype=float)[valid],
                variance=np.diagonal(covariance[valid], axis1=1, axis2=2),
            ),
            prior_reliability=vertex_prior[valid],
            config=robust_config,
        )
        posterior_reliability[valid] = likelihood.posterior_inlier_probability

    diagnostic_weight = flat_evidence
    diagnostic_denominator = np.bincount(
        flat_vertex, weights=diagnostic_weight, minlength=vertex_count
    )

    def aggregate_diagnostic(measurement: np.ndarray) -> np.ndarray:
        repeated = np.repeat(np.asarray(measurement, dtype=float), candidate_count)[
            active
        ]
        finite = np.isfinite(repeated)
        numerator = np.bincount(
            flat_vertex[finite],
            weights=diagnostic_weight[finite] * repeated[finite],
            minlength=vertex_count,
        )
        denominator = np.bincount(
            flat_vertex[finite],
            weights=diagnostic_weight[finite],
            minlength=vertex_count,
        )
        return np.divide(
            numerator,
            denominator,
            out=np.full(vertex_count, np.nan, dtype=float),
            where=denominator > 0.0,
        )

    entropy = aggregate_diagnostic(measurement_entropy)
    position_error = aggregate_diagnostic(position_error_m)
    flow_error = aggregate_diagnostic(flow_error_m)
    mean[~valid] = np.nan
    covariance[~valid] = np.nan
    vertex_prior[~valid] = 0.0
    posterior_reliability[~valid] = 0.0
    return (
        mean,
        valid,
        posterior_reliability,
        vertex_prior,
        mass,
        covariance,
        effective_sample_size,
        entropy,
        position_error,
        flow_error,
        diagnostic_denominator,
    )


def _associate_anonymous_scene_flow_decoupled(
    prediction: MotionCrafterPrediction,
    object_masks: np.ndarray,
    graph_trajectory: np.ndarray,
    *,
    config: AnonymousSceneFlowConfig,
) -> FramewiseGraphObservations:
    masks = np.asarray(object_masks, dtype=bool)
    graph = np.asarray(graph_trajectory, dtype=float)
    if masks.shape != prediction.valid_mask.shape:
        raise ValueError("object_masks must match MotionCrafter frame/image shape")
    if graph.ndim != 3 or graph.shape[2] != 3 or len(graph) != len(masks):
        raise ValueError("graph_trajectory must have matching shape (T, N, 3)")
    if not np.all(np.isfinite(graph)):
        raise ValueError("graph_trajectory must be finite")
    frame_count, height, width = masks.shape
    vertex_count = graph.shape[1]
    candidate_count = min(config.candidate_count, vertex_count)
    shape = (frame_count, vertex_count)
    positions = np.full(shape + (3,), np.nan, dtype=np.float32)
    endpoints = np.full_like(positions, np.nan)
    valid = np.zeros(shape, dtype=bool)
    flow_valid = np.zeros(shape, dtype=bool)
    reliability = np.zeros(shape, dtype=np.float32)
    flow_reliability = np.zeros(shape, dtype=np.float32)
    prior_reliability = np.zeros(shape, dtype=np.float32)
    flow_prior_reliability = np.zeros(shape, dtype=np.float32)
    mass = np.zeros(shape, dtype=np.float32)
    flow_mass = np.zeros(shape, dtype=np.float32)
    covariance = np.full(shape + (3, 3), np.nan, dtype=np.float32)
    flow_covariance = np.full_like(covariance, np.nan)
    effective_sample_size = np.zeros(shape, dtype=np.float32)
    flow_effective_sample_size = np.zeros(shape, dtype=np.float32)
    entropy = np.full(shape, np.nan, dtype=np.float32)
    position_error = np.full(shape, np.nan, dtype=np.float32)
    flow_error = np.full(shape, np.nan, dtype=np.float32)
    contributor_count = np.zeros(shape, dtype=np.float32)
    sampled_count = np.zeros(frame_count, dtype=np.int32)
    accepted_count = np.zeros(frame_count, dtype=np.int32)
    grid_y, grid_x = np.indices((height, width))
    stride_mask = (grid_y % config.measurement_stride_pixels == 0) & (
        grid_x % config.measurement_stride_pixels == 0
    )
    robust_config = RobustLikelihoodConfig(
        outlier_variance_multiplier=config.robust_outlier_variance_multiplier,
        model_discrepancy_variance=(config.robust_model_discrepancy_variance_m2),
        probability_floor=config.robust_probability_floor,
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
        flow_is_usable: np.ndarray = np.zeros(len(points), dtype=bool)
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
        pixels = pixels[accepted]
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
        boundary_distance = _mask_boundary_distance(masks[frame])[
            pixels[:, 0], pixels[:, 1]
        ]
        boundary_reliability = config.boundary_reliability_floor + (
            1.0 - config.boundary_reliability_floor
        ) * (
            1.0 - np.exp(-boundary_distance / config.boundary_reliability_scale_pixels)
        )
        source_confidence = (
            np.ones(len(pixels), dtype=float)
            if prediction.source_confidence is None
            else np.clip(
                prediction.source_confidence[frame, pixels[:, 0], pixels[:, 1]].astype(
                    float
                ),
                0.0,
                1.0,
            )
        )
        measurement_prior = np.clip(
            source_confidence
            * boundary_reliability
            * np.exp(-config.entropy_strength * measurement_entropy),
            config.robust_probability_floor,
            1.0 - config.robust_probability_floor,
        )
        expected_position_error = np.sum(candidate_weights * candidate_distance, axis=1)
        expected_flow_error: np.ndarray = np.full(len(points), np.nan, dtype=float)
        expected_flow_error[flow_is_usable] = np.sum(
            candidate_weights[flow_is_usable] * candidate_flow_distance[flow_is_usable],
            axis=1,
        )
        point_covariance = _measurement_covariance(
            prediction.point_covariance_m2,
            frame,
            pixels,
            fallback_variance_m2=config.position_scale_m**2,
        )
        candidate_points = graph[frame][candidate_indices]
        assignment_covariance = _candidate_mixture_covariance(
            candidate_points, candidate_weights
        )
        point_result = _clustered_vertex_measurements(
            vertex_count=vertex_count,
            candidate_indices=candidate_indices,
            candidate_weights=candidate_weights,
            measurement_values=points,
            measurement_covariance_m2=point_covariance,
            assignment_covariance_m2=assignment_covariance,
            measurement_prior_reliability=measurement_prior,
            measurement_entropy=measurement_entropy,
            position_error_m=expected_position_error,
            flow_error_m=expected_flow_error,
            pixels=pixels,
            image_width=width,
            correlation_block_pixels=config.correlation_block_pixels,
            minimum_observation_mass=config.minimum_observation_mass,
            observation_variance_floor_m2=config.observation_variance_floor_m2,
            predicted_values=graph[frame],
            robust_config=robust_config,
        )
        (
            frame_positions,
            frame_valid,
            frame_reliability,
            frame_prior,
            frame_mass,
            frame_covariance,
            frame_ess,
            frame_entropy,
            frame_position_error,
            _,
            _,
        ) = point_result
        positions[frame] = frame_positions.astype(np.float32)
        valid[frame] = frame_valid
        reliability[frame] = frame_reliability.astype(np.float32)
        prior_reliability[frame] = frame_prior.astype(np.float32)
        mass[frame] = frame_mass.astype(np.float32)
        covariance[frame] = frame_covariance.astype(np.float32)
        effective_sample_size[frame] = frame_ess.astype(np.float32)
        entropy[frame] = frame_entropy.astype(np.float32)
        position_error[frame] = frame_position_error.astype(np.float32)
        if prediction.contributors is not None:
            selected_contributors = prediction.contributors[
                frame, pixels[:, 0], pixels[:, 1]
            ].astype(float)
            contributor_numerator = np.zeros(vertex_count, dtype=float)
            contributor_denominator = np.zeros(vertex_count, dtype=float)
            for candidate in range(candidate_count):
                np.add.at(
                    contributor_numerator,
                    candidate_indices[:, candidate],
                    candidate_weights[:, candidate] * selected_contributors,
                )
                np.add.at(
                    contributor_denominator,
                    candidate_indices[:, candidate],
                    candidate_weights[:, candidate],
                )
            contributor_count[frame] = np.divide(
                contributor_numerator,
                contributor_denominator,
                out=np.zeros(vertex_count, dtype=float),
                where=contributor_denominator > 0.0,
            ).astype(np.float32)

        if not np.any(flow_is_usable):
            continue
        usable = flow_is_usable
        flow_candidate_indices = candidate_indices[usable]
        flow_candidate_weights = candidate_weights[usable]
        flow_pixels = pixels[usable]
        flow_point_covariance = point_covariance[usable]
        raw_flow_covariance = _measurement_covariance(
            prediction.flow_covariance_m2,
            frame,
            flow_pixels,
            fallback_variance_m2=config.flow_scale_m**2,
        )
        # With unknown point/flow cross-covariance, 2(Cx + Cf) is a PSD upper bound.
        endpoint_covariance = 2.0 * (flow_point_covariance + raw_flow_covariance)
        endpoint_candidates = graph[min(frame + 1, frame_count - 1)][
            flow_candidate_indices
        ]
        endpoint_assignment_covariance = _candidate_mixture_covariance(
            endpoint_candidates, flow_candidate_weights
        )
        flow_result = _clustered_vertex_measurements(
            vertex_count=vertex_count,
            candidate_indices=flow_candidate_indices,
            candidate_weights=flow_candidate_weights,
            measurement_values=measurement_endpoints[usable],
            measurement_covariance_m2=endpoint_covariance,
            assignment_covariance_m2=endpoint_assignment_covariance,
            measurement_prior_reliability=measurement_prior[usable],
            measurement_entropy=measurement_entropy[usable],
            position_error_m=expected_position_error[usable],
            flow_error_m=expected_flow_error[usable],
            pixels=flow_pixels,
            image_width=width,
            correlation_block_pixels=config.correlation_block_pixels,
            minimum_observation_mass=config.minimum_observation_mass,
            observation_variance_floor_m2=config.observation_variance_floor_m2,
            predicted_values=graph[min(frame + 1, frame_count - 1)],
            robust_config=robust_config,
        )
        (
            frame_endpoints,
            frame_flow_valid,
            frame_flow_reliability,
            frame_flow_prior,
            frame_flow_mass,
            frame_flow_covariance,
            frame_flow_ess,
            _,
            _,
            frame_flow_error,
            _,
        ) = flow_result
        endpoints[frame] = frame_endpoints.astype(np.float32)
        flow_valid[frame] = frame_flow_valid
        flow_reliability[frame] = frame_flow_reliability.astype(np.float32)
        flow_prior_reliability[frame] = frame_flow_prior.astype(np.float32)
        flow_mass[frame] = frame_flow_mass.astype(np.float32)
        flow_covariance[frame] = frame_flow_covariance.astype(np.float32)
        flow_effective_sample_size[frame] = frame_flow_ess.astype(np.float32)
        flow_error[frame] = frame_flow_error.astype(np.float32)

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
        prior_reliability=prior_reliability,
        flow_prior_reliability=flow_prior_reliability,
        observation_covariance_m2=covariance,
        flow_observation_covariance_m2=flow_covariance,
        effective_sample_size=effective_sample_size,
        flow_effective_sample_size=flow_effective_sample_size,
        contributor_count=(
            contributor_count if prediction.contributors is not None else None
        ),
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
    if config.reliability_mode == "decoupled_robust":
        return _associate_anonymous_scene_flow_decoupled(
            prediction,
            object_masks,
            graph_trajectory,
            config=config,
        )
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
        flow_is_usable: np.ndarray = np.zeros(len(points), dtype=bool)
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
        expected_flow_error: np.ndarray = np.full(len(points), np.nan, dtype=float)
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


def _positive_definite_covariance(covariance: np.ndarray) -> np.ndarray:
    values = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.swapaxes(np.asarray(covariance, dtype=float), -1, -2)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    floor = np.maximum(np.max(eigenvalues, axis=-1, keepdims=True) * 1e-9, 1e-12)
    eigenvalues = np.maximum(eigenvalues, floor)
    return np.einsum("...ik,...k,...jk->...ij", eigenvectors, eigenvalues, eigenvectors)


def _covariance_intersection_pair(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
    *,
    grid_size: int = 21,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuse two estimates without assuming their cross-correlation is known."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    mean_one = np.asarray(first_mean, dtype=float)
    mean_two = np.asarray(second_mean, dtype=float)
    covariance_one = _positive_definite_covariance(first_covariance)
    covariance_two = _positive_definite_covariance(second_covariance)
    if mean_one.shape != (3,) or mean_two.shape != (3,):
        raise ValueError("covariance-intersection means must have shape (3,)")
    if covariance_one.shape != (3, 3) or covariance_two.shape != (3, 3):
        raise ValueError("covariance-intersection covariances must have shape (3, 3)")
    information_one = np.linalg.inv(covariance_one)
    information_two = np.linalg.inv(covariance_two)
    best_score = np.inf
    best_weight = 0.5
    best_covariance = covariance_one
    for weight in np.linspace(0.0, 1.0, grid_size):
        information = weight * information_one + (1.0 - weight) * information_two
        candidate_covariance = np.linalg.inv(information)
        _, score = np.linalg.slogdet(candidate_covariance)
        if score < best_score - 1e-12:
            best_score = float(score)
            best_weight = float(weight)
            best_covariance = candidate_covariance
    information_vector = (
        best_weight * information_one @ mean_one
        + (1.0 - best_weight) * information_two @ mean_two
    )
    best_mean = best_covariance @ information_vector
    return best_mean, best_covariance, best_weight


def _fuse_multiview_values_covariance_intersection(
    values: np.ndarray,
    valid: np.ndarray,
    reliability: np.ndarray,
    prior_reliability: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    consistency_scale_m: float,
    minimum_reliability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measurements = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    posterior = np.asarray(reliability, dtype=float)
    prior = np.asarray(prior_reliability, dtype=float)
    covariance = np.asarray(covariance_m2, dtype=float)
    expected_covariance_shape = measurements.shape[:3] + (3, 3)
    if measurements.ndim != 4 or measurements.shape[-1] != 3:
        raise ValueError("multiview values must have shape (V, T, N, 3)")
    if mask.shape != measurements.shape[:3]:
        raise ValueError("multiview validity shape disagrees with values")
    if posterior.shape != mask.shape or prior.shape != mask.shape:
        raise ValueError("multiview reliability shapes disagree")
    if covariance.shape != expected_covariance_shape:
        raise ValueError("multiview covariance shape disagrees with values")

    base_weight = np.where(mask, np.clip(posterior, 0.0, 1.0), 0.0)
    denominator = np.sum(base_weight, axis=0)
    center = np.zeros(measurements.shape[1:], dtype=float)
    available = denominator > 0.0
    center[available] = (
        np.sum(
            base_weight[..., None] * np.where(mask[..., None], measurements, 0.0),
            axis=0,
        )[available]
        / denominator[available, None]
    )
    effective_weight = base_weight.copy()
    consistency = np.where(mask, 1.0, 0.0)
    for _ in range(2):
        residual = np.linalg.norm(measurements - center[None], axis=3)
        consistency = np.where(
            mask,
            np.exp(
                np.clip(
                    -0.5 * np.square(residual / consistency_scale_m),
                    -50.0,
                    0.0,
                )
            ),
            0.0,
        )
        effective_weight = base_weight * consistency
        denominator = np.sum(effective_weight, axis=0)
        available = denominator > 0.0
        center[available] = (
            np.sum(
                effective_weight[..., None]
                * np.where(mask[..., None], measurements, 0.0),
                axis=0,
            )[available]
            / denominator[available, None]
        )

    output = np.full(measurements.shape[1:], np.nan, dtype=np.float32)
    output_covariance = np.full(
        measurements.shape[1:-1] + (3, 3), np.nan, dtype=np.float32
    )
    output_valid = np.zeros(mask.shape[1:], dtype=bool)
    output_reliability = np.zeros(mask.shape[1:], dtype=np.float32)
    output_prior = np.zeros(mask.shape[1:], dtype=np.float32)
    for frame, vertex in np.argwhere(np.any(mask, axis=0)):
        active_views = np.flatnonzero(mask[:, frame, vertex])
        active_views = active_views[effective_weight[active_views, frame, vertex] > 0.0]
        if len(active_views) == 0:
            continue
        first = int(active_views[0])
        fused_mean = measurements[first, frame, vertex]
        fused_covariance = covariance[first, frame, vertex] / max(
            consistency[first, frame, vertex], 1e-6
        )
        for view in active_views[1:]:
            fused_mean, fused_covariance, _ = _covariance_intersection_pair(
                fused_mean,
                fused_covariance,
                measurements[view, frame, vertex],
                covariance[view, frame, vertex]
                / max(consistency[view, frame, vertex], 1e-6),
            )
        combined_reliability = float(
            np.max(effective_weight[active_views, frame, vertex])
        )
        if combined_reliability < minimum_reliability:
            continue
        output[frame, vertex] = fused_mean.astype(np.float32)
        output_covariance[frame, vertex] = fused_covariance.astype(np.float32)
        output_valid[frame, vertex] = True
        output_reliability[frame, vertex] = combined_reliability
        output_prior[frame, vertex] = float(
            np.max(
                prior[active_views, frame, vertex]
                * consistency[active_views, frame, vertex]
            )
        )
    return (
        output,
        output_valid,
        output_reliability,
        output_prior,
        output_covariance,
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
    fusion_mode: str = "legacy_independent",
) -> FramewiseGraphObservations:
    """Reliability-fuse calibrated views and suppress cross-view outliers."""

    if not observations_by_view:
        raise ValueError("at least one view is required")
    if consistency_scale_m <= 0.0 or not 0.0 < minimum_reliability <= 1.0:
        raise ValueError("invalid multiview fusion settings")
    if fusion_mode not in {"legacy_independent", "covariance_intersection"}:
        raise ValueError("unknown multiview fusion mode")
    ordered = [observations_by_view[key] for key in sorted(observations_by_view)]
    shapes = {observation.positions.shape for observation in ordered}
    if len(shapes) != 1:
        raise ValueError("all views must share graph observation shape")
    if len(ordered) == 1:
        return ordered[0]

    if fusion_mode == "covariance_intersection":
        if any(
            observation.observation_covariance_m2 is None
            or observation.flow_observation_covariance_m2 is None
            or observation.prior_reliability is None
            or observation.flow_prior_reliability is None
            for observation in ordered
        ):
            raise ValueError(
                "covariance-intersection fusion requires decoupled covariance "
                "and prior-reliability fields"
            )
        (
            positions,
            valid,
            reliability,
            prior_reliability,
            position_covariance,
            position_weight,
        ) = _fuse_multiview_values_covariance_intersection(
            np.stack([observation.positions for observation in ordered]),
            np.stack([observation.valid for observation in ordered]),
            np.stack([observation.reliability for observation in ordered]),
            np.stack([observation.prior_reliability for observation in ordered]),
            np.stack(
                [observation.observation_covariance_m2 for observation in ordered]
            ),
            consistency_scale_m=consistency_scale_m,
            minimum_reliability=minimum_reliability,
        )
        (
            endpoints,
            flow_valid,
            flow_reliability,
            flow_prior_reliability,
            flow_covariance,
            flow_weight,
        ) = _fuse_multiview_values_covariance_intersection(
            np.stack([observation.flow_endpoints for observation in ordered]),
            np.stack([observation.flow_valid for observation in ordered]),
            np.stack([observation.flow_reliability for observation in ordered]),
            np.stack([observation.flow_prior_reliability for observation in ordered]),
            np.stack(
                [observation.flow_observation_covariance_m2 for observation in ordered]
            ),
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
            measurement_mass=np.max(
                np.stack([observation.measurement_mass for observation in ordered]),
                axis=0,
            ).astype(np.float32),
            flow_measurement_mass=np.max(
                np.stack(
                    [observation.flow_measurement_mass for observation in ordered]
                ),
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
                np.stack(
                    [observation.flow_endpoint_error_m for observation in ordered]
                ),
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
            prior_reliability=prior_reliability,
            flow_prior_reliability=flow_prior_reliability,
            observation_covariance_m2=position_covariance,
            flow_observation_covariance_m2=flow_covariance,
            effective_sample_size=np.max(
                np.stack(
                    [observation.effective_sample_size for observation in ordered]
                ),
                axis=0,
            ).astype(np.float32),
            flow_effective_sample_size=np.max(
                np.stack(
                    [observation.flow_effective_sample_size for observation in ordered]
                ),
                axis=0,
            ).astype(np.float32),
            contributor_count=(
                None
                if all(observation.contributor_count is None for observation in ordered)
                else np.max(
                    np.stack(
                        [
                            np.zeros_like(ordered[0].measurement_mass)
                            if observation.contributor_count is None
                            else observation.contributor_count
                            for observation in ordered
                        ]
                    ),
                    axis=0,
                ).astype(np.float32)
            ),
        )

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
    parent: np.ndarray = np.arange(node_count, dtype=np.int64)

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
    covariance_probes: int = 0,
    covariance_node_indices: np.ndarray | None = None,
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
        or covariance_probes < 0
        or maximum_correction_m <= 0.0
    ):
        raise ValueError("graph smoothing settings must be positive")
    node_count = graph.shape[1]
    laplacian = normalized_spring_laplacian(node_count, edges)
    components = _spring_components(node_count, edges)
    positions = np.full_like(graph, np.nan, dtype=np.float32)
    correction = np.full_like(graph, np.nan, dtype=np.float32)
    valid = np.zeros(graph.shape[:2], dtype=bool)
    iterations: np.ndarray = np.zeros((len(graph), 3), dtype=np.int32)
    residual: np.ndarray = np.full((len(graph), 3), np.nan, dtype=np.float64)
    marginal_variance = (
        np.full(graph.shape[:2], np.nan, dtype=np.float64)
        if covariance_probes > 0 or covariance_node_indices is not None
        else None
    )

    for frame in range(len(graph)):
        direct = observations.valid[frame]
        if not np.any(direct):
            continue
        innovation = np.zeros((node_count, 3), dtype=float)
        innovation[direct] = (
            observations.positions[frame, direct] - graph[frame, direct]
        )
        variance = np.ones(node_count, dtype=float)
        if observations.observation_covariance_m2 is None:
            variance[direct] = 1.0 / np.maximum(
                observations.reliability[frame, direct], 1e-6
            )
        else:
            metric_covariance = np.asarray(
                observations.observation_covariance_m2[frame, direct],
                dtype=float,
            )
            metric_variance = np.max(np.linalg.eigvalsh(metric_covariance), axis=1)
            variance[direct] = metric_variance / np.maximum(
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
            covariance_probes=covariance_probes,
            covariance_seed=20260711 + frame,
            covariance_indices=covariance_node_indices,
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
        iterations[frame] = np.asarray(posterior.solve_iterations[:3], dtype=np.int32)
        residual[frame] = np.asarray(
            posterior.solve_relative_residuals[:3], dtype=float
        )
        if marginal_variance is not None:
            marginal_variance[frame] = np.maximum(
                np.asarray(posterior.marginal_variance, dtype=float), 1e-12
            )

    return GraphRegularizedObservations(
        positions=positions,
        correction=correction,
        valid=valid,
        direct_valid=observations.valid.copy(),
        direct_reliability=observations.reliability.copy(),
        solve_iterations=iterations,
        solve_relative_residual=residual,
        marginal_variance_m2=marginal_variance,
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
    object_points: np.ndarray,
    object_visibilities: np.ndarray,
    observation_covariance_m2: np.ndarray | None = None,
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
    chamfer_by_frame: np.ndarray = np.full(len(frame_indices), np.nan, dtype=float)
    points = np.asarray(object_points, dtype=float)
    visibility = np.asarray(object_visibilities, dtype=bool)
    for output_frame, source_frame in enumerate(frame_indices):
        observed = points[source_frame, visibility[source_frame]]
        predicted = np.asarray(positions[output_frame], dtype=float)
        predicted = predicted[
            np.asarray(valid[output_frame], dtype=bool)
            & np.all(np.isfinite(predicted), axis=1)
        ]
        observed = observed[np.all(np.isfinite(observed), axis=1)]
        if len(predicted) and len(observed):
            distance, _ = _nearest_distances(predicted, observed, p=1)
            chamfer_by_frame[output_frame] = float(np.mean(distance))
    summary.update(
        {
            "visible_to_supported_surface_chamfer_by_sampled_frame_m": (
                chamfer_by_frame.tolist()
            ),
            "visible_to_supported_surface_chamfer_training_mean_m": (
                _mean_on_frames(chamfer_by_frame, train_selection)
            ),
            "visible_to_supported_surface_chamfer_future_mean_m": (
                _mean_on_frames(chamfer_by_frame, future_selection)
            ),
        }
    )
    if audit is not None and manual_error is not None and manual_tracks is not None:
        summary["manual_identity_audit"] = {
            "available": True,
            **audit,
            "training_mean_m": _mean_on_frames(manual_error, train_selection),
            "future_mean_m": _mean_on_frames(manual_error, future_selection),
        }
        if observation_covariance_m2 is not None:
            track_indices = np.asarray(audit["graph_vertex_indices"], dtype=np.int64)
            initial_track_mask = np.all(np.isfinite(manual_tracks[0]), axis=1)
            selected_tracks = np.asarray(manual_tracks, dtype=float)[
                :, initial_track_mask
            ]
            nees_by_frame: np.ndarray = np.full(len(frame_indices), np.nan, dtype=float)
            coverage_by_frame: np.ndarray = np.full(
                len(frame_indices), np.nan, dtype=float
            )
            count_by_frame: np.ndarray = np.zeros(len(frame_indices), dtype=np.int32)
            covariance_values = np.asarray(observation_covariance_m2, dtype=float)
            for output_frame, source_frame in enumerate(frame_indices):
                target = selected_tracks[source_frame]
                selected_covariance = covariance_values[output_frame, track_indices]
                selected_reliability = np.asarray(reliability[output_frame])[
                    track_indices
                ]
                usable = (
                    np.all(np.isfinite(target), axis=1)
                    & np.asarray(valid[output_frame])[track_indices]
                    & np.all(
                        np.isfinite(positions[output_frame, track_indices]), axis=1
                    )
                    & np.all(np.isfinite(selected_covariance), axis=(1, 2))
                    & (selected_reliability > 0.0)
                )
                if not np.any(usable):
                    continue
                residual = (
                    positions[output_frame, track_indices[usable]] - target[usable]
                )
                effective_covariance = selected_covariance[usable] / np.maximum(
                    selected_reliability[usable, None, None], 1e-6
                )
                inverse = np.linalg.pinv(effective_covariance, hermitian=True)
                squared_mahalanobis = np.einsum(
                    "ni,nij,nj->n", residual, inverse, residual
                )
                nees_by_frame[output_frame] = float(np.mean(squared_mahalanobis / 3.0))
                coverage_by_frame[output_frame] = float(
                    np.mean(squared_mahalanobis <= 6.251388631170325)
                )
                count_by_frame[output_frame] = int(np.sum(usable))
            summary["manual_identity_uncertainty_audit"] = {
                "available": True,
                "interpretation": (
                    "uncalibrated Gaussian inlier-component diagnostic; covariance "
                    "is not fitted on manual tracks"
                ),
                "normalized_nees_by_sampled_frame": nees_by_frame.tolist(),
                "ellipsoid_coverage_90_by_sampled_frame": coverage_by_frame.tolist(),
                "track_count_by_sampled_frame": count_by_frame.tolist(),
                "training_normalized_nees": _mean_on_frames(
                    nees_by_frame, train_selection
                ),
                "future_normalized_nees": _mean_on_frames(
                    nees_by_frame, future_selection
                ),
                "training_ellipsoid_coverage_90": _mean_on_frames(
                    coverage_by_frame, train_selection
                ),
                "future_ellipsoid_coverage_90": _mean_on_frames(
                    coverage_by_frame, future_selection
                ),
            }
        else:
            summary["manual_identity_uncertainty_audit"] = {
                "available": False,
                "reason": "the input path carries no metric observation covariance",
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
    frame_indices = (
        np.arange(0, frame_count, config.process_stride, dtype=np.int64)[
            : len(prediction.point_map)
        ]
        if prediction.frame_indices is None
        else np.asarray(prediction.frame_indices, dtype=np.int64)
    )
    if len(frame_indices) != len(prediction.point_map):
        raise ValueError("process_stride does not explain MotionCrafter frame count")
    if np.any(frame_indices < 0) or np.any(frame_indices >= frame_count):
        raise ValueError("MotionCrafter frame_indices exceed the released case")
    for camera, candidate in predictions.items():
        if candidate.point_map.shape != prediction.point_map.shape:
            raise ValueError(
                f"MotionCrafter camera {camera} shape does not match the primary view"
            )
        candidate_frames = (
            frame_indices
            if candidate.frame_indices is None
            else np.asarray(candidate.frame_indices, dtype=np.int64)
        )
        if not np.array_equal(candidate_frames, frame_indices):
            raise ValueError(
                f"MotionCrafter camera {camera} frame indices do not match the primary view"
            )

    alignment_reference_frame = int(frame_indices[0])
    camera_points, alignment_reference_source = load_phystwin_world_point_grid(
        raw_path,
        alignment_reference_frame,
    )
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
    object_visibilities = np.asarray(
        data.get(
            "object_visibilities",
            np.ones(observed.shape[:2], dtype=bool),
        ),
        dtype=bool,
    )
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
    manual_tracks = (
        None
        if not track_path.is_file()
        else np.asarray(_load_pickle(track_path), dtype=float)
    )
    covariance_node_indices = None
    if config.graph_covariance_manual_track_audit and manual_tracks is not None:
        initial_track_mask = np.all(np.isfinite(manual_tracks[0]), axis=1)
        if np.any(initial_track_mask):
            _, nearest = _nearest_neighbors(
                graph_initial,
                manual_tracks[0, initial_track_mask],
                k=1,
            )
            covariance_node_indices = np.unique(nearest[:, 0])
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
        fusion_mode=config.multiview_fusion_mode,
    )
    flow_observations = combine_framewise_graph_observations(
        flow_observations_by_view,
        consistency_scale_m=config.multiview_consistency_scale_m,
        minimum_reliability=config.minimum_multiview_reliability,
        fusion_mode=config.multiview_fusion_mode,
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
        covariance_probes=config.graph_covariance_probes,
        covariance_node_indices=covariance_node_indices,
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
        covariance_probes=config.graph_covariance_probes,
        covariance_node_indices=covariance_node_indices,
    )
    position_graph_covariance = (
        None
        if position_graph.marginal_variance_m2 is None
        else position_graph.marginal_variance_m2[..., None, None] * np.eye(3)
    )
    flow_graph_covariance = (
        None
        if flow_graph.marginal_variance_m2 is None
        else flow_graph.marginal_variance_m2[..., None, None] * np.eye(3)
    )
    train_selection = frame_indices < train_end
    future_selection = frame_indices >= train_end

    variants: dict[str, dict[str, object]] = {}
    for name, positions_array, valid_array, reliability_array, covariance_array in (
        (
            "released_phystwin",
            sampled_baseline,
            np.ones(sampled_baseline.shape[:2], dtype=bool),
            np.ones(sampled_baseline.shape[:2], dtype=float),
            None,
        ),
        (
            "position_only_direct",
            position_observations.positions,
            position_observations.valid,
            position_observations.reliability,
            position_observations.observation_covariance_m2,
        ),
        (
            "position_flow_direct",
            flow_observations.positions,
            flow_observations.valid,
            flow_observations.reliability,
            flow_observations.observation_covariance_m2,
        ),
        (
            "position_only_graph",
            position_graph.positions,
            position_graph.valid,
            np.where(position_graph.valid, 1.0, 0.0),
            position_graph_covariance,
        ),
        (
            "position_flow_graph",
            flow_graph.positions,
            flow_graph.valid,
            np.where(flow_graph.valid, 1.0, 0.0),
            flow_graph_covariance,
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
            observed,
            object_visibilities,
            covariance_array,
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
    archive_payload: dict[str, Any] = {
        "frame_indices": frame_indices.astype(np.int32),
        "position_only_positions": position_observations.positions,
        "position_only_valid": position_observations.valid,
        "position_only_reliability": position_observations.reliability,
        "position_only_measurement_mass": position_observations.measurement_mass,
        "position_flow_positions": flow_observations.positions,
        "position_flow_endpoints": flow_observations.flow_endpoints,
        "position_flow_valid": flow_observations.valid,
        "position_flow_endpoint_valid": flow_observations.flow_valid,
        "position_flow_reliability": flow_observations.reliability,
        "position_flow_endpoint_reliability": flow_observations.flow_reliability,
        "position_flow_measurement_mass": flow_observations.measurement_mass,
        "position_flow_entropy": flow_observations.normalized_entropy,
        "position_flow_position_error_m": flow_observations.position_error_m,
        "position_flow_endpoint_error_m": flow_observations.flow_endpoint_error_m,
        "position_only_graph_positions": position_graph.positions,
        "position_only_graph_valid": position_graph.valid,
        "position_only_graph_correction": position_graph.correction,
        "position_flow_graph_positions": flow_graph.positions,
        "position_flow_graph_valid": flow_graph.valid,
        "position_flow_graph_correction": flow_graph.correction,
    }
    if position_graph.marginal_variance_m2 is not None:
        archive_payload["position_only_graph_marginal_variance_m2"] = (
            position_graph.marginal_variance_m2
        )
    if flow_graph.marginal_variance_m2 is not None:
        archive_payload["position_flow_graph_marginal_variance_m2"] = (
            flow_graph.marginal_variance_m2
        )
    for prefix, observations in (
        ("position_only", position_observations),
        ("position_flow", flow_observations),
    ):
        optional_arrays = {
            "prior_reliability": observations.prior_reliability,
            "observation_covariance_m2": observations.observation_covariance_m2,
            "effective_sample_size": observations.effective_sample_size,
            "contributor_count": observations.contributor_count,
        }
        if prefix == "position_flow":
            optional_arrays.update(
                {
                    "endpoint_prior_reliability": (observations.flow_prior_reliability),
                    "endpoint_observation_covariance_m2": (
                        observations.flow_observation_covariance_m2
                    ),
                    "endpoint_effective_sample_size": (
                        observations.flow_effective_sample_size
                    ),
                }
            )
        for suffix, values in optional_arrays.items():
            if values is not None:
                archive_payload[f"{prefix}_{suffix}"] = np.asarray(values)
    np.savez_compressed(archive_path, **archive_payload)
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
        "schema_version": 3,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case_path.name,
        "config": asdict(config),
        "contract": {
            "persistent_identity": "PhysTwin graph vertex IDs",
            "motioncrafter_role": "anonymous per-frame 3D position and forward-flow likelihood",
            "association_prior": "one fixed released PhysTwin trajectory for all variants",
            "reassociation": "independent at every frame; no MotionCrafter identity transport",
            "graph_regularization": "Laplacian innovation prior; direct and imputed support are reported separately",
            "prior_reliability": "perception and assignment cues only in decoupled_robust mode; state innovation enters robust_mixture_likelihood once",
            "correlated_evidence": "spatial block ESS within a view and covariance intersection across views when enabled",
            "covariance_units": "metric m^2 propagated into graph smoothing when present",
            "future_use": "future MotionCrafter frames are reconstruction-only and never valid future-prediction inputs",
            "manual_tracks": "post-lock audit only; never used by alignment, assignment, posterior mean, or smoothing weights; frame-zero identities may select exact covariance diagonal entries when explicitly requested",
        },
        "software": {
            "motioncrafter_repository": MOTIONCRAFTER_REPOSITORY,
            "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        },
        "frame_indices": frame_indices.astype(int).tolist(),
        "train_end_frame": train_end,
        "alignment": {
            "reference_frame": alignment_reference_frame,
            "reference_source": alignment_reference_source,
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
            "position_flow_prior_reliability": (
                None
                if flow_observations.prior_reliability is None
                else _distribution(flow_observations.prior_reliability)
            ),
            "position_flow_posterior_inlier_probability": _distribution(
                flow_observations.reliability
            ),
            "position_flow_effective_sample_size": (
                None
                if flow_observations.effective_sample_size is None
                else _distribution(flow_observations.effective_sample_size)
            ),
            "position_flow_metric_variance_m2": (
                None
                if flow_observations.observation_covariance_m2 is None
                else _distribution(
                    np.max(
                        np.linalg.eigvalsh(
                            flow_observations.observation_covariance_m2[
                                flow_observations.valid
                            ]
                        ),
                        axis=-1,
                    )
                )
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
                    "point_covariance_available": (
                        aligned_predictions[camera].point_covariance_m2 is not None
                    ),
                    "flow_covariance_available": (
                        aligned_predictions[camera].flow_covariance_m2 is not None
                    ),
                    "contributors_available": (
                        aligned_predictions[camera].contributors is not None
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
                    "embedded_frame_indices": (
                        None
                        if predictions[camera].frame_indices is None
                        else predictions[camera].frame_indices.astype(int).tolist()
                    ),
                    "point_covariance_available": (
                        predictions[camera].point_covariance_m2 is not None
                    ),
                    "flow_covariance_available": (
                        predictions[camera].flow_covariance_m2 is not None
                    ),
                    "contributors_available": (
                        predictions[camera].contributors is not None
                    ),
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
