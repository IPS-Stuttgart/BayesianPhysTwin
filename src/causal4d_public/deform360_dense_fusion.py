"""Conservative correlated-view velocity fusion for dense Deform360 tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesian_phystwin.phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)


@dataclass(frozen=True)
class DenseVelocityFusionConfig:
    """Locked source-only settings for a conservative velocity posterior."""

    base_standard_deviation_mps: float = 0.20
    consistency_scale_mps: float = 0.25
    student_t_degrees_of_freedom: float = 4.0
    minimum_prior_reliability: float = 0.05
    maximum_effective_views: float = 2.0
    graph_neighbors: int = 12
    graph_radius_m: float = 0.04
    graph_prior_strength: float = 1.0
    graph_ridge: float = 1e-6
    spatial_innovation_scale_mps: float = 0.30

    def __post_init__(self) -> None:
        positive = (
            self.base_standard_deviation_mps,
            self.consistency_scale_mps,
            self.student_t_degrees_of_freedom,
            self.minimum_prior_reliability,
            self.maximum_effective_views,
            self.graph_radius_m,
            self.graph_prior_strength,
            self.graph_ridge,
            self.spatial_innovation_scale_mps,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("dense velocity fusion settings must be positive")
        if self.minimum_prior_reliability > 1.0:
            raise ValueError("minimum prior reliability cannot exceed one")
        if self.graph_neighbors < 1:
            raise ValueError("graph_neighbors must be positive")


@dataclass(frozen=True)
class CorrelatedVelocityObservations:
    """Per-point robust moments before spatial graph completion."""

    mean_mps: np.ndarray
    covariance_m2ps2: np.ndarray
    valid: np.ndarray
    contributor_count: np.ndarray
    effective_sample_size: np.ndarray
    prior_reliability: np.ndarray
    posterior_reliability: np.ndarray
    consistency_weight: np.ndarray


@dataclass(frozen=True)
class DenseVelocityPosterior:
    """Graph-completed velocity mean with auditable direct evidence."""

    mean_mps: np.ndarray
    observation_variance_m2ps2: np.ndarray
    directly_observed: np.ndarray
    contributor_count: np.ndarray
    effective_sample_size: np.ndarray
    prior_reliability: np.ndarray
    posterior_reliability: np.ndarray
    consistency_weight: np.ndarray
    spatial_robust_weight: np.ndarray
    springs: np.ndarray
    solve_iterations: tuple[int, ...]
    solve_relative_residuals: tuple[float, ...]


def _student_t_weight(
    residual_norm: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    squared = np.square(np.asarray(residual_norm, dtype=float) / scale)
    dimension = 3.0
    return (degrees_of_freedom + dimension) / (degrees_of_freedom + squared)


def _weighted_coordinate_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    center = np.zeros((len(values), 3), dtype=float)
    total = np.sum(weights, axis=1)
    observed = total > 0.0
    for coordinate in range(3):
        order = np.argsort(values[:, :, coordinate], axis=1)
        ordered_values = np.take_along_axis(values[:, :, coordinate], order, axis=1)
        ordered_weights = np.take_along_axis(weights, order, axis=1)
        cumulative = np.cumsum(ordered_weights, axis=1)
        median_index = np.argmax(cumulative >= 0.5 * total[:, None], axis=1)
        center[observed, coordinate] = ordered_values[
            np.flatnonzero(observed), median_index[observed]
        ]
    return center


def fuse_correlated_velocity_observations(
    per_view_velocity_mps: np.ndarray,
    valid: np.ndarray,
    prior_reliability: np.ndarray,
    config: DenseVelocityFusionConfig,
) -> CorrelatedVelocityObservations:
    """Fuse views without treating their unknown correlation as independence.

    Prior reliability is supplied by perception-only cues. Residuals are used
    once, inside the robust observation likelihood, and never fed back into the
    prior reliability. The fused covariance includes source variance and
    between-view spread but is not divided by the number of cameras.
    """

    values = np.asarray(per_view_velocity_mps, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    prior = np.asarray(prior_reliability, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("per-view velocities must have shape (N, V, 3)")
    if mask.shape != values.shape[:2] or prior.shape != mask.shape:
        raise ValueError("validity and reliability must match velocity views")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError("valid velocity observations must be finite")
    if not np.all(np.isfinite(prior)) or np.any((prior < 0.0) | (prior > 1.0)):
        raise ValueError("prior reliability must lie in [0, 1]")

    usable = mask & (prior >= config.minimum_prior_reliability)
    base_weight = np.where(usable, prior, 0.0)
    denominator = np.sum(base_weight, axis=1)
    observed = denominator > 0.0
    center = _weighted_coordinate_median(values, base_weight)

    residual = np.linalg.norm(values - center[:, None, :], axis=2)
    consistency = np.where(
        usable,
        np.minimum(
            1.0,
            _student_t_weight(
                residual,
                config.consistency_scale_mps,
                config.student_t_degrees_of_freedom,
            ),
        ),
        0.0,
    )
    robust_weight = base_weight * consistency
    robust_denominator = np.sum(robust_weight, axis=1)
    observed = robust_denominator > 0.0
    center[observed] = (
        np.sum(
            robust_weight[..., None] * np.where(usable[..., None], values, 0.0),
            axis=1,
        )[observed]
        / robust_denominator[observed, None]
    )

    centered = values - center[:, None, :]
    source_variance = np.square(config.base_standard_deviation_mps) / np.maximum(
        prior, config.minimum_prior_reliability
    )
    covariance = np.zeros((len(values), 3, 3), dtype=float)
    covariance[observed] = (
        np.sum(
            robust_weight[..., None, None]
            * (
                source_variance[..., None, None] * np.eye(3)[None, None]
                + np.einsum("nvi,nvj->nvij", centered, centered)
            ),
            axis=1,
        )[observed]
        / robust_denominator[observed, None, None]
    )
    covariance += np.square(config.base_standard_deviation_mps) * 1e-6 * np.eye(3)

    squared_weight = np.sum(np.square(robust_weight), axis=1)
    effective_sample_size = np.divide(
        np.square(robust_denominator),
        squared_weight,
        out=np.zeros(len(values), dtype=float),
        where=squared_weight > 0.0,
    )
    effective_sample_size = np.minimum(
        effective_sample_size, config.maximum_effective_views
    )
    contributor_count = np.sum(usable, axis=1).astype(np.int32)
    fused_prior = np.max(np.where(usable, prior, 0.0), axis=1)
    fused_posterior = np.max(robust_weight, axis=1)
    return CorrelatedVelocityObservations(
        mean_mps=center.astype(np.float32),
        covariance_m2ps2=covariance.astype(np.float32),
        valid=observed,
        contributor_count=contributor_count,
        effective_sample_size=effective_sample_size.astype(np.float32),
        prior_reliability=fused_prior.astype(np.float32),
        posterior_reliability=fused_posterior.astype(np.float32),
        consistency_weight=consistency.astype(np.float32),
    )


def knn_springs(
    points_m: np.ndarray,
    *,
    neighbors: int,
    radius_m: float,
) -> np.ndarray:
    """Build an undirected local graph and connect any isolated vertices."""

    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError) as error:
        raise RuntimeError("dense velocity fusion requires scipy") from error
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("graph points must be finite")
    tree = cKDTree(points)
    distances, indices = tree.query(
        points,
        k=min(neighbors + 1, len(points)),
        distance_upper_bound=radius_m,
    )
    edges: set[tuple[int, int]] = set()
    for source, (row_distance, row_index) in enumerate(
        zip(np.atleast_2d(distances), np.atleast_2d(indices), strict=True)
    ):
        for distance, target in zip(row_distance[1:], row_index[1:], strict=True):
            if not np.isfinite(distance) or target >= len(points) or source == target:
                continue
            edges.add(tuple(sorted((source, int(target)))))
    degree = np.zeros(len(points), dtype=np.int32)
    for source, target in edges:
        degree[source] += 1
        degree[target] += 1
    for source in np.flatnonzero(degree == 0):
        _, nearest = tree.query(points[source], k=2)
        target = int(np.asarray(nearest)[1])
        edges.add(tuple(sorted((int(source), target))))
    return np.asarray(sorted(edges), dtype=np.int32)


def graph_complete_velocity(
    points_m: np.ndarray,
    observations: CorrelatedVelocityObservations,
    config: DenseVelocityFusionConfig,
    *,
    springs: np.ndarray | None = None,
) -> DenseVelocityPosterior:
    """Complete sparse velocities with metric-variance graph smoothing."""

    points = np.asarray(points_m, dtype=float)
    edges = (
        knn_springs(
            points,
            neighbors=config.graph_neighbors,
            radius_m=config.graph_radius_m,
        )
        if springs is None
        else np.asarray(springs, dtype=np.int32)
    )
    laplacian = normalized_spring_laplacian(len(points), edges)
    variance = np.trace(observations.covariance_m2ps2, axis1=1, axis2=2) / 3.0
    variance = np.maximum(
        variance, np.square(config.base_standard_deviation_mps) * 1e-6
    )
    preliminary = graph_smoothed_discrepancy_posterior(
        observations.mean_mps,
        variance,
        observations.valid,
        laplacian,
        prior_strength=config.graph_prior_strength,
        ridge=config.graph_ridge,
    )
    spatial_residual = np.linalg.norm(
        observations.mean_mps - preliminary.mean[: len(points)], axis=1
    )
    spatial_weight = np.ones(len(points), dtype=float)
    spatial_weight[observations.valid] = np.minimum(
        1.0,
        _student_t_weight(
            spatial_residual[observations.valid],
            config.spatial_innovation_scale_mps,
            config.student_t_degrees_of_freedom,
        ),
    )
    robust_variance = variance / np.maximum(spatial_weight, 1e-6)
    posterior = graph_smoothed_discrepancy_posterior(
        observations.mean_mps,
        robust_variance,
        observations.valid,
        laplacian,
        prior_strength=config.graph_prior_strength,
        ridge=config.graph_ridge,
    )
    return DenseVelocityPosterior(
        mean_mps=posterior.mean[: len(points)].astype(np.float32),
        observation_variance_m2ps2=robust_variance.astype(np.float32),
        directly_observed=observations.valid.copy(),
        contributor_count=observations.contributor_count.copy(),
        effective_sample_size=observations.effective_sample_size.copy(),
        prior_reliability=observations.prior_reliability.copy(),
        posterior_reliability=observations.posterior_reliability.copy(),
        consistency_weight=observations.consistency_weight.copy(),
        spatial_robust_weight=spatial_weight.astype(np.float32),
        springs=edges,
        solve_iterations=posterior.solve_iterations,
        solve_relative_residuals=posterior.solve_relative_residuals,
    )


__all__ = [
    "CorrelatedVelocityObservations",
    "DenseVelocityFusionConfig",
    "DenseVelocityPosterior",
    "fuse_correlated_velocity_observations",
    "graph_complete_velocity",
    "knn_springs",
]
