"""Covariance-aware sparse TAPNext++ updates for PhysTwin graph discrepancy.

The observation provider owns perception reliability and metric covariance.  A
physical prediction may determine the innovation, but never retroactively
changes that prior reliability.  Geometry-only candidate associations are
fixed at the query frame before innovations are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phystwin_graph_discrepancy import (
    GraphDiscrepancyPosterior,
    graph_smoothed_discrepancy_posterior,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _immutable(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _psd(value: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _clip_rows(values: np.ndarray, maximum_norm_m: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1)
    scale = np.minimum(1.0, maximum_norm_m / np.maximum(norms, 1e-15))
    result *= scale[..., None]
    return result


@dataclass(frozen=True)
class SparseAssimilationConfig:
    """Frozen numerical settings for a sparse graph-discrepancy update."""

    process_std_m: float = 0.001
    initial_std_m: float = 0.010
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0
    minimum_inlier_prior: float = 1e-3
    maximum_effective_rows_per_identity: float = 4.0
    association_neighbor_count: int = 4
    association_temperature_m: float = 0.005
    maximum_query_to_graph_distance_m: float = 0.030
    graph_prior_strength: float = 0.1
    maximum_sparse_delta_m: float = 0.010
    graph_covariance_probes: int = 32
    minimum_material_attachment_std_m: float = 0.001

    def __post_init__(self) -> None:
        for name, value in (
            ("process standard deviation", self.process_std_m),
            ("initial standard deviation", self.initial_std_m),
            ("association temperature", self.association_temperature_m),
            (
                "maximum query-to-graph distance",
                self.maximum_query_to_graph_distance_m,
            ),
            ("graph prior strength", self.graph_prior_strength),
            ("maximum sparse delta", self.maximum_sparse_delta_m),
            (
                "minimum material attachment standard deviation",
                self.minimum_material_attachment_std_m,
            ),
        ):
            _require(np.isfinite(value) and value > 0.0, f"{name} must be positive")
        _require(
            0.0 < self.minimum_inlier_prior < self.inlier_prior < 1.0,
            "inlier priors must satisfy 0 < minimum < base < 1",
        )
        _require(
            self.outlier_variance_multiplier > 1.0,
            "outlier variance multiplier must exceed one",
        )
        _require(
            self.maximum_effective_rows_per_identity >= 1.0,
            "effective-row cap must be at least one",
        )
        _require(
            self.association_neighbor_count >= 1,
            "association neighbor count must be positive",
        )
        _require(
            self.graph_covariance_probes >= 0,
            "graph covariance probes must be nonnegative",
        )


@dataclass(frozen=True)
class SparseAssociation:
    """Geometry-only graph association and its metric uncertainty."""

    candidate_indices: np.ndarray
    candidate_probabilities: np.ndarray
    map_indices: np.ndarray
    source_distance_m: np.ndarray
    entropy: np.ndarray
    predicted_points_m: np.ndarray
    innovation_m: np.ndarray
    covariance_m2: np.ndarray
    support: np.ndarray
    prior_reliability: np.ndarray

    def __post_init__(self) -> None:
        candidates = _immutable(self.candidate_indices, np.int64)
        probabilities = _immutable(self.candidate_probabilities, np.float64)
        maps = _immutable(self.map_indices, np.int64)
        distances = _immutable(self.source_distance_m, np.float64)
        entropy = _immutable(self.entropy, np.float64)
        predicted = _immutable(self.predicted_points_m, np.float64)
        innovation = _immutable(self.innovation_m, np.float64)
        covariance = _immutable(self.covariance_m2, np.float64)
        support = _immutable(self.support, bool)
        reliability = _immutable(self.prior_reliability, np.float64)
        _require(candidates.ndim == 2, "candidate indices must be a matrix")
        identity_count, neighbor_count = candidates.shape
        _require(
            probabilities.shape == (identity_count, neighbor_count),
            "candidate probability shape changed",
        )
        _require(maps.shape == (identity_count,), "MAP index shape changed")
        _require(distances.shape == (identity_count,), "distance shape changed")
        _require(entropy.shape == (identity_count,), "entropy shape changed")
        _require(
            predicted.ndim == 3 and predicted.shape[1:] == (identity_count, 3),
            "predicted points must have shape (T, N, 3)",
        )
        _require(innovation.shape == predicted.shape, "innovation shape changed")
        _require(support.shape == predicted.shape[:2], "support shape changed")
        _require(reliability.shape == support.shape, "reliability shape changed")
        _require(
            covariance.shape == (*support.shape, 3, 3),
            "covariance must have shape (T, N, 3, 3)",
        )
        _require(
            np.allclose(np.sum(probabilities, axis=1), 1.0),
            "association probabilities must sum to one",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
        for name, value in (
            ("candidate_indices", candidates),
            ("candidate_probabilities", probabilities),
            ("map_indices", maps),
            ("source_distance_m", distances),
            ("entropy", entropy),
            ("predicted_points_m", predicted),
            ("innovation_m", innovation),
            ("covariance_m2", covariance),
            ("support", support),
            ("prior_reliability", reliability),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MetricEndpointPosterior:
    """Per-identity robust endpoint posterior with full metric covariance."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    final_inlier_probability: np.ndarray
    update_count: np.ndarray
    effective_row_count: np.ndarray
    temporal_covariance_inflation: np.ndarray

    def __post_init__(self) -> None:
        mean = _immutable(self.mean_m, np.float64)
        covariance = _immutable(self.covariance_m2, np.float64)
        probability = _immutable(self.final_inlier_probability, np.float64)
        update_count = _immutable(self.update_count, np.int64)
        effective = _immutable(self.effective_row_count, np.float64)
        inflation = _immutable(self.temporal_covariance_inflation, np.float64)
        _require(mean.ndim == 2 and mean.shape[1] == 3, "mean must be (N, 3)")
        identity_count = len(mean)
        _require(
            covariance.shape == (identity_count, 3, 3),
            "endpoint covariance must be (N, 3, 3)",
        )
        for value in (probability, update_count, effective, inflation):
            _require(value.shape == (identity_count,), "endpoint vector shape changed")
        for name, value in (
            ("mean_m", mean),
            ("covariance_m2", covariance),
            ("final_inlier_probability", probability),
            ("update_count", update_count),
            ("effective_row_count", effective),
            ("temporal_covariance_inflation", inflation),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SparseGraphUpdate:
    """Direct and graph-smoothed baseline-relative sparse corrections."""

    accepted: bool
    reason: str
    direct_delta_m: np.ndarray
    graph_delta_m: np.ndarray
    graph_marginal_variance_m2: np.ndarray
    observed_nodes: np.ndarray
    observed_delta_m: np.ndarray
    observed_variance_m2: np.ndarray
    graph_posterior: GraphDiscrepancyPosterior | None

    def __post_init__(self) -> None:
        direct = _immutable(self.direct_delta_m, np.float64)
        graph = _immutable(self.graph_delta_m, np.float64)
        marginal = _immutable(self.graph_marginal_variance_m2, np.float64)
        nodes = _immutable(self.observed_nodes, np.int64)
        delta = _immutable(self.observed_delta_m, np.float64)
        variance = _immutable(self.observed_variance_m2, np.float64)
        _require(direct.ndim == 2 and direct.shape[1] == 3, "direct delta changed")
        _require(graph.shape == direct.shape, "graph delta shape changed")
        _require(marginal.shape == (len(direct),), "marginal variance shape changed")
        _require(delta.shape == (len(nodes), 3), "observed delta shape changed")
        _require(variance.shape == (len(nodes),), "observed variance shape changed")
        for name, value in (
            ("direct_delta_m", direct),
            ("graph_delta_m", graph),
            ("graph_marginal_variance_m2", marginal),
            ("observed_nodes", nodes),
            ("observed_delta_m", delta),
            ("observed_variance_m2", variance),
        ):
            object.__setattr__(self, name, value)


def associate_sparse_observations(
    observed_points_m: np.ndarray,
    support: np.ndarray,
    prior_reliability: np.ndarray,
    observation_covariance_m2: np.ndarray,
    baseline_segment_m: np.ndarray,
    *,
    config: SparseAssimilationConfig | None = None,
) -> SparseAssociation:
    """Associate queries from source geometry, then form one state innovation."""

    cfg = config or SparseAssimilationConfig()
    observations = np.asarray(observed_points_m, dtype=np.float64)
    validity = np.asarray(support, dtype=bool)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    covariance = np.asarray(observation_covariance_m2, dtype=np.float64)
    baseline = np.asarray(baseline_segment_m, dtype=np.float64)
    _require(
        observations.ndim == 3 and observations.shape[-1] == 3,
        "observations must have shape (T, N, 3)",
    )
    frame_count, identity_count, _ = observations.shape
    _require(validity.shape == (frame_count, identity_count), "support shape changed")
    _require(reliability.shape == validity.shape, "reliability shape changed")
    _require(
        covariance.shape == (frame_count, identity_count, 3, 3),
        "observation covariance shape changed",
    )
    _require(
        baseline.ndim == 3
        and baseline.shape[0] == frame_count
        and baseline.shape[2] == 3,
        "baseline segment must have shape (T, M, 3)",
    )
    _require(
        np.all(validity[0]),
        "every sparse identity must have a supported query-frame carrier",
    )
    _require(
        np.all(np.isfinite(observations[validity])),
        "supported observations are not finite",
    )
    _require(
        np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "prior reliability must lie in [0, 1]",
    )
    _require(
        np.all(reliability[~validity] == 0.0), "unsupported reliability must be zero"
    )
    node_count = baseline.shape[1]
    neighbor_count = min(cfg.association_neighbor_count, node_count)
    source_delta = observations[0, :, None, :] - baseline[0, None, :, :]
    squared_distance = np.sum(np.square(source_delta), axis=-1)
    candidates = np.argsort(squared_distance, axis=1)[:, :neighbor_count]
    candidate_squared = np.take_along_axis(squared_distance, candidates, axis=1)
    source_distance = np.sqrt(candidate_squared[:, 0])
    _require(
        np.all(source_distance <= cfg.maximum_query_to_graph_distance_m),
        "a query is too far from the physical graph",
    )
    centered_score = -(candidate_squared - candidate_squared[:, :1]) / (
        2.0 * cfg.association_temperature_m**2
    )
    centered_score -= np.max(centered_score, axis=1, keepdims=True)
    probabilities = np.exp(centered_score)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    maps = candidates[np.arange(identity_count), np.argmax(probabilities, axis=1)]
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-15)),
        axis=1,
    )

    candidate_trajectory = baseline[:, candidates, :]
    predicted = np.sum(
        candidate_trajectory * probabilities[None, :, :, None],
        axis=2,
    )
    centered = candidate_trajectory - predicted[:, :, None, :]
    assignment_covariance = np.einsum(
        "nk,tnki,tnkj->tnij",
        probabilities,
        centered,
        centered,
        optimize=True,
    )
    total_covariance = covariance + assignment_covariance
    innovation = np.zeros_like(observations)
    innovation[validity] = observations[validity] - predicted[validity]
    return SparseAssociation(
        candidate_indices=candidates,
        candidate_probabilities=probabilities,
        map_indices=maps,
        source_distance_m=source_distance,
        entropy=entropy,
        predicted_points_m=predicted,
        innovation_m=innovation,
        covariance_m2=total_covariance,
        support=validity,
        prior_reliability=reliability,
    )


def associate_fixed_material_displacements(
    observed_points_m: np.ndarray,
    support: np.ndarray,
    prior_reliability: np.ndarray,
    observation_covariance_m2: np.ndarray,
    baseline_segment_m: np.ndarray,
    material_node_indices: np.ndarray,
    *,
    config: SparseAssimilationConfig | None = None,
) -> SparseAssociation:
    """Form relative innovations at immutable frame-zero material nodes.

    The query-frame observation is an anchor, not an update. Relative motion
    cancels a constant point-to-node attachment offset, while its magnitude is
    retained as a conservative covariance term for non-rigid local motion.
    """

    cfg = config or SparseAssimilationConfig()
    observations = np.asarray(observed_points_m, dtype=np.float64)
    validity = np.asarray(support, dtype=bool)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    covariance = np.asarray(observation_covariance_m2, dtype=np.float64)
    baseline = np.asarray(baseline_segment_m, dtype=np.float64)
    nodes = np.asarray(material_node_indices, dtype=np.int64)
    _require(
        observations.ndim == 3 and observations.shape[-1] == 3,
        "observations must have shape (T, N, 3)",
    )
    frame_count, identity_count, _ = observations.shape
    _require(validity.shape == (frame_count, identity_count), "support shape changed")
    _require(reliability.shape == validity.shape, "reliability shape changed")
    _require(
        covariance.shape == (frame_count, identity_count, 3, 3),
        "observation covariance shape changed",
    )
    _require(
        baseline.ndim == 3
        and baseline.shape[0] == frame_count
        and baseline.shape[2] == 3,
        "baseline segment must have shape (T, M, 3)",
    )
    _require(nodes.shape == (identity_count,), "material-node shape changed")
    _require(
        np.all((nodes >= 0) & (nodes < baseline.shape[1])),
        "material node lies outside the graph",
    )
    _require(
        np.all(validity[0]),
        "every material identity must have a supported query-frame carrier",
    )
    _require(
        np.all(np.isfinite(observations[validity])),
        "supported observations are not finite",
    )
    _require(
        np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "prior reliability must lie in [0, 1]",
    )
    _require(
        np.all(reliability[~validity] == 0.0), "unsupported reliability must be zero"
    )

    predicted = baseline[:, nodes]
    source_offset = observations[0] - predicted[0]
    source_distance = np.linalg.norm(source_offset, axis=1)
    _require(
        np.all(source_distance <= cfg.maximum_query_to_graph_distance_m),
        "a material query is too far from its fixed physical node",
    )
    observed_displacement = observations - observations[:1]
    predicted_displacement = predicted - predicted[:1]
    relative_support = validity & validity[:1]
    relative_support[0] = False
    relative_reliability = np.minimum(reliability, reliability[:1])
    relative_reliability[~relative_support] = 0.0
    innovation = np.zeros_like(observations)
    innovation[relative_support] = (
        observed_displacement[relative_support]
        - predicted_displacement[relative_support]
    )

    attachment_std = np.maximum(
        source_distance,
        cfg.minimum_material_attachment_std_m,
    )
    attachment_covariance = (
        np.eye(3)[None, None] * np.square(attachment_std)[None, :, None, None]
    )
    total_covariance = covariance + covariance[:1] + attachment_covariance
    candidates = nodes[:, None]
    probabilities = np.ones((identity_count, 1), dtype=np.float64)
    return SparseAssociation(
        candidate_indices=candidates,
        candidate_probabilities=probabilities,
        map_indices=nodes,
        source_distance_m=source_distance,
        entropy=np.zeros(identity_count, dtype=np.float64),
        predicted_points_m=predicted,
        innovation_m=innovation,
        covariance_m2=total_covariance,
        support=relative_support,
        prior_reliability=relative_reliability,
    )


def _log_normal_zero(innovation: np.ndarray, covariance: np.ndarray) -> float:
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("innovation covariance is not positive definite")
    squared = float(innovation @ np.linalg.solve(covariance, innovation))
    return -0.5 * (3.0 * np.log(2.0 * np.pi) + log_determinant + squared)


def robust_metric_random_walk_endpoint(
    innovation_m: np.ndarray,
    support: np.ndarray,
    prior_reliability: np.ndarray,
    observation_covariance_m2: np.ndarray,
    *,
    config: SparseAssimilationConfig | None = None,
) -> MetricEndpointPosterior:
    """Process each innovation once with a reliability-conditioned mixture."""

    cfg = config or SparseAssimilationConfig()
    innovation = np.asarray(innovation_m, dtype=np.float64)
    validity = np.asarray(support, dtype=bool)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    observation_covariance = np.asarray(observation_covariance_m2, dtype=np.float64)
    _require(
        innovation.ndim == 3 and innovation.shape[-1] == 3,
        "innovation must have shape (T, N, 3)",
    )
    frame_count, identity_count, _ = innovation.shape
    _require(validity.shape == (frame_count, identity_count), "support shape changed")
    _require(reliability.shape == validity.shape, "reliability shape changed")
    _require(
        observation_covariance.shape == (frame_count, identity_count, 3, 3),
        "observation covariance shape changed",
    )
    _require(np.all(np.isfinite(innovation[validity])), "innovation is not finite")
    _require(
        np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "prior reliability must lie in [0, 1]",
    )
    counts = np.sum(validity, axis=0).astype(np.float64)
    inflation = np.maximum(
        1.0,
        counts / cfg.maximum_effective_rows_per_identity,
    )
    effective = counts / inflation
    mean = np.zeros((identity_count, 3), dtype=np.float64)
    covariance = np.broadcast_to(
        np.eye(3) * cfg.initial_std_m**2,
        (identity_count, 3, 3),
    ).copy()
    final_probability = np.zeros(identity_count, dtype=np.float64)
    update_count = np.zeros(identity_count, dtype=np.int64)
    process_covariance = np.eye(3) * cfg.process_std_m**2
    for frame in range(frame_count):
        covariance += process_covariance[None]
        for identity in np.flatnonzero(validity[frame]):
            prior_probability = float(
                np.clip(
                    cfg.inlier_prior * reliability[frame, identity],
                    cfg.minimum_inlier_prior,
                    1.0 - cfg.minimum_inlier_prior,
                )
            )
            measurement_covariance = _psd(
                observation_covariance[frame, identity] * inflation[identity],
                floor=1e-12,
            )
            predicted_covariance = covariance[identity]
            current_innovation = innovation[frame, identity] - mean[identity]
            inlier_innovation_covariance = _psd(
                predicted_covariance + measurement_covariance,
                floor=1e-12,
            )
            outlier_measurement_covariance = (
                cfg.outlier_variance_multiplier * measurement_covariance
            )
            outlier_innovation_covariance = _psd(
                predicted_covariance + outlier_measurement_covariance,
                floor=1e-12,
            )
            log_inlier = np.log(prior_probability) + _log_normal_zero(
                current_innovation,
                inlier_innovation_covariance,
            )
            log_outlier = np.log1p(-prior_probability) + _log_normal_zero(
                current_innovation,
                outlier_innovation_covariance,
            )
            probability = float(
                np.exp(log_inlier - np.logaddexp(log_inlier, log_outlier))
            )
            inlier_gain = predicted_covariance @ np.linalg.inv(
                inlier_innovation_covariance
            )
            outlier_gain = predicted_covariance @ np.linalg.inv(
                outlier_innovation_covariance
            )
            inlier_mean = mean[identity] + inlier_gain @ current_innovation
            outlier_mean = mean[identity] + outlier_gain @ current_innovation
            inlier_covariance = _psd(
                predicted_covariance - inlier_gain @ predicted_covariance,
                floor=1e-12,
            )
            outlier_covariance = _psd(
                predicted_covariance - outlier_gain @ predicted_covariance,
                floor=1e-12,
            )
            mixture_mean = (
                probability * inlier_mean + (1.0 - probability) * outlier_mean
            )
            inlier_offset = inlier_mean - mixture_mean
            outlier_offset = outlier_mean - mixture_mean
            mixture_covariance = probability * (
                inlier_covariance + np.outer(inlier_offset, inlier_offset)
            ) + (1.0 - probability) * (
                outlier_covariance + np.outer(outlier_offset, outlier_offset)
            )
            mean[identity] = mixture_mean
            covariance[identity] = _psd(mixture_covariance, floor=1e-12)
            final_probability[identity] = probability
            update_count[identity] += 1
    return MetricEndpointPosterior(
        mean_m=mean,
        covariance_m2=covariance,
        final_inlier_probability=final_probability,
        update_count=update_count,
        effective_row_count=effective,
        temporal_covariance_inflation=inflation,
    )


def build_sparse_graph_update(
    endpoint: MetricEndpointPosterior,
    association: SparseAssociation,
    dense_correction_m: np.ndarray,
    laplacian,
    *,
    config: SparseAssimilationConfig | None = None,
) -> SparseGraphUpdate:
    """Build a baseline-relative direct field and graph-smoothed alternative."""

    cfg = config or SparseAssimilationConfig()
    dense = np.asarray(dense_correction_m, dtype=np.float64)
    _require(dense.ndim == 2 and dense.shape[1] == 3, "dense correction must be (M, 3)")
    _require(laplacian.shape == (len(dense), len(dense)), "Laplacian shape changed")
    updated = endpoint.update_count > 0
    inlier = endpoint.final_inlier_probability >= 0.5
    accepted_identity = updated & inlier
    empty_direct = np.zeros_like(dense)
    empty_variance = np.zeros(len(dense), dtype=np.float64)
    if not np.any(accepted_identity):
        return SparseGraphUpdate(
            accepted=False,
            reason="no-robust-sparse-identity-update",
            direct_delta_m=empty_direct,
            graph_delta_m=empty_direct,
            graph_marginal_variance_m2=empty_variance,
            observed_nodes=np.empty(0, dtype=np.int64),
            observed_delta_m=np.empty((0, 3), dtype=np.float64),
            observed_variance_m2=np.empty(0, dtype=np.float64),
            graph_posterior=None,
        )
    dense_at_query = np.sum(
        dense[association.candidate_indices]
        * association.candidate_probabilities[..., None],
        axis=1,
    )
    sparse_delta = endpoint.mean_m - dense_at_query
    return _build_graph_update_from_delta(
        endpoint,
        association,
        sparse_delta,
        laplacian,
        node_count=len(dense),
        config=cfg,
    )


def build_material_transport_graph_update(
    endpoint: MetricEndpointPosterior,
    association: SparseAssociation,
    laplacian,
    *,
    config: SparseAssimilationConfig | None = None,
) -> SparseGraphUpdate:
    """Propagate a fixed-node relative-displacement posterior over the graph."""

    cfg = config or SparseAssimilationConfig()
    node_count = int(laplacian.shape[0])
    _require(laplacian.shape == (node_count, node_count), "Laplacian shape changed")
    return _build_graph_update_from_delta(
        endpoint,
        association,
        endpoint.mean_m,
        laplacian,
        node_count=node_count,
        config=cfg,
    )


def _build_graph_update_from_delta(
    endpoint: MetricEndpointPosterior,
    association: SparseAssociation,
    sparse_delta_m: np.ndarray,
    laplacian,
    *,
    node_count: int,
    config: SparseAssimilationConfig,
) -> SparseGraphUpdate:
    """Build direct and graph fields from one delta per sparse identity."""

    cfg = config
    sparse_delta = np.asarray(sparse_delta_m, dtype=np.float64)
    _require(
        sparse_delta.shape == endpoint.mean_m.shape,
        "sparse delta shape changed",
    )
    _require(laplacian.shape == (node_count, node_count), "Laplacian shape changed")
    updated = endpoint.update_count > 0
    inlier = endpoint.final_inlier_probability >= 0.5
    accepted_identity = updated & inlier
    empty_direct = np.zeros((node_count, 3), dtype=np.float64)
    empty_variance = np.zeros(node_count, dtype=np.float64)
    if not np.any(accepted_identity):
        return SparseGraphUpdate(
            accepted=False,
            reason="no-robust-sparse-identity-update",
            direct_delta_m=empty_direct,
            graph_delta_m=empty_direct,
            graph_marginal_variance_m2=empty_variance,
            observed_nodes=np.empty(0, dtype=np.int64),
            observed_delta_m=np.empty((0, 3), dtype=np.float64),
            observed_variance_m2=np.empty(0, dtype=np.float64),
            graph_posterior=None,
        )
    scalar_variance = np.linalg.eigvalsh(endpoint.covariance_m2)[:, -1]

    # Correlated identities that map to one node do not accumulate precision.
    # Retain the smallest-variance representative and record one observation.
    representatives: list[int] = []
    for node in np.unique(association.map_indices[accepted_identity]):
        rows = np.flatnonzero(accepted_identity & (association.map_indices == node))
        representatives.append(
            int(min(rows, key=lambda row: (scalar_variance[row], row)))
        )
    representative = np.asarray(representatives, dtype=np.int64)
    nodes = association.map_indices[representative]
    deltas = sparse_delta[representative]
    variances = np.maximum(scalar_variance[representative], 1e-12)
    direct = np.zeros((node_count, 3), dtype=np.float64)
    direct[nodes] = deltas
    direct = _clip_rows(direct, cfg.maximum_sparse_delta_m)

    observed_mean = np.zeros((node_count, 3), dtype=np.float64)
    observed_variance = np.ones(node_count, dtype=np.float64)
    observed_mask = np.zeros(node_count, dtype=bool)
    observed_mean[nodes] = deltas
    observed_variance[nodes] = variances
    observed_mask[nodes] = True
    posterior = graph_smoothed_discrepancy_posterior(
        observed_mean,
        observed_variance,
        observed_mask,
        laplacian,
        prior_strength=cfg.graph_prior_strength,
        covariance_probes=cfg.graph_covariance_probes,
    )
    graph = _clip_rows(posterior.mean, cfg.maximum_sparse_delta_m)
    marginal = (
        np.zeros(node_count, dtype=np.float64)
        if posterior.marginal_variance is None
        else posterior.marginal_variance
    )
    return SparseGraphUpdate(
        accepted=True,
        reason="robust-sparse-update-accepted",
        direct_delta_m=direct,
        graph_delta_m=graph,
        graph_marginal_variance_m2=marginal,
        observed_nodes=nodes,
        observed_delta_m=deltas,
        observed_variance_m2=variances,
        graph_posterior=posterior,
    )
