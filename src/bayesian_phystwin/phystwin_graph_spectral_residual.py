"""Low-capacity graph-spectral continuation of PhysTwin discrepancy.

The model is deliberately diagonal in graph frequency and isotropic in xyz.
This makes graph-eigenvector signs and global coordinate rotations nuisance
choices rather than learned shortcuts. Endpoint persistence is represented by
an exact all-zero transition and remains the mandatory fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .phystwin_residual_dynamics import _temporally_fill


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GraphSpectralResidualConfig:
    """Geometry, filtering, and stability settings."""

    rank: int = 16
    neighbor_count: int = 8
    mode_group_count: int = 4
    temporal_smoothing: float = 0.5
    controller_kernel_fraction: float = 0.25
    ridge_fraction: float = 0.01
    minimum_group_samples: int = 24
    minimum_object_scale_m: float = 1.0e-6
    maximum_residual_m: float = 0.01
    minimum_edge_weight: float = 1.0e-8
    minimum_bandwidth_m: float = 1.0e-6
    minimum_velocity_coefficient: float = -0.25
    maximum_velocity_coefficient: float = 0.98
    maximum_action_coefficient: float = 2.0

    def __post_init__(self) -> None:
        _require(self.rank >= 2, "rank must be at least two")
        _require(self.neighbor_count >= 1, "neighbor count must be positive")
        _require(
            2 <= self.mode_group_count <= self.rank,
            "mode group count must lie in [2, rank]",
        )
        _require(
            0.0 < self.temporal_smoothing <= 1.0,
            "temporal smoothing must lie in (0, 1]",
        )
        _require(
            self.controller_kernel_fraction > 0.0,
            "controller kernel fraction must be positive",
        )
        positive = (
            self.ridge_fraction,
            self.minimum_object_scale_m,
            self.maximum_residual_m,
            self.minimum_edge_weight,
            self.minimum_bandwidth_m,
            self.maximum_action_coefficient,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "all numerical scales must be finite and positive",
        )
        _require(
            self.minimum_group_samples >= 3,
            "each spectral group needs at least three samples",
        )
        _require(
            -1.0 < self.minimum_velocity_coefficient
            <= self.maximum_velocity_coefficient
            < 1.0,
            "velocity coefficient bounds must be stable and ordered",
        )


@dataclass(frozen=True)
class GraphSpectralSeries:
    """One episode represented in its material graph basis."""

    basis: np.ndarray
    eigenvalues: np.ndarray
    mode_groups: np.ndarray
    residual_coefficients: np.ndarray
    action_coefficients: np.ndarray
    object_scale_m: float

    def __post_init__(self) -> None:
        basis = _readonly(self.basis, dtype=np.float64)
        eigenvalues = _readonly(self.eigenvalues, dtype=np.float64)
        groups = _readonly(self.mode_groups, dtype=np.int16)
        residual = _readonly(self.residual_coefficients, dtype=np.float64)
        action = _readonly(self.action_coefficients, dtype=np.float64)
        _require(
            basis.ndim == 2 and basis.shape[1] >= 2,
            "basis must have shape (point, mode)",
        )
        rank = basis.shape[1]
        _require(eigenvalues.shape == (rank,), "eigenvalue shape changed")
        _require(groups.shape == (rank,), "mode-group shape changed")
        _require(
            residual.ndim == 3
            and residual.shape[1:] == (rank, 3)
            and action.ndim == 3
            and action.shape[1:] == (rank, 3)
            and len(action) >= len(residual),
            "action series must cover the observed residual series",
        )
        _require(
            np.all(np.isfinite(basis))
            and np.all(np.isfinite(eigenvalues))
            and np.all(np.isfinite(residual))
            and np.all(np.isfinite(action)),
            "spectral series contains non-finite values",
        )
        _require(
            np.allclose(
                basis.T @ basis,
                np.eye(rank),
                atol=1.0e-7,
                rtol=0.0,
            ),
            "graph basis must be orthonormal",
        )
        _require(
            np.isfinite(self.object_scale_m) and self.object_scale_m > 0.0,
            "object scale must be finite and positive",
        )
        for name, values in {
            "basis": basis,
            "eigenvalues": eigenvalues,
            "mode_groups": groups,
            "residual_coefficients": residual,
            "action_coefficients": action,
        }.items():
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class GraphSpectralTransition:
    """Scalar transition coefficients shared within graph-frequency groups."""

    velocity_retention: np.ndarray
    action_current: np.ndarray
    action_change: np.ndarray
    sample_count: np.ndarray

    def __post_init__(self) -> None:
        retention = _readonly(self.velocity_retention, dtype=np.float64)
        current = _readonly(self.action_current, dtype=np.float64)
        change = _readonly(self.action_change, dtype=np.float64)
        count = _readonly(self.sample_count, dtype=np.int64)
        _require(
            retention.ndim == 1
            and current.shape == retention.shape
            and change.shape == retention.shape
            and count.shape == retention.shape,
            "transition arrays must be one-dimensional and aligned",
        )
        _require(
            np.all(np.isfinite(retention))
            and np.all(np.isfinite(current))
            and np.all(np.isfinite(change)),
            "transition contains non-finite values",
        )
        _require(np.all(count >= 0), "sample counts must be nonnegative")
        for name, values in {
            "velocity_retention": retention,
            "action_current": current,
            "action_change": change,
            "sample_count": count,
        }.items():
            object.__setattr__(self, name, values)


def default_mode_groups(rank: int, group_count: int) -> np.ndarray:
    """Keep translation separate and split nonconstant modes evenly."""

    _require(rank >= 2, "rank must be at least two")
    _require(2 <= group_count <= rank, "group count must lie in [2, rank]")
    groups = np.zeros(rank, dtype=np.int16)
    chunks = np.array_split(np.arange(1, rank), group_count - 1)
    for group, indices in enumerate(chunks, start=1):
        groups[indices] = group
    return groups


def _canonicalize_eigenvectors(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float64).copy()
    for mode in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, mode])))
        if result[pivot, mode] < 0.0:
            result[:, mode] *= -1.0
    return result


def deterministic_farthest_point_sample(
    points_m: np.ndarray,
    candidate_indices: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select a deterministic prefix-supported, geometry-spanning subset."""

    points = np.asarray(points_m, dtype=np.float64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and np.all(np.isfinite(points)),
        "points must have finite shape (point, 3)",
    )
    _require(
        candidates.ndim == 1
        and len(candidates) >= 2
        and len(np.unique(candidates)) == len(candidates)
        and np.all(candidates >= 0)
        and np.all(candidates < len(points)),
        "candidate indices must be unique valid point IDs",
    )
    _require(2 <= count <= len(candidates), "sample count exceeds candidates")
    ordered = np.sort(candidates, kind="stable")
    centroid = np.mean(points[ordered], axis=0)
    radial = np.sum(np.square(points[ordered] - centroid), axis=1)
    first_candidates = ordered[np.isclose(radial, np.max(radial), atol=1.0e-15)]
    first = int(np.min(first_candidates))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_squared = np.sum(np.square(points[ordered] - points[first]), axis=1)
    selected_mask = ordered == first
    for position in range(1, count):
        scores = minimum_squared.copy()
        scores[selected_mask] = -np.inf
        maximum = float(np.max(scores))
        tied = ordered[
            (~selected_mask)
            & np.isclose(scores, maximum, rtol=0.0, atol=1.0e-15)
        ]
        chosen = int(np.min(tied))
        selected[position] = chosen
        selected_mask |= ordered == chosen
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum(np.square(points[ordered] - points[chosen]), axis=1),
        )
    return selected


def inverse_distance_map(
    reference_points_m: np.ndarray,
    query_points_m: np.ndarray,
    *,
    neighbor_count: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Map query points to nearby anchors with deterministic inverse-distance weights."""

    reference = np.asarray(reference_points_m, dtype=np.float64)
    query = np.asarray(query_points_m, dtype=np.float64)
    _require(
        reference.ndim == 2
        and query.ndim == 2
        and reference.shape[1] == query.shape[1] == 3
        and np.all(np.isfinite(reference))
        and np.all(np.isfinite(query)),
        "reference and query points must have finite shape (point, 3)",
    )
    _require(
        1 <= neighbor_count <= len(reference),
        "neighbor count exceeds the anchor inventory",
    )
    _require(chunk_size >= 1, "chunk size must be positive")
    indices = np.empty((len(query), neighbor_count), dtype=np.int64)
    weights = np.empty((len(query), neighbor_count), dtype=np.float64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        squared = np.sum(
            np.square(query[start:stop, None] - reference[None]),
            axis=2,
        )
        local = np.argpartition(
            squared,
            neighbor_count - 1,
            axis=1,
        )[:, :neighbor_count]
        local_squared = np.take_along_axis(squared, local, axis=1)
        order = np.argsort(local_squared, axis=1, kind="stable")
        local = np.take_along_axis(local, order, axis=1)
        local_squared = np.take_along_axis(local_squared, order, axis=1)
        local_weights = 1.0 / np.maximum(local_squared, 1.0e-16)
        exact = local_squared[:, 0] <= 1.0e-16
        local_weights /= np.sum(local_weights, axis=1, keepdims=True)
        local_weights[exact] = 0.0
        local_weights[exact, 0] = 1.0
        indices[start:stop] = local
        weights[start:stop] = local_weights
    return indices, weights


def compose_dense_endpoint_with_anchor_dynamics(
    anchor_dynamic_m: np.ndarray,
    dense_endpoint_m: np.ndarray,
    anchor_indices: np.ndarray,
    interpolation_indices: np.ndarray,
    interpolation_weights: np.ndarray,
    *,
    maximum_residual_m: float,
) -> np.ndarray:
    """Preserve dense persistence and interpolate only the forecast increment."""

    dynamic = np.asarray(anchor_dynamic_m, dtype=np.float64)
    endpoint = np.asarray(dense_endpoint_m, dtype=np.float64)
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    indices = np.asarray(interpolation_indices, dtype=np.int64)
    weights = np.asarray(interpolation_weights, dtype=np.float64)
    _require(
        dynamic.ndim == 3
        and endpoint.ndim == 2
        and dynamic.shape[1:] == (len(anchors), 3)
        and endpoint.shape[1] == 3,
        "dynamic anchors and dense endpoint have incompatible shapes",
    )
    _require(
        indices.shape == weights.shape
        and indices.ndim == 2
        and len(indices) == len(endpoint),
        "interpolation arrays have incompatible shapes",
    )
    _require(
        np.all(indices >= 0) and np.all(indices < len(anchors)),
        "interpolation index exceeds anchor inventory",
    )
    anchor_change = dynamic - endpoint[anchors][None]
    dense_change = np.sum(
        anchor_change[:, indices] * weights[None, :, :, None],
        axis=2,
    )
    result = endpoint[None] + dense_change
    norm = np.linalg.norm(result, axis=2, keepdims=True)
    result *= np.minimum(
        1.0,
        maximum_residual_m / np.maximum(norm, 1.0e-12),
    )
    return result


def build_knn_laplacian_basis(
    points_m: np.ndarray,
    *,
    rank: int,
    neighbor_count: int,
    minimum_edge_weight: float = 1.0e-8,
    minimum_bandwidth_m: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic symmetric-normalized kNN Laplacian basis."""

    points = np.asarray(points_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and np.all(np.isfinite(points)),
        "points must have finite shape (point, 3)",
    )
    _require(2 <= rank < len(points), "rank must lie in [2, point_count)")
    count = min(int(neighbor_count), len(points) - 1)
    _require(count >= 1, "neighbor count must be positive")
    squared = (
        np.sum(np.square(points), axis=1)[:, None]
        + np.sum(np.square(points), axis=1)[None]
        - 2.0 * points @ points.T
    )
    squared = np.maximum(squared, 0.0)
    np.fill_diagonal(squared, np.inf)
    indices = np.argsort(squared, axis=1, kind="stable")[:, :count]
    rows = np.arange(len(points), dtype=np.int64)[:, None]
    distances = np.sqrt(squared[rows, indices])
    positive = distances[distances > 0.0]
    bandwidth = (
        float(np.median(positive))
        if len(positive)
        else float(minimum_bandwidth_m)
    )
    bandwidth = max(bandwidth, float(minimum_bandwidth_m))
    weights = np.exp(-0.5 * np.square(distances / bandwidth))
    weights = np.maximum(weights, float(minimum_edge_weight))
    adjacency = np.zeros((len(points), len(points)), dtype=np.float64)
    adjacency[rows, indices] = weights
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = np.sum(adjacency, axis=1)
    _require(np.all(degree > 0.0), "graph contains an isolated point")
    inverse_sqrt = 1.0 / np.sqrt(degree)
    laplacian = np.eye(len(points), dtype=np.float64) - (
        inverse_sqrt[:, None] * adjacency * inverse_sqrt[None]
    )
    eigenvalues, basis = np.linalg.eigh(laplacian)
    eigenvalues = eigenvalues[:rank]
    basis = basis[:, :rank]
    basis = _canonicalize_eigenvectors(basis)
    return basis, np.asarray(eigenvalues, dtype=np.float64)


def _object_scale(points_m: np.ndarray, minimum_scale_m: float) -> float:
    centered = points_m - np.mean(points_m, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(np.square(centered), axis=1))))
    _require(
        np.isfinite(scale) and scale >= minimum_scale_m,
        "object geometry has degenerate scale",
    )
    return scale


def controller_action_field(
    baseline_positions_m: np.ndarray,
    controller_positions_m: np.ndarray,
    *,
    object_scale_m: float,
    kernel_fraction: float,
) -> np.ndarray:
    """Construct a local, rotation-equivariant controller-velocity field."""

    baseline = np.asarray(baseline_positions_m, dtype=np.float64)
    controllers = np.asarray(controller_positions_m, dtype=np.float64)
    _require(
        baseline.ndim == 3
        and baseline.shape[2] == 3
        and controllers.ndim == 3
        and controllers.shape[0] == baseline.shape[0]
        and controllers.shape[2] == 3,
        "baseline and controller trajectories have incompatible shapes",
    )
    _require(
        np.all(np.isfinite(baseline)) and np.all(np.isfinite(controllers)),
        "action-field inputs must be finite",
    )
    _require(
        object_scale_m > 0.0 and kernel_fraction > 0.0,
        "action-field scales must be positive",
    )
    result = np.zeros_like(baseline)
    controller_velocity = np.zeros_like(controllers)
    controller_velocity[1:] = (
        controllers[1:] - controllers[:-1]
    ) / object_scale_m
    width = max(
        2.0 * (object_scale_m * kernel_fraction) ** 2,
        1.0e-12,
    )
    for frame in range(1, len(baseline)):
        differences = (
            controllers[frame][None] - baseline[frame][:, None]
        )
        squared = np.sum(np.square(differences), axis=2)
        nearest = np.argmin(squared, axis=1)
        rows = np.arange(baseline.shape[1])
        proximity = np.exp(-squared[rows, nearest] / width)
        result[frame] = (
            proximity[:, None] * controller_velocity[frame, nearest]
        )
    return result


def prepare_graph_spectral_series(
    initial_points_m: np.ndarray,
    residual_m: np.ndarray,
    residual_valid: np.ndarray,
    baseline_positions_m: np.ndarray,
    controller_positions_m: np.ndarray,
    *,
    end_frame: int,
    action_end_frame: int | None = None,
    config: GraphSpectralResidualConfig,
) -> GraphSpectralSeries:
    """Project causal residual evidence and known actions into a graph basis."""

    initial = np.asarray(initial_points_m, dtype=np.float64)
    residual = np.asarray(residual_m, dtype=np.float64)
    valid = np.asarray(residual_valid, dtype=bool)
    baseline = np.asarray(baseline_positions_m, dtype=np.float64)
    controllers = np.asarray(controller_positions_m, dtype=np.float64)
    _require(
        initial.ndim == 2 and initial.shape[1] == 3,
        "initial points must have shape (point, 3)",
    )
    _require(
        residual.ndim == 3
        and residual.shape[1:] == initial.shape
        and valid.shape == residual.shape[:2],
        "residual and validity shapes changed",
    )
    _require(
        baseline.ndim == 3
        and baseline.shape[1:] == initial.shape
        and controllers.ndim == 3
        and controllers.shape[0] == len(baseline)
        and controllers.shape[2] == 3,
        "baseline must cover the residual point inventory",
    )
    action_end = end_frame if action_end_frame is None else int(action_end_frame)
    _require(
        3 <= end_frame <= len(residual),
        "end_frame must leave at least three frames",
    )
    _require(
        end_frame <= action_end <= len(baseline),
        "known actions must cover the observed residual endpoint",
    )
    basis, eigenvalues = build_knn_laplacian_basis(
        initial,
        rank=config.rank,
        neighbor_count=config.neighbor_count,
        minimum_edge_weight=config.minimum_edge_weight,
        minimum_bandwidth_m=config.minimum_bandwidth_m,
    )
    scale = _object_scale(initial, config.minimum_object_scale_m)
    filled = _temporally_fill(residual, valid, end_frame) / scale
    smoothed = filled.copy()
    for frame in range(1, end_frame):
        smoothed[frame] = (
            config.temporal_smoothing * filled[frame]
            + (1.0 - config.temporal_smoothing) * smoothed[frame - 1]
        )
    smoothed[-1] = filled[-1]
    action = controller_action_field(
        baseline[:action_end],
        controllers[:action_end],
        object_scale_m=scale,
        kernel_fraction=config.controller_kernel_fraction,
    )
    residual_coefficients = np.einsum(
        "nk,tnc->tkc",
        basis,
        smoothed,
    )
    action_coefficients = np.einsum(
        "nk,tnc->tkc",
        basis,
        action,
    )
    return GraphSpectralSeries(
        basis=basis,
        eigenvalues=eigenvalues,
        mode_groups=default_mode_groups(
            config.rank,
            config.mode_group_count,
        ),
        residual_coefficients=residual_coefficients,
        action_coefficients=action_coefficients,
        object_scale_m=scale,
    )


def fit_graph_spectral_transition(
    series: Sequence[GraphSpectralSeries],
    *,
    config: GraphSpectralResidualConfig,
    prior: GraphSpectralTransition | None = None,
    prior_strength: float = 0.0,
) -> GraphSpectralTransition:
    """Fit group dynamics, optionally shrinking a prefix fit to a source prior."""

    _require(len(series) > 0, "at least one source series is required")
    group_count = config.mode_group_count
    _require(
        (prior is None and prior_strength == 0.0)
        or (
            prior is not None
            and np.isfinite(prior_strength)
            and prior_strength > 0.0
            and len(prior.velocity_retention) == group_count
        ),
        "a local fit needs an aligned positive-strength source prior",
    )
    prior_parameters = (
        np.zeros((group_count, 3), dtype=np.float64)
        if prior is None
        else np.column_stack(
            (
                prior.velocity_retention,
                prior.action_current,
                prior.action_change,
            )
        )
    )
    design_blocks: list[list[np.ndarray]] = [[] for _ in range(group_count)]
    target_blocks: list[list[np.ndarray]] = [[] for _ in range(group_count)]
    for item in series:
        _require(
            item.residual_coefficients.shape[1] == config.rank,
            "series rank differs from the config",
        )
        coefficients = item.residual_coefficients
        action = item.action_coefficients
        velocity = np.diff(coefficients, axis=0)
        for frame in range(1, len(velocity)):
            target = velocity[frame]
            features = np.stack(
                (
                    velocity[frame - 1],
                    action[frame + 1],
                    action[frame + 1] - action[frame],
                ),
                axis=-1,
            )
            for group in range(group_count):
                selected = item.mode_groups == group
                design_blocks[group].append(
                    features[selected].reshape(-1, 3)
                )
                target_blocks[group].append(target[selected].reshape(-1))

    retention = np.zeros(group_count, dtype=np.float64)
    current = np.zeros(group_count, dtype=np.float64)
    change = np.zeros(group_count, dtype=np.float64)
    sample_count = np.zeros(group_count, dtype=np.int64)
    for group in range(group_count):
        design = np.concatenate(design_blocks[group], axis=0)
        target = np.concatenate(target_blocks[group], axis=0)
        finite = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
        design = design[finite]
        target = target[finite]
        sample_count[group] = len(target)
        if len(target) < config.minimum_group_samples:
            retention[group], current[group], change[group] = prior_parameters[group]
            continue
        gram = design.T @ design
        scale = max(float(np.trace(gram) / 3.0), 1.0e-12)
        penalty = config.ridge_fraction * scale
        right_hand_side = design.T @ target
        if prior is not None:
            penalty += prior_strength * scale
            right_hand_side += (
                prior_strength * scale * prior_parameters[group]
            )
        parameters = np.linalg.solve(
            gram + penalty * np.eye(3),
            right_hand_side,
        )
        retention[group] = np.clip(
            parameters[0],
            config.minimum_velocity_coefficient,
            config.maximum_velocity_coefficient,
        )
        current[group] = np.clip(
            parameters[1],
            -config.maximum_action_coefficient,
            config.maximum_action_coefficient,
        )
        change[group] = np.clip(
            parameters[2],
            -config.maximum_action_coefficient,
            config.maximum_action_coefficient,
        )
    return GraphSpectralTransition(
        velocity_retention=retention,
        action_current=current,
        action_change=change,
        sample_count=sample_count,
    )


def rollout_graph_spectral_transition(
    series: GraphSpectralSeries,
    transition: GraphSpectralTransition,
    *,
    start_frame: int,
    end_frame: int,
    config: GraphSpectralResidualConfig,
) -> np.ndarray:
    """Continue discrepancy from a causal endpoint under known action fields."""

    _require(
        2 <= start_frame <= len(series.residual_coefficients)
        and start_frame < end_frame <= len(series.action_coefficients),
        "rollout interval is invalid",
    )
    group_parameters = np.column_stack(
        (
            transition.velocity_retention,
            transition.action_current,
            transition.action_change,
        )
    )[series.mode_groups]
    state = series.residual_coefficients[start_frame - 1].copy()
    velocity = (
        series.residual_coefficients[start_frame - 1]
        - series.residual_coefficients[start_frame - 2]
    )
    previous_action = series.action_coefficients[start_frame - 1]
    fields = []
    for frame in range(start_frame, end_frame):
        action = series.action_coefficients[frame]
        velocity = (
            group_parameters[:, 0, None] * velocity
            + group_parameters[:, 1, None] * action
            + group_parameters[:, 2, None] * (action - previous_action)
        )
        state = state + velocity
        field_m = (series.basis @ state) * series.object_scale_m
        norm = np.linalg.norm(field_m, axis=1, keepdims=True)
        field_m *= np.minimum(
            1.0,
            config.maximum_residual_m / np.maximum(norm, 1.0e-12),
        )
        state = (series.basis.T @ field_m) / series.object_scale_m
        fields.append(field_m)
        previous_action = action
    return np.asarray(fields, dtype=np.float64)


def endpoint_persistence(
    series: GraphSpectralSeries,
    *,
    start_frame: int,
    end_frame: int,
    config: GraphSpectralResidualConfig,
) -> np.ndarray:
    """Return the exact capped endpoint field for every forecast frame."""

    _require(
        1 <= start_frame <= len(series.residual_coefficients)
        and start_frame < end_frame <= len(series.action_coefficients),
        "persistence interval is invalid",
    )
    endpoint = (
        series.basis @ series.residual_coefficients[start_frame - 1]
    ) * series.object_scale_m
    norm = np.linalg.norm(endpoint, axis=1, keepdims=True)
    endpoint *= np.minimum(
        1.0,
        config.maximum_residual_m / np.maximum(norm, 1.0e-12),
    )
    return np.broadcast_to(
        endpoint,
        (end_frame - start_frame, *endpoint.shape),
    ).copy()


def blend_with_endpoint_persistence(
    dynamic_m: np.ndarray,
    persistence_m: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    """Blend a dynamic field with exact persistence."""

    dynamic = np.asarray(dynamic_m, dtype=np.float64)
    persistence = np.asarray(persistence_m, dtype=np.float64)
    _require(
        dynamic.shape == persistence.shape and dynamic.ndim == 3,
        "dynamic and persistence fields must share shape (frame, point, 3)",
    )
    _require(0.0 <= coefficient <= 1.0, "coefficient must lie in [0, 1]")
    if coefficient == 0.0:
        return persistence.copy()
    return persistence + float(coefficient) * (dynamic - persistence)


__all__ = [
    "GraphSpectralResidualConfig",
    "GraphSpectralSeries",
    "GraphSpectralTransition",
    "blend_with_endpoint_persistence",
    "build_knn_laplacian_basis",
    "compose_dense_endpoint_with_anchor_dynamics",
    "controller_action_field",
    "default_mode_groups",
    "deterministic_farthest_point_sample",
    "endpoint_persistence",
    "fit_graph_spectral_transition",
    "inverse_distance_map",
    "prepare_graph_spectral_series",
    "rollout_graph_spectral_transition",
]
