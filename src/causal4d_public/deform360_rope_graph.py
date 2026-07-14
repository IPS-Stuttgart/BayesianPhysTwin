"""Ordered rope graph extraction from public Deform360 3D point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RopeCenterlineConfig:
    node_count: int = 21
    initial_neighbor_count: int = 6
    maximum_neighbor_count: int = 24
    density_keep_quantile: float = 0.97
    smoothing_strength: float = 12.0
    refinement_iterations: int = 8
    pca_lower_quantile: float = 0.01
    pca_upper_quantile: float = 0.99
    length_projection_iterations: int = 32

    def __post_init__(self) -> None:
        _require(self.node_count >= 4, "rope centerline needs at least four nodes")
        _require(
            2 <= self.initial_neighbor_count <= self.maximum_neighbor_count,
            "invalid rope-graph neighbor counts",
        )
        _require(
            0.5 <= self.density_keep_quantile <= 1.0,
            "invalid density keep quantile",
        )
        _require(self.smoothing_strength >= 0.0, "smoothing must be nonnegative")
        _require(self.refinement_iterations >= 0, "iterations must be nonnegative")
        _require(
            0.0 <= self.pca_lower_quantile < self.pca_upper_quantile <= 1.0,
            "invalid robust-PCA projection quantiles",
        )
        _require(
            self.length_projection_iterations >= 1,
            "length projection needs at least one iteration",
        )


def _resample_polyline(points: np.ndarray, node_count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    _require(
        values.ndim == 2 and values.shape[1] == 3 and len(values) >= 2,
        "polyline must have shape (N,3) with at least two points",
    )
    edge_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    keep = np.concatenate(([True], edge_lengths > 1e-10))
    values = values[keep]
    _require(len(values) >= 2, "polyline has zero arc length")
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1)))
    )
    targets = np.linspace(0.0, cumulative[-1], node_count)
    output = np.column_stack(
        [np.interp(targets, cumulative, values[:, axis]) for axis in range(3)]
    )
    return output


def _minimum_spanning_diameter(
    points: np.ndarray, config: RopeCenterlineConfig
) -> tuple[np.ndarray, int]:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import (
            connected_components,
            dijkstra,
            minimum_spanning_tree,
        )
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("SciPy is required for rope graph extraction") from error

    tree = cKDTree(points)
    used_neighbors = config.initial_neighbor_count
    graph = None
    for neighbor_count in range(
        config.initial_neighbor_count,
        min(config.maximum_neighbor_count, len(points) - 1) + 1,
        2,
    ):
        distances, indices = tree.query(points, k=neighbor_count + 1)
        rows = np.repeat(np.arange(len(points)), neighbor_count)
        columns = indices[:, 1:].reshape(-1)
        values = distances[:, 1:].reshape(-1)
        directed = coo_matrix(
            (values, (rows, columns)), shape=(len(points), len(points))
        ).tocsr()
        graph = directed.maximum(directed.T)
        component_count = connected_components(
            graph, directed=False, return_labels=False
        )
        used_neighbors = neighbor_count
        if component_count == 1:
            break
    _require(graph is not None, "cannot build a rope neighborhood graph")
    _require(
        connected_components(graph, directed=False, return_labels=False) == 1,
        "rope point cloud remains disconnected at the maximum neighbor count",
    )
    spanning = minimum_spanning_tree(graph)
    spanning = (spanning + spanning.T).tocsr()
    first_distances = dijkstra(spanning, directed=False, indices=0)
    first = int(np.argmax(first_distances))
    second_distances, predecessors = dijkstra(
        spanning,
        directed=False,
        indices=first,
        return_predecessors=True,
    )
    second = int(np.argmax(second_distances))
    path = [second]
    cursor = second
    while cursor != first:
        cursor = int(predecessors[cursor])
        _require(cursor >= 0, "cannot reconstruct rope graph diameter")
        path.append(cursor)
    path.reverse()
    return points[np.asarray(path, dtype=np.int64)], used_neighbors


def _density_filter(points: np.ndarray, config: RopeCenterlineConfig) -> np.ndarray:
    if config.density_keep_quantile >= 1.0 or len(points) < 16:
        return points
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("SciPy is required for rope graph extraction") from error
    neighbor = min(6, len(points) - 1)
    distances, _ = cKDTree(points).query(points, k=neighbor + 1)
    local_scale = distances[:, -1]
    threshold = np.quantile(local_scale, config.density_keep_quantile)
    retained = points[local_scale <= threshold]
    _require(
        len(retained) >= config.node_count,
        "density filtering retained too few points for the rope graph",
    )
    return retained


def _refine_centerline(
    points: np.ndarray,
    initial: np.ndarray,
    config: RopeCenterlineConfig,
    *,
    target_length_m: float | None = None,
) -> np.ndarray:
    centerline = np.asarray(initial, dtype=np.float64)
    node_count = len(centerline)
    second_difference = np.zeros((node_count - 2, node_count), dtype=np.float64)
    for row in range(node_count - 2):
        second_difference[row, row : row + 3] = (1.0, -2.0, 1.0)
    regularizer = config.smoothing_strength * (second_difference.T @ second_difference)
    for _ in range(config.refinement_iterations):
        squared_distance = np.sum(
            (points[:, None, :] - centerline[None, :, :]) ** 2,
            axis=2,
        )
        assignment = np.argmin(squared_distance, axis=1)
        counts = np.bincount(assignment, minlength=node_count).astype(np.float64)
        sums = np.zeros((node_count, 3), dtype=np.float64)
        np.add.at(sums, assignment, points)
        # A small anchor keeps temporarily empty nodes ordered while data and
        # curvature determine the actual curve.
        anchor_weight = max(float(np.median(counts[counts > 0])) * 1e-3, 1e-6)
        system = np.diag(counts + anchor_weight) + regularizer
        right = sums + anchor_weight * centerline
        updated = np.linalg.solve(system, right)
        centerline = _resample_polyline(updated, node_count)
        if target_length_m is not None:
            centerline = _project_equal_edge_lengths(
                centerline,
                target_length_m,
                iterations=config.length_projection_iterations,
            )
    return centerline


def _project_equal_edge_lengths(
    points: np.ndarray, target_length_m: float, *, iterations: int
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).copy()
    _require(target_length_m > 0.0, "target rope length must be positive")
    _require(iterations >= 1, "length projection needs at least one iteration")
    segment_length = target_length_m / (len(values) - 1)
    for _ in range(iterations):
        for edge in range(len(values) - 1):
            difference = values[edge + 1] - values[edge]
            length = float(np.linalg.norm(difference))
            if length <= 1e-12:
                continue
            correction = 0.5 * (length - segment_length) * difference / length
            values[edge] += correction
            values[edge + 1] -= correction
        for edge in range(len(values) - 2, -1, -1):
            difference = values[edge + 1] - values[edge]
            length = float(np.linalg.norm(difference))
            if length <= 1e-12:
                continue
            correction = 0.5 * (length - segment_length) * difference / length
            values[edge] += correction
            values[edge + 1] -= correction
    total = float(np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1)))
    _require(total > 1e-12, "length projection collapsed the rope")
    centroid = np.mean(values, axis=0)
    values = centroid + (values - centroid) * (target_length_m / total)
    return values


def extract_rope_centerline(
    points_world_m: np.ndarray,
    *,
    config: RopeCenterlineConfig | None = None,
    initial_centerline_m: np.ndarray | None = None,
    reference_centerline_m: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract an ordered, equally spaced chain from a rope point cloud."""

    cfg = config or RopeCenterlineConfig()
    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        "rope point cloud must have shape (N,3)",
    )
    _require(len(points) >= cfg.node_count, "rope point cloud has too few points")
    _require(np.all(np.isfinite(points)), "rope point cloud contains non-finite values")
    retained = _density_filter(points, cfg)
    diameter = None
    used_neighbors = None
    initialization = "graph-diameter"
    if initial_centerline_m is None:
        diameter, used_neighbors = _minimum_spanning_diameter(retained, cfg)
        initial = _resample_polyline(diameter, cfg.node_count)
    else:
        initial = np.asarray(initial_centerline_m, dtype=np.float64)
        _require(
            initial.shape == (cfg.node_count, 3) and np.all(np.isfinite(initial)),
            "initial centerline shape or values are invalid",
        )
        initialization = "provided-centerline"
    target_length = None
    if initial_centerline_m is not None:
        target_length = float(np.sum(np.linalg.norm(np.diff(initial, axis=0), axis=1)))
    centerline = _refine_centerline(
        retained,
        initial,
        cfg,
        target_length_m=target_length,
    )

    orientation = "graph-diameter"
    if reference_centerline_m is not None:
        reference = np.asarray(reference_centerline_m, dtype=np.float64)
        _require(
            reference.shape == centerline.shape and np.all(np.isfinite(reference)),
            "reference centerline shape or values are invalid",
        )
        forward = float(np.sum((centerline - reference) ** 2))
        reverse = float(np.sum((centerline[::-1] - reference) ** 2))
        if reverse < forward:
            centerline = centerline[::-1].copy()
            orientation = "reference-reversed"
        else:
            orientation = "reference-forward"

    distance = np.linalg.norm(
        retained[:, None, :] - centerline[None, :, :], axis=2
    ).min(axis=1)
    edge_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    diagnostics = {
        "input_point_count": len(points),
        "density_retained_point_count": len(retained),
        "initialization": initialization,
        "fixed_length_m": target_length,
        "neighbor_count_used": used_neighbors,
        "graph_diameter_vertex_count": None if diameter is None else len(diameter),
        "graph_diameter_length_m": (
            None
            if diameter is None
            else float(np.sum(np.linalg.norm(np.diff(diameter, axis=0), axis=1)))
        ),
        "centerline_node_count": len(centerline),
        "centerline_length_m": float(np.sum(edge_lengths)),
        "edge_length_coefficient_of_variation": float(
            np.std(edge_lengths) / np.mean(edge_lengths)
        ),
        "point_to_centerline_node_distance_m": {
            "median": float(np.median(distance)),
            "p95": float(np.quantile(distance, 0.95)),
            "maximum": float(np.max(distance)),
        },
        "orientation": orientation,
    }
    return centerline, diagnostics


def initialize_rope_centerline_pca(
    points_world_m: np.ndarray,
    *,
    config: RopeCenterlineConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Initialize an approximately straight rope from its robust principal axis."""

    cfg = config or RopeCenterlineConfig()
    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        "rope point cloud must have shape (N,3)",
    )
    _require(len(points) >= cfg.node_count, "rope point cloud has too few points")
    _require(np.all(np.isfinite(points)), "rope point cloud contains non-finite values")
    retained = points
    center = np.median(retained, axis=0)
    _, singular_values, right = np.linalg.svd(retained - center, full_matrices=False)
    axis = right[0]
    dominant_coordinate = int(np.argmax(np.abs(axis)))
    if axis[dominant_coordinate] < 0.0:
        axis = -axis
    projection = (retained - center) @ axis
    lower, upper = np.quantile(
        projection, [cfg.pca_lower_quantile, cfg.pca_upper_quantile]
    )
    _require(upper - lower > 1e-6, "robust PCA rope extent is degenerate")
    parameters = np.linspace(lower, upper, cfg.node_count)
    centerline = center + parameters[:, None] * axis
    orthogonal = (retained - center) - projection[:, None] * axis
    orthogonal_distance = np.linalg.norm(orthogonal, axis=1)
    explained = float(singular_values[0] ** 2 / np.sum(singular_values**2))
    diagnostics = {
        "initialization": "robust-principal-axis",
        "input_point_count": len(points),
        "density_retained_point_count": len(retained),
        "centerline_node_count": cfg.node_count,
        "centerline_length_m": float(upper - lower),
        "projection_quantiles": [
            cfg.pca_lower_quantile,
            cfg.pca_upper_quantile,
        ],
        "principal_axis": axis.tolist(),
        "principal_variance_fraction": explained,
        "orthogonal_distance_m": {
            "median": float(np.median(orthogonal_distance)),
            "p95": float(np.quantile(orthogonal_distance, 0.95)),
            "maximum": float(np.max(orthogonal_distance)),
        },
    }
    return centerline, diagnostics


def rope_chain_edges(node_count: int) -> np.ndarray:
    """Return the canonical open-chain edges for an ordered centerline."""

    _require(node_count >= 2, "rope chain requires at least two nodes")
    return np.column_stack(
        (
            np.arange(node_count - 1, dtype=np.int32),
            np.arange(1, node_count, dtype=np.int32),
        )
    )


__all__ = [
    "RopeCenterlineConfig",
    "extract_rope_centerline",
    "initialize_rope_centerline_pca",
    "rope_chain_edges",
]
