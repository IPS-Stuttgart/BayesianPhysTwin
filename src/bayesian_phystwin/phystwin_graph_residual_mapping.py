"""Geometry-regularized online residual mapping for deformable forecasts.

This is a deliberately strong, training-free control inspired by online
real-to-sim residual mapping.  A material graph is fixed from frame-zero
geometry.  At an update, the mapper fits a vector displacement field that
matches the currently observed material points while penalizing graph
Laplacian roughness.  It neither reads a future target nor changes simulator
parameters.

The regularization strength is selected from a fixed grid by exact linear-
smoother leave-one-observation-out error.  Selection therefore uses only the
measurements available at the current update.  This is a comparator, not an
implementation of Liang et al.'s PBD stiffness adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phystwin_geodesic_belief import (
    MaterialGeodesicGraph,
    build_reference_knn_geodesic_graph,
)


@dataclass(frozen=True)
class GraphResidualMappingConfig:
    """Fixed graph and causal regularization policy for residual mapping."""

    neighbor_count: int = 4
    regularization_grid: tuple[float, ...] = (
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    )
    ridge: float = 1e-6
    maximum_correction_m: float = 0.10
    minimum_observation_count: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.neighbor_count, int) or self.neighbor_count < 1:
            raise ValueError("neighbor_count must be a positive integer")
        grid = tuple(float(value) for value in self.regularization_grid)
        if not grid or any(not np.isfinite(value) or value <= 0.0 for value in grid):
            raise ValueError("regularization_grid must contain positive finite values")
        if len(set(grid)) != len(grid) or tuple(sorted(grid)) != grid:
            raise ValueError("regularization_grid must be unique and increasing")
        if not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if (
            not np.isfinite(self.maximum_correction_m)
            or self.maximum_correction_m <= 0.0
        ):
            raise ValueError("maximum_correction_m must be positive")
        if self.minimum_observation_count < 2:
            raise ValueError("minimum_observation_count must be at least two")
        object.__setattr__(self, "regularization_grid", grid)


@dataclass(frozen=True)
class GraphResidualMappingResult:
    """One causal graph-regularized displacement estimate."""

    correction_m: np.ndarray
    selected_regularization: float
    leave_one_out_rmse_m: float
    observation_count: int
    clipped_point_count: int

    def __post_init__(self) -> None:
        correction = np.asarray(self.correction_m, dtype=float).copy()
        if correction.ndim != 2 or correction.shape[1] != 3:
            raise ValueError("correction_m must have shape (N, 3)")
        if not np.all(np.isfinite(correction)):
            raise ValueError("correction_m must be finite")
        if (
            not np.isfinite(self.selected_regularization)
            or self.selected_regularization <= 0.0
        ):
            raise ValueError("selected_regularization must be positive")
        if not np.isfinite(self.leave_one_out_rmse_m):
            raise ValueError("leave_one_out_rmse_m must be finite")
        if self.observation_count < 2 or self.clipped_point_count < 0:
            raise ValueError("invalid residual-mapping counts")
        correction.setflags(write=False)
        object.__setattr__(self, "correction_m", correction)


def _normalized_laplacian(graph: MaterialGeodesicGraph):
    try:
        from scipy.sparse import coo_matrix, diags
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("SciPy is required for graph residual mapping") from error

    node_count = graph.node_count
    edges = graph.edges
    lengths = graph.edge_lengths_m
    median_length = max(float(np.median(lengths)), 1e-12)
    weights = np.exp(-0.5 * np.square(lengths / median_length))
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((weights, weights))
    adjacency = coo_matrix(
        (values, (rows, columns)), shape=(node_count, node_count)
    ).tocsc()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    positive = degree[degree > 0.0]
    scale = max(float(np.median(positive)), 1e-12)
    # A scaled combinatorial Laplacian keeps a spatially constant residual in
    # its exact null space.  The symmetric normalized Laplacian instead keeps
    # sqrt(degree), which creates endpoint artifacts on irregular material
    # graphs and is undesirable for a global object translation.
    return ((diags(degree, format="csc") - adjacency) / scale).tocsc()


def fit_graph_residual_mapping(
    reference_positions_m: np.ndarray,
    center_ids: np.ndarray,
    measured_residual_m: np.ndarray,
    available: np.ndarray,
    *,
    config: GraphResidualMappingConfig | None = None,
    graph: MaterialGeodesicGraph | None = None,
) -> GraphResidualMappingResult:
    """Fit a smooth displacement field using only one update's measurements."""

    try:
        from scipy.sparse import diags, eye
        from scipy.sparse.linalg import splu
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("SciPy is required for graph residual mapping") from error

    cfg = config or GraphResidualMappingConfig()
    reference = np.asarray(reference_positions_m, dtype=float)
    centers = np.asarray(center_ids, dtype=np.int64)
    residual = np.asarray(measured_residual_m, dtype=float)
    mask = np.asarray(available, dtype=bool).copy()
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
        raise ValueError("reference_positions_m must have shape (N, 3), N >= 2")
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_positions_m must be finite")
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError("center_ids must be a nonempty vector")
    if len(np.unique(centers)) != len(centers):
        raise ValueError("center_ids must be unique")
    if np.any(centers < 0) or np.any(centers >= len(reference)):
        raise ValueError("center ID exceeds reference geometry")
    if residual.shape != (len(centers), 3):
        raise ValueError("measured_residual_m must have shape (K, 3)")
    if mask.shape != (len(centers),):
        raise ValueError("available must have shape (K,)")
    mask &= np.all(np.isfinite(residual), axis=1)
    observed_ids = centers[mask]
    observed = residual[mask]
    if len(observed_ids) < cfg.minimum_observation_count:
        raise ValueError("too few finite observations for graph residual mapping")

    material_graph = graph or build_reference_knn_geodesic_graph(
        reference, neighbor_count=cfg.neighbor_count
    )
    if material_graph.node_count != len(reference) or not np.array_equal(
        material_graph.reference_positions_m, reference
    ):
        raise ValueError("material graph differs from reference geometry")
    laplacian = _normalized_laplacian(material_graph)
    observation_diagonal = np.zeros(len(reference), dtype=float)
    observation_diagonal[observed_ids] = 1.0
    observation_operator = diags(observation_diagonal, format="csc")
    ridge = cfg.ridge * eye(len(reference), format="csc")
    right_hand_side = np.zeros((len(reference), 3), dtype=float)
    right_hand_side[observed_ids] = observed
    selection_columns = np.zeros((len(reference), len(observed_ids)), dtype=float)
    selection_columns[observed_ids, np.arange(len(observed_ids))] = 1.0

    candidates: list[tuple[float, float, np.ndarray]] = []
    for regularization in cfg.regularization_grid:
        system = (observation_operator + regularization * laplacian + ridge).tocsc()
        factor = splu(system)
        correction = factor.solve(right_hand_side)
        influence_columns = factor.solve(selection_columns)
        hat = influence_columns[observed_ids]
        fitted = correction[observed_ids]
        denominator = 1.0 - np.diag(hat)
        if np.any(denominator <= 1e-8) or not np.all(np.isfinite(denominator)):
            score = float("inf")
        else:
            leave_one_out_error = (observed - fitted) / denominator[:, None]
            score = float(np.sqrt(np.mean(np.sum(leave_one_out_error**2, axis=1))))
        candidates.append((score, regularization, correction))

    finite = [candidate for candidate in candidates if np.isfinite(candidate[0])]
    if not finite:
        raise RuntimeError("all graph residual regularization candidates failed")
    score, regularization, correction = min(
        finite, key=lambda candidate: (candidate[0], candidate[1])
    )
    norm = np.linalg.norm(correction, axis=1, keepdims=True)
    clipped = norm[:, 0] > cfg.maximum_correction_m
    correction = correction * np.minimum(
        1.0, cfg.maximum_correction_m / np.maximum(norm, 1e-15)
    )
    return GraphResidualMappingResult(
        correction_m=correction,
        selected_regularization=regularization,
        leave_one_out_rmse_m=score,
        observation_count=len(observed_ids),
        clipped_point_count=int(np.sum(clipped)),
    )


__all__ = [
    "GraphResidualMappingConfig",
    "GraphResidualMappingResult",
    "fit_graph_residual_mapping",
]
