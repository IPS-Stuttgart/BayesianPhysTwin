"""Shape-stratified sparse physical graphs for the Deform360 replication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .deform360_rope_graph import RopeCenterlineConfig, extract_rope_centerline


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Deform360SparseGraph:
    """A sparse embedded graph with stretch/shear and bend spring families."""

    positions_m: np.ndarray
    spring_edges: np.ndarray
    spring_families: np.ndarray
    masses: np.ndarray
    stratum: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64)
        edges = np.asarray(self.spring_edges, dtype=np.int32)
        families = np.asarray(self.spring_families, dtype=np.int8)
        masses = np.asarray(self.masses, dtype=np.float64)
        _require(
            positions.ndim == 2 and positions.shape[1] == 3 and len(positions) >= 4,
            "graph positions must have shape (N,3) with at least four nodes",
        )
        _require(
            edges.ndim == 2 and edges.shape[1] == 2 and len(edges) >= len(positions) - 1,
            "graph edges must have shape (E,2)",
        )
        _require(families.shape == (len(edges),), "spring family count differs")
        _require(masses.shape == (len(positions),), "mass count differs")
        _require(
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(masses))
            and np.all(masses > 0.0),
            "graph positions and masses must be finite",
        )
        _require(
            np.all(edges >= 0) and np.all(edges < len(positions)),
            "graph edge index is out of bounds",
        )
        _require(np.all(edges[:, 0] != edges[:, 1]), "graph contains a self edge")
        canonical = np.sort(edges, axis=1)
        _require(
            len(np.unique(canonical, axis=0)) == len(edges),
            "graph contains duplicate edges",
        )
        _require(
            set(np.unique(families)).issubset({0, 1}),
            "spring families must be stretch/shear=0 or bend=1",
        )
        lengths = np.linalg.norm(
            positions[edges[:, 1]] - positions[edges[:, 0]], axis=1
        )
        _require(np.all(lengths > 1e-5), "graph contains a degenerate spring")
        for name, values in (
            ("positions_m", positions),
            ("spring_edges", edges),
            ("spring_families", families),
            ("masses", masses),
        ):
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def _edge_arrays(
    stretch_edges: set[tuple[int, int]], bend_edges: set[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    stretch = sorted(tuple(sorted(edge)) for edge in stretch_edges)
    bend = sorted(
        tuple(sorted(edge))
        for edge in bend_edges
        if tuple(sorted(edge)) not in set(stretch)
    )
    edges = np.asarray(stretch + bend, dtype=np.int32)
    families = np.concatenate(
        (
            np.zeros(len(stretch), dtype=np.int8),
            np.ones(len(bend), dtype=np.int8),
        )
    )
    return edges, families


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic Euclidean farthest-point sampling."""

    values = np.asarray(points, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 3, "points must be (N,3)")
    _require(1 <= count <= len(values), "invalid farthest-point sample count")
    center = np.mean(values, axis=0)
    centered_distance = np.sum((values - center) ** 2, axis=1)
    selected = [int(np.argmax(centered_distance))]
    nearest = np.sum((values - values[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(
            nearest, np.sum((values - values[index]) ** 2, axis=1)
        )
    return np.asarray(selected, dtype=np.int64)


def _connected_knn_edges(points: np.ndarray, neighbor_count: int) -> set[tuple[int, int]]:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is required by project
        raise RuntimeError("SciPy is required for sparse graph construction") from error

    count = len(points)
    neighbors = min(max(1, neighbor_count), count - 1)
    distances, indices = cKDTree(points).query(points, k=neighbors + 1)
    edges = {
        tuple(sorted((row, int(column))))
        for row in range(count)
        for column in indices[row, 1:]
        if row != int(column)
    }
    rows = np.asarray([edge[0] for edge in edges] + [edge[1] for edge in edges])
    columns = np.asarray([edge[1] for edge in edges] + [edge[0] for edge in edges])
    weights = np.linalg.norm(points[rows] - points[columns], axis=1)
    graph = coo_matrix((weights, (rows, columns)), shape=(count, count)).tocsr()
    spanning = minimum_spanning_tree(graph)
    span_rows, span_columns = spanning.nonzero()
    edges.update(
        tuple(sorted((int(row), int(column))))
        for row, column in zip(span_rows, span_columns, strict=True)
    )
    return edges


def build_filament_sparse_graph(
    points_world_m: np.ndarray,
    *,
    node_count: int = 21,
) -> Deform360SparseGraph:
    """Extract a chain graph from a filament visual hull."""

    centerline, centerline_diagnostics = extract_rope_centerline(
        points_world_m,
        config=RopeCenterlineConfig(node_count=node_count),
    )
    stretch = {(index, index + 1) for index in range(node_count - 1)}
    bend = {(index, index + 2) for index in range(node_count - 2)}
    edges, families = _edge_arrays(stretch, bend)
    return Deform360SparseGraph(
        positions_m=centerline,
        spring_edges=edges,
        spring_families=families,
        masses=np.ones(node_count, dtype=np.float64),
        stratum="filament",
        diagnostics={
            "construction": "ordered visual-hull centerline",
            "centerline": centerline_diagnostics,
        },
    )


def build_sheet_sparse_graph(
    points_world_m: np.ndarray,
    *,
    rows: int = 5,
    columns: int = 5,
    local_neighbor_count: int = 16,
) -> Deform360SparseGraph:
    """Fit a regular material-like lattice to a sheet visual hull."""

    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) >= rows * columns,
        "sheet hull is too small",
    )
    _require(rows >= 3 and columns >= 3, "sheet lattice is too small")
    center = np.median(points, axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, basis = np.linalg.eigh(covariance)
    plane = basis[:, np.argsort(eigenvalues)[-2:]]
    projected = (points - center) @ plane
    low = np.quantile(projected, 0.03, axis=0)
    high = np.quantile(projected, 0.97, axis=0)
    _require(np.all(high - low > 1e-4), "sheet projected extent is degenerate")
    u = np.linspace(low[0], high[0], columns)
    v = np.linspace(low[1], high[1], rows)
    targets = np.asarray([(x, y) for y in v for x in u], dtype=np.float64)
    squared = np.sum((targets[:, None] - projected[None]) ** 2, axis=2)
    neighbor_count = min(local_neighbor_count, len(points))
    nearest = np.argpartition(squared, neighbor_count - 1, axis=1)[
        :, :neighbor_count
    ]
    nearest_squared = np.take_along_axis(squared, nearest, axis=1)
    scale = np.maximum(np.median(nearest_squared, axis=1, keepdims=True), 1e-10)
    weights = np.exp(-nearest_squared / scale)
    weights /= np.sum(weights, axis=1, keepdims=True)
    positions = np.einsum("nk,nkd->nd", weights, points[nearest])

    def index(row: int, column: int) -> int:
        return row * columns + column

    stretch: set[tuple[int, int]] = set()
    bend: set[tuple[int, int]] = set()
    for row in range(rows):
        for column in range(columns):
            if column + 1 < columns:
                stretch.add((index(row, column), index(row, column + 1)))
            if row + 1 < rows:
                stretch.add((index(row, column), index(row + 1, column)))
            if row + 1 < rows and column + 1 < columns:
                stretch.add((index(row, column), index(row + 1, column + 1)))
                stretch.add((index(row + 1, column), index(row, column + 1)))
            if column + 2 < columns:
                bend.add((index(row, column), index(row, column + 2)))
            if row + 2 < rows:
                bend.add((index(row, column), index(row + 2, column)))
    edges, families = _edge_arrays(stretch, bend)
    residual = np.sqrt(np.min(squared, axis=1))
    return Deform360SparseGraph(
        positions_m=positions,
        spring_edges=edges,
        spring_families=families,
        masses=np.ones(len(positions), dtype=np.float64),
        stratum="sheet",
        diagnostics={
            "construction": "robust-PCA local-average lattice",
            "lattice_shape": [rows, columns],
            "input_point_count": len(points),
            "projection_eigenvalues": np.sort(eigenvalues).tolist(),
            "target_to_hull_projected_distance_m": {
                "median": float(np.median(residual)),
                "maximum": float(np.max(residual)),
            },
        },
    )


def build_volumetric_sparse_graph(
    points_world_m: np.ndarray,
    *,
    node_count: int = 32,
    neighbor_count: int = 4,
) -> Deform360SparseGraph:
    """Build a connected surface graph from deterministic hull samples."""

    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) >= node_count,
        "volumetric hull is too small",
    )
    sample_indices = _farthest_point_indices(points, node_count)
    positions = points[sample_indices]
    stretch = _connected_knn_edges(positions, neighbor_count)
    adjacency = [set() for _ in range(node_count)]
    for left, right in stretch:
        adjacency[left].add(right)
        adjacency[right].add(left)
    bend: set[tuple[int, int]] = set()
    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            for second in adjacency[neighbor]:
                if second != node and second not in neighbors:
                    bend.add(tuple(sorted((node, second))))
    edges, families = _edge_arrays(stretch, bend)
    return Deform360SparseGraph(
        positions_m=positions,
        spring_edges=edges,
        spring_families=families,
        masses=np.ones(node_count, dtype=np.float64),
        stratum="volumetric",
        diagnostics={
            "construction": "deterministic farthest-point kNN surface graph",
            "input_point_count": len(points),
            "sample_indices": sample_indices.tolist(),
            "neighbor_count": neighbor_count,
        },
    )


def build_sparse_graph_for_stratum(
    points_world_m: np.ndarray,
    stratum: str,
) -> Deform360SparseGraph:
    """Dispatch to the preregistered graph family for an object stratum."""

    if stratum == "filament":
        return build_filament_sparse_graph(points_world_m)
    if stratum == "sheet":
        return build_sheet_sparse_graph(points_world_m)
    if stratum == "volumetric":
        return build_volumetric_sparse_graph(points_world_m)
    raise ValueError(f"unsupported Deform360 stratum: {stratum}")


__all__ = [
    "Deform360SparseGraph",
    "build_filament_sparse_graph",
    "build_sheet_sparse_graph",
    "build_sparse_graph_for_stratum",
    "build_volumetric_sparse_graph",
]
