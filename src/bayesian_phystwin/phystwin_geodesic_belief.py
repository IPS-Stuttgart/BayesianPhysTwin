"""Topology-aware decoding for recursive PhysTwin discrepancy beliefs.

The default online-belief decoder measures RBF distance in the object's
current Euclidean embedding.  That is inexpensive, but two material regions
which approach during a fold or self-contact can exchange local corrections
despite being far apart on the object.  This module instead measures distance
on a fixed, prefix-time material graph.

The graph may come from a simulator mesh/spring topology (preferred), or from
a deterministic k-nearest-neighbour proxy built once from frame-zero geometry.
No trajectory future is required by either path.  Query and centre IDs are
material identities, so the same graph can be reused after arbitrary folding.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .phystwin_online_belief import (
    BeliefFieldPrediction,
    RecursiveRbfBeliefConfig,
    RecursiveRbfBeliefSnapshot,
)


@dataclass(frozen=True)
class MaterialGeodesicGraph:
    """An immutable weighted material graph fixed before online updates."""

    reference_positions_m: np.ndarray
    edges: np.ndarray
    construction: str = "explicit_material_topology"

    def __post_init__(self) -> None:
        positions = np.asarray(self.reference_positions_m, dtype=float).copy()
        edges = np.asarray(self.edges, dtype=np.int64).copy()
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("reference_positions_m must have nonempty shape (N, 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("reference positions must be finite")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2)")
        if len(positions) > 1 and len(edges) == 0:
            raise ValueError("a multi-node material graph must contain edges")
        if len(edges):
            if np.any(edges < 0) or np.any(edges >= len(positions)):
                raise ValueError("material graph edge index is out of bounds")
            if np.any(edges[:, 0] == edges[:, 1]):
                raise ValueError("material graph contains a self edge")
            canonical = np.sort(edges, axis=1)
            if len(np.unique(canonical, axis=0)) != len(canonical):
                raise ValueError("material graph contains duplicate edges")
            edges = canonical[np.lexsort((canonical[:, 1], canonical[:, 0]))]
            edge_lengths = np.linalg.norm(
                positions[edges[:, 1]] - positions[edges[:, 0]], axis=1
            )
            if np.any(edge_lengths <= 0.0) or not np.all(np.isfinite(edge_lengths)):
                raise ValueError("material graph edges must have positive length")
        if not isinstance(self.construction, str) or not self.construction:
            raise ValueError("construction must be a nonempty string")
        positions.setflags(write=False)
        edges.setflags(write=False)
        object.__setattr__(self, "reference_positions_m", positions)
        object.__setattr__(self, "edges", edges)

    @property
    def node_count(self) -> int:
        return len(self.reference_positions_m)

    @property
    def edge_lengths_m(self) -> np.ndarray:
        lengths = np.linalg.norm(
            self.reference_positions_m[self.edges[:, 1]]
            - self.reference_positions_m[self.edges[:, 0]],
            axis=1,
        )
        lengths.setflags(write=False)
        return lengths


def build_reference_knn_geodesic_graph(
    reference_positions_m: np.ndarray,
    *,
    neighbor_count: int = 6,
    maximum_edge_length_m: float | None = None,
) -> MaterialGeodesicGraph:
    """Build a deterministic prefix-only proxy for unavailable topology.

    Directed KNN selections are symmetrized by union.  Distance ties are
    broken by material ID.  The function deliberately does *not* connect
    remaining components with arbitrary long edges: an unsupported component
    should receive only the global correction rather than a false local
    shortcut.  Explicit physical edges should be used whenever available.
    """

    positions = np.asarray(reference_positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("reference_positions_m must have shape (N, 3), N >= 2")
    if not np.all(np.isfinite(positions)):
        raise ValueError("reference positions must be finite")
    if not isinstance(neighbor_count, (int, np.integer)) or neighbor_count < 1:
        raise ValueError("neighbor_count must be a positive integer")
    if maximum_edge_length_m is not None and (
        not np.isfinite(maximum_edge_length_m) or maximum_edge_length_m <= 0.0
    ):
        raise ValueError("maximum_edge_length_m must be positive when provided")

    count = len(positions)
    neighbors = min(int(neighbor_count), count - 1)
    squared_distance = np.sum(
        np.square(positions[:, None, :] - positions[None, :, :]), axis=2
    )
    edge_set: set[tuple[int, int]] = set()
    ids = np.arange(count, dtype=np.int64)
    for source in range(count):
        order = np.lexsort((ids, squared_distance[source]))
        selected = order[order != source][:neighbors]
        for target in selected:
            distance = float(np.sqrt(squared_distance[source, target]))
            if distance <= 0.0:
                continue
            if maximum_edge_length_m is not None and distance > maximum_edge_length_m:
                continue
            edge_set.add(tuple(sorted((source, int(target)))))
    if not edge_set:
        raise ValueError("the KNN constraints produced no positive-length edge")
    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    return MaterialGeodesicGraph(
        reference_positions_m=positions,
        edges=edges,
        construction=f"frame_zero_symmetric_union_{neighbors}nn",
    )


def geodesic_distances_to_centers_m(
    graph: MaterialGeodesicGraph,
    center_ids: np.ndarray,
) -> np.ndarray:
    """Return shortest-path distances with shape ``(N, K)``.

    Distances to another connected component remain infinity.  Dijkstra is
    implemented with the standard library so the core decoder retains the
    package's NumPy-only dependency contract.
    """

    centers = np.asarray(center_ids, dtype=np.int64)
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError("center_ids must be a nonempty vector")
    if np.any(centers < 0) or np.any(centers >= graph.node_count):
        raise ValueError("center ID exceeds the material graph")
    if len(np.unique(centers)) != len(centers):
        raise ValueError("center_ids must be unique")

    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(graph.node_count)
    ]
    for (left, right), length in zip(
        graph.edges, graph.edge_lengths_m, strict=True
    ):
        adjacency[int(left)].append((int(right), float(length)))
        adjacency[int(right)].append((int(left), float(length)))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])

    output = np.full((graph.node_count, len(centers)), np.inf, dtype=float)
    for column, center in enumerate(centers):
        center = int(center)
        output[center, column] = 0.0
        queue: list[tuple[float, int]] = [(0.0, center)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > output[node, column]:
                continue
            for neighbor, edge_length in adjacency[node]:
                candidate = distance + edge_length
                if candidate < output[neighbor, column]:
                    output[neighbor, column] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
    output.setflags(write=False)
    return output


def deterministic_geodesic_farthest_point_ids(
    graph: MaterialGeodesicGraph,
    candidate_ids: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select material centres by deterministic farthest-point geodesics."""

    candidates = np.asarray(candidate_ids, dtype=np.int64)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("candidate_ids must be a nonempty vector")
    if np.any(candidates < 0) or np.any(candidates >= graph.node_count):
        raise ValueError("candidate ID exceeds the material graph")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate_ids must be unique")
    if count < 1:
        raise ValueError("count must be positive")

    ordered = np.sort(candidates, kind="mergesort")
    selected = [int(ordered[0])]
    selected_mask = ordered == selected[0]
    minimum_distance = geodesic_distances_to_centers_m(
        graph, np.asarray(selected)
    )[ordered, 0]
    while len(selected) < min(count, len(ordered)):
        candidate_distance = minimum_distance.copy()
        candidate_distance[selected_mask] = -np.inf
        maximum = float(np.max(candidate_distance))
        tied = ordered[(~selected_mask) & (candidate_distance == maximum)]
        next_id = int(np.min(tied))
        selected.append(next_id)
        selected_mask |= ordered == next_id
        distance = geodesic_distances_to_centers_m(
            graph, np.asarray([next_id])
        )[ordered, 0]
        minimum_distance = np.minimum(minimum_distance, distance)
    output = np.asarray(selected, dtype=np.int64)
    output.setflags(write=False)
    return output


def decode_recursive_geodesic_rbf_belief(
    belief: RecursiveRbfBeliefSnapshot,
    graph: MaterialGeodesicGraph,
    query_ids: np.ndarray,
    *,
    forecast_frames: int,
    config: RecursiveRbfBeliefConfig,
    distances_to_belief_centers_m: np.ndarray | None = None,
) -> BeliefFieldPrediction:
    """Decode local discrepancy over fixed material, rather than Euclidean, distance."""

    queries = np.asarray(query_ids, dtype=np.int64)
    if queries.ndim != 1:
        raise ValueError("query_ids must be a vector")
    if np.any(queries < 0) or np.any(queries >= graph.node_count):
        raise ValueError("query ID exceeds the material graph")
    if np.any(belief.center_ids < 0) or np.any(
        belief.center_ids >= graph.node_count
    ):
        raise ValueError("belief centre ID exceeds the material graph")
    if forecast_frames < 0:
        raise ValueError("forecast_frames must be nonnegative")
    if distances_to_belief_centers_m is None:
        distances = geodesic_distances_to_centers_m(graph, belief.center_ids)
    else:
        distances = np.asarray(distances_to_belief_centers_m, dtype=float)
        if distances.shape != (graph.node_count, len(belief.center_ids)):
            raise ValueError(
                "distances_to_belief_centers_m must have shape (N, K)"
            )
        if np.any(np.isnan(distances)) or np.any(distances < 0.0):
            raise ValueError("cached geodesic distances must be nonnegative or inf")

    active = belief.update_count > 0
    mean = np.repeat(belief.global_mean_m[None], len(queries), axis=0)
    variance = np.repeat(belief.global_variance_m2[None], len(queries), axis=0)
    process_variance = forecast_frames * config.process_std_m_per_sqrt_frame**2
    variance += process_variance

    if np.any(active) and config.local_blend > 0.0:
        distance = distances[queries][:, active]
        length_scale = max(
            belief.object_scale_m * config.length_scale_fraction,
            config.minimum_length_scale_m,
        )
        weight = np.exp(-0.5 * np.square(distance / length_scale))
        weight_sum = np.sum(weight, axis=1, keepdims=True)
        normalized = weight / np.maximum(weight_sum, 1e-15)
        unsupported = weight_sum[:, 0] < 1e-12
        normalized[unsupported] = 0.0
        mean += config.local_blend * (normalized @ belief.local_mean_m[active])
        local_variance = belief.local_variance_m2[active] + process_variance
        variance += config.local_blend**2 * (np.square(normalized) @ local_variance)

    norm = np.linalg.norm(mean, axis=1, keepdims=True)
    mean *= np.minimum(1.0, config.maximum_correction_m / np.maximum(norm, 1e-15))
    return BeliefFieldPrediction(
        mean_m=mean,
        variance_m2=np.maximum(variance, 1e-12),
    )
