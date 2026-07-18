"""Reconstruct the spring graph used by the official PhysTwin trainer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysTwinSpringGraphConfig:
    """Radius-neighborhood settings stored in PhysTwin's optimal parameters."""

    object_radius: float
    object_max_neighbours: int
    controller_radius: float
    controller_max_neighbours: int


@dataclass(frozen=True)
class PhysTwinPiecewiseSpringGraphConfig:
    """Region-specific object topology with a fixed controller topology."""

    object_radii: tuple[float, ...]
    object_max_neighbours: tuple[int, ...]
    controller_radius: float
    controller_max_neighbours: int


@dataclass(frozen=True)
class PhysTwinSpringGraph:
    """NumPy representation of PhysTwin's simulator initialization arrays."""

    vertices: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    masses: np.ndarray
    num_object_springs: int
    num_object_points: int | None = None


@dataclass(frozen=True)
class PartPairSpringGrouping:
    """Compact spring groups induced by unordered endpoint-part pairs."""

    group_ids: np.ndarray
    object_part_pairs: np.ndarray
    group_counts: np.ndarray
    controller_group: int | None


@dataclass(frozen=True)
class TransferredSpringField:
    """Teacher spring values transferred onto a candidate topology."""

    spring_y: np.ndarray
    exact_edge_count: int
    interpolated_edge_count: int
    removed_teacher_edge_count: int


def part_pair_spring_grouping(
    springs: np.ndarray,
    part_assignments: np.ndarray,
    *,
    num_object_springs: int,
) -> PartPairSpringGrouping:
    """Group object springs by within-part or cross-part connectivity.

    Every unique unordered pair of endpoint parts receives one group. All
    controller springs share a final group because controller vertices do not
    have material-part identities. This preserves the released graph while
    allowing cross-part springs to become softer or stiffer than within-part
    springs, a continuous proxy for piecewise topology refinement.
    """

    edges = np.asarray(springs, dtype=np.int64)
    assignments = np.asarray(part_assignments, dtype=np.int64).reshape(-1)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("springs must have shape (S, 2)")
    if not 0 < num_object_springs <= len(edges):
        raise ValueError("num_object_springs must lie in (0, S]")
    if len(assignments) == 0 or np.any(assignments < 0):
        raise ValueError("part assignments must be nonempty and nonnegative")
    object_edges = edges[:num_object_springs]
    if np.any(object_edges < 0) or np.any(object_edges >= len(assignments)):
        raise ValueError("object spring endpoint exceeds the part assignments")

    endpoint_parts = assignments[object_edges]
    unordered_pairs = np.sort(endpoint_parts, axis=1)
    object_part_pairs, object_group_ids = np.unique(
        unordered_pairs,
        axis=0,
        return_inverse=True,
    )
    group_ids = np.empty(len(edges), dtype=np.int32)
    group_ids[:num_object_springs] = object_group_ids.astype(np.int32)
    controller_group = None
    if num_object_springs < len(edges):
        controller_group = len(object_part_pairs)
        group_ids[num_object_springs:] = controller_group
    group_counts = np.bincount(group_ids).astype(np.int64)
    if np.any(group_counts == 0):
        raise RuntimeError("part-pair grouping produced an empty group")
    return PartPairSpringGrouping(
        group_ids=group_ids,
        object_part_pairs=object_part_pairs.astype(np.int32),
        group_counts=group_counts,
        controller_group=controller_group,
    )


def spatial_spring_region_ids(
    vertices: np.ndarray,
    springs: np.ndarray,
    *,
    num_object_springs: int,
    region_count: int,
) -> np.ndarray:
    """Partition object springs into equal principal-axis material bands.

    The final group is reserved for controller springs. Principal-axis
    coordinates make the partition invariant to rigid camera/world rotation,
    while a skewness sign convention keeps the band order deterministic.
    """

    points = _points(vertices, name="vertices").astype(float)
    edges = np.asarray(springs, dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("springs must have shape (S, 2)")
    if not 0 < num_object_springs <= len(edges):
        raise ValueError("num_object_springs must lie in (0, S]")
    if not 2 <= region_count <= num_object_springs:
        raise ValueError("region_count must lie in [2, num_object_springs]")
    if np.any(edges < 0) or np.any(edges >= len(points)):
        raise ValueError("spring endpoint exceeds the vertex array")

    object_edges = edges[:num_object_springs]
    midpoints = 0.5 * (
        points[object_edges[:, 0]] + points[object_edges[:, 1]]
    )
    centered = midpoints - np.mean(midpoints, axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= 0.0:
        raise ValueError("object spring midpoints have no spatial extent")
    axis = right[0]
    coordinate = centered @ axis
    skewness = float(np.sum(np.power(coordinate, 3)))
    if skewness < 0.0:
        coordinate = -coordinate
    elif abs(skewness) < 1e-12:
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            coordinate = -coordinate

    order = np.lexsort((np.arange(num_object_springs), coordinate))
    object_regions = np.empty(num_object_springs, dtype=np.int32)
    for region, selected in enumerate(np.array_split(order, region_count)):
        object_regions[selected] = region
    group_ids = np.full(len(edges), region_count, dtype=np.int32)
    group_ids[:num_object_springs] = object_regions
    return group_ids


def _points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain finite values")
    return points.astype(np.float32, copy=False)


def _radius_neighbors(
    points: np.ndarray,
    query: np.ndarray,
    *,
    radius: float,
    maximum: int,
    self_index: int | None = None,
) -> np.ndarray:
    """Match Open3D's sorted hybrid radius/k-nearest query contract."""

    delta = points.astype(np.float64) - query.astype(np.float64)
    distance_sq = np.einsum("ij,ij->i", delta, delta)
    indices = np.flatnonzero(distance_sq <= radius * radius)
    if self_index is None:
        order = np.lexsort((indices, distance_sq[indices]))
    else:
        # Open3D returns the queried point first, including when points coincide.
        self_rank = indices != self_index
        order = np.lexsort((indices, self_rank, distance_sq[indices]))
    return indices[order[:maximum]]


def build_phystwin_spring_graph(
    structure_points: np.ndarray,
    controller_points: np.ndarray | None,
    *,
    config: PhysTwinSpringGraphConfig,
) -> PhysTwinSpringGraph:
    """Build springs in the same object-then-controller order as PhysTwin.

    PhysTwin converts the processed arrays to float32 tensors before passing
    them through Open3D. This function performs the same cast before searching,
    which matters when a point lies close to a radius boundary.
    """

    object_points = _points(structure_points, name="structure_points")
    return build_piecewise_phystwin_spring_graph(
        object_points,
        controller_points,
        np.zeros(len(object_points), dtype=np.int32),
        config=PhysTwinPiecewiseSpringGraphConfig(
            object_radii=(config.object_radius,),
            object_max_neighbours=(config.object_max_neighbours,),
            controller_radius=config.controller_radius,
            controller_max_neighbours=config.controller_max_neighbours,
        ),
    )


def build_piecewise_phystwin_spring_graph(
    structure_points: np.ndarray,
    controller_points: np.ndarray | None,
    region_assignments: np.ndarray,
    *,
    config: PhysTwinPiecewiseSpringGraphConfig,
) -> PhysTwinSpringGraph:
    """Build a union of region-specific radius/KNN object topologies.

    Each object point queries neighbors using the radius and maximum-neighbor
    count of its assigned region. Undirected edges proposed by either endpoint
    are retained. When every point has region zero, this is byte-compatible
    with :func:`build_phystwin_spring_graph`.
    """

    object_points = _points(structure_points, name="structure_points")
    controls = (
        None
        if controller_points is None
        else _points(controller_points, name="controller_points")
    )
    assignments = np.asarray(region_assignments, dtype=np.int64).reshape(-1)
    radii = np.asarray(config.object_radii, dtype=float).reshape(-1)
    maximums = np.asarray(config.object_max_neighbours, dtype=np.int64).reshape(-1)
    if len(assignments) != len(object_points):
        raise ValueError("region_assignments must label every structure point")
    if len(radii) == 0 or len(maximums) != len(radii):
        raise ValueError("piecewise radii and neighbor limits must have equal size")
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("spring radii must be positive and finite")
    if config.controller_radius <= 0.0 or not np.isfinite(config.controller_radius):
        raise ValueError("spring radii must be positive and finite")
    if np.any(maximums < 1):
        raise ValueError("object_max_neighbours must be positive")
    if config.controller_max_neighbours < 1:
        raise ValueError("controller_max_neighbours must be positive")
    if np.any(assignments < 0) or np.any(assignments >= len(radii)):
        raise ValueError("region assignment exceeds the piecewise configuration")

    springs: list[tuple[int, int]] = []
    rest_lengths: list[float] = []
    seen: set[tuple[int, int]] = set()

    for point_index, point in enumerate(object_points):
        region = int(assignments[point_index])
        neighbors = _radius_neighbors(
            object_points,
            point,
            radius=float(radii[region]),
            maximum=int(maximums[region]),
            self_index=point_index,
        )
        # The official builder drops the first result because it is the query.
        for neighbor_index in neighbors[1:]:
            neighbor = int(neighbor_index)
            edge = (min(point_index, neighbor), max(point_index, neighbor))
            distance = float(
                np.linalg.norm(
                    object_points[point_index].astype(np.float64)
                    - object_points[neighbor].astype(np.float64)
                )
            )
            if edge in seen or distance <= 1e-4:
                continue
            seen.add(edge)
            springs.append((point_index, neighbor))
            rest_lengths.append(distance)

    num_object_springs = len(springs)
    if controls is None:
        vertices = object_points
    else:
        object_count = len(object_points)
        vertices = np.concatenate((object_points, controls), axis=0)
        for control_index, control in enumerate(controls):
            neighbors = _radius_neighbors(
                object_points,
                control,
                radius=config.controller_radius,
                maximum=config.controller_max_neighbours,
            )
            for neighbor_index in neighbors:
                neighbor = int(neighbor_index)
                springs.append((object_count + control_index, neighbor))
                rest_lengths.append(
                    float(
                        np.linalg.norm(
                            control.astype(np.float64)
                            - object_points[neighbor].astype(np.float64)
                        )
                    )
                )

    spring_array = np.asarray(springs, dtype=np.int32).reshape(-1, 2)
    rest_array = np.asarray(rest_lengths, dtype=np.float32)
    return PhysTwinSpringGraph(
        vertices=np.asarray(vertices, dtype=np.float32),
        springs=spring_array,
        rest_lengths=rest_array,
        masses=np.ones(len(vertices), dtype=np.float32),
        num_object_springs=num_object_springs,
        num_object_points=len(object_points),
    )


def _graph_object_count(graph: PhysTwinSpringGraph) -> int:
    if graph.num_object_points is not None:
        count = int(graph.num_object_points)
    elif graph.num_object_springs < len(graph.springs):
        controller_edges = graph.springs[graph.num_object_springs :]
        count = int(np.min(np.max(controller_edges, axis=1)))
    else:
        count = len(graph.vertices)
    if not 1 <= count <= len(graph.vertices):
        raise ValueError("graph has an invalid object-point boundary")
    return count


def transfer_teacher_spring_field(
    teacher_graph: PhysTwinSpringGraph,
    candidate_graph: PhysTwinSpringGraph,
    teacher_spring_y: Sequence[float] | np.ndarray,
) -> TransferredSpringField:
    """Transfer a positive teacher field while preserving shared edges exactly.

    New object edges receive the geometric mean of robust endpoint-local
    stiffness summaries. This is deterministic and avoids turning topology
    search into an implicit nearest-midpoint model-selection step.
    """

    teacher_y = np.asarray(teacher_spring_y, dtype=float).reshape(-1)
    if len(teacher_y) != len(teacher_graph.springs):
        raise ValueError("teacher spring field must match the teacher graph")
    if np.any(~np.isfinite(teacher_y)) or np.any(teacher_y <= 0.0):
        raise ValueError("teacher spring values must be finite and positive")
    if teacher_graph.vertices.shape != candidate_graph.vertices.shape or not np.array_equal(
        teacher_graph.vertices,
        candidate_graph.vertices,
    ):
        raise ValueError("teacher and candidate graphs must share ordered vertices")

    teacher_edges = {
        tuple(sorted((int(first), int(second)))): index
        for index, (first, second) in enumerate(teacher_graph.springs)
    }
    if len(teacher_edges) != len(teacher_graph.springs):
        raise ValueError("teacher graph contains duplicate undirected edges")
    object_count = _graph_object_count(teacher_graph)
    if object_count != _graph_object_count(candidate_graph):
        raise ValueError("teacher and candidate object-point boundaries differ")
    incident_logs: list[list[float]] = [[] for _ in range(object_count)]
    for edge_index, (first, second) in enumerate(
        teacher_graph.springs[: teacher_graph.num_object_springs]
    ):
        value = float(np.log(teacher_y[edge_index]))
        incident_logs[int(first)].append(value)
        incident_logs[int(second)].append(value)
    global_object_log = float(
        np.median(np.log(teacher_y[: teacher_graph.num_object_springs]))
    )
    local_logs = np.asarray(
        [np.median(values) if values else global_object_log for values in incident_logs],
        dtype=float,
    )
    controller_values = teacher_y[teacher_graph.num_object_springs :]
    controller_fallback = float(
        np.median(np.log(controller_values))
        if len(controller_values)
        else global_object_log
    )

    candidate_y = np.empty(len(candidate_graph.springs), dtype=np.float32)
    exact = 0
    interpolated = 0
    for edge_index, (first_raw, second_raw) in enumerate(candidate_graph.springs):
        first, second = int(first_raw), int(second_raw)
        key = tuple(sorted((first, second)))
        teacher_index = teacher_edges.get(key)
        if teacher_index is not None:
            candidate_y[edge_index] = teacher_y[teacher_index]
            exact += 1
            continue
        if edge_index < candidate_graph.num_object_springs:
            log_value = 0.5 * (local_logs[first] + local_logs[second])
        else:
            object_endpoint = second if second < object_count else first
            log_value = 0.5 * (local_logs[object_endpoint] + controller_fallback)
        candidate_y[edge_index] = np.exp(log_value)
        interpolated += 1
    removed = len(set(teacher_edges) - {
        tuple(sorted((int(first), int(second))))
        for first, second in candidate_graph.springs
    })
    return TransferredSpringField(
        spring_y=candidate_y,
        exact_edge_count=exact,
        interpolated_edge_count=interpolated,
        removed_teacher_edge_count=removed,
    )
