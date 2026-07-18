"""Reconstruct the spring graph used by the official PhysTwin trainer."""

from __future__ import annotations

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
class PhysTwinSpringGraph:
    """NumPy representation of PhysTwin's simulator initialization arrays."""

    vertices: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    masses: np.ndarray
    num_object_springs: int


@dataclass(frozen=True)
class PartPairSpringGrouping:
    """Compact spring groups induced by unordered endpoint-part pairs."""

    group_ids: np.ndarray
    object_part_pairs: np.ndarray
    group_counts: np.ndarray
    controller_group: int | None


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

    if config.object_radius <= 0.0 or config.controller_radius <= 0.0:
        raise ValueError("spring radii must be positive")
    if config.object_max_neighbours < 1:
        raise ValueError("object_max_neighbours must be positive")
    if config.controller_max_neighbours < 1:
        raise ValueError("controller_max_neighbours must be positive")

    object_points = _points(structure_points, name="structure_points")
    controls = (
        None
        if controller_points is None
        else _points(controller_points, name="controller_points")
    )
    springs: list[tuple[int, int]] = []
    rest_lengths: list[float] = []
    seen: set[tuple[int, int]] = set()

    for point_index, point in enumerate(object_points):
        neighbors = _radius_neighbors(
            object_points,
            point,
            radius=config.object_radius,
            maximum=config.object_max_neighbours,
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
    )
