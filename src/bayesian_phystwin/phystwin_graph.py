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
