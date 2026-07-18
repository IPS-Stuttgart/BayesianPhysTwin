"""Low-dimensional spring fields on the canonical PhysTwin graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanonicalSpringBasis:
    """Smooth basis mapping log-stiffness coefficients to PhysTwin springs."""

    weights: np.ndarray
    object_centers: np.ndarray
    center_spring_indices: np.ndarray
    object_rank: int
    length_scale_m: float
    controller_parameter_index: int | None


def build_canonical_spring_basis(
    vertices: np.ndarray,
    springs: np.ndarray,
    *,
    num_object_springs: int,
    rank: int,
    length_scale_multiplier: float = 1.0,
) -> CanonicalSpringBasis:
    """Build a deterministic RBF basis over canonical spring midpoints.

    Object-spring rows are normalized RBF weights, so equal coefficients
    recover a global object stiffness scale. Controller springs use one
    separate coefficient and never borrow object-field evidence. Zero
    coefficients reproduce the supplied reference stiffness exactly.
    """

    points = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(springs, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("vertices must have shape (V, 3)")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("springs must have shape (S, 2)")
    if np.any(~np.isfinite(points)):
        raise ValueError("vertices must be finite")
    if np.any(edges < 0) or np.any(edges >= len(points)):
        raise ValueError("spring endpoint exceeds the vertex array")
    if not 0 < num_object_springs <= len(edges):
        raise ValueError("num_object_springs must lie in (0, S]")
    if rank < 1:
        raise ValueError("rank must be positive")
    if not np.isfinite(length_scale_multiplier) or length_scale_multiplier <= 0.0:
        raise ValueError("length_scale_multiplier must be positive and finite")

    object_edges = edges[:num_object_springs]
    midpoints = 0.5 * (points[object_edges[:, 0]] + points[object_edges[:, 1]])
    effective_rank = min(int(rank), num_object_springs)
    center_indices = _farthest_point_indices(midpoints, effective_rank)
    centers = midpoints[center_indices]
    base_length_scale = _basis_length_scale(midpoints, centers)
    length_scale = float(base_length_scale * length_scale_multiplier)

    squared_distance = np.sum(
        np.square(midpoints[:, None, :] - centers[None, :, :]),
        axis=2,
    )
    logits = -0.5 * squared_distance / (length_scale * length_scale)
    logits -= np.max(logits, axis=1, keepdims=True)
    object_weights = np.exp(logits)
    object_weights /= np.sum(object_weights, axis=1, keepdims=True)

    controller_parameter_index = None
    parameter_count = effective_rank
    if num_object_springs < len(edges):
        controller_parameter_index = parameter_count
        parameter_count += 1
    weights = np.zeros((len(edges), parameter_count), dtype=np.float32)
    weights[:num_object_springs, :effective_rank] = object_weights.astype(np.float32)
    if controller_parameter_index is not None:
        weights[num_object_springs:, controller_parameter_index] = 1.0

    return CanonicalSpringBasis(
        weights=np.ascontiguousarray(weights),
        object_centers=np.ascontiguousarray(centers.astype(np.float32)),
        center_spring_indices=np.ascontiguousarray(center_indices.astype(np.int32)),
        object_rank=effective_rank,
        length_scale_m=length_scale,
        controller_parameter_index=controller_parameter_index,
    )


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    first = _stable_extreme_index(
        np.sum(np.square(points - centroid), axis=1),
        largest=False,
    )
    selected = [first]
    minimum_squared_distance = np.sum(np.square(points - points[first]), axis=1)
    selected_mask = np.zeros(len(points), dtype=bool)
    selected_mask[first] = True
    while len(selected) < count:
        candidates = minimum_squared_distance.copy()
        candidates[selected_mask] = -np.inf
        next_index = _stable_extreme_index(candidates, largest=True)
        selected.append(next_index)
        selected_mask[next_index] = True
        squared_distance = np.sum(np.square(points - points[next_index]), axis=1)
        minimum_squared_distance = np.minimum(
            minimum_squared_distance,
            squared_distance,
        )
    return np.asarray(selected, dtype=np.int64)


def _stable_extreme_index(values: np.ndarray, *, largest: bool) -> int:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("no finite candidate remains")
    finite_values = values[finite]
    extreme = float(np.max(finite_values)) if largest else float(np.min(finite_values))
    scale = max(float(np.max(np.abs(finite_values))), 1e-12)
    tolerance = max(scale * 1e-8, 1e-12)
    if largest:
        candidates = np.flatnonzero(finite & (values >= extreme - tolerance))
    else:
        candidates = np.flatnonzero(finite & (values <= extreme + tolerance))
    return int(candidates[0])


def _basis_length_scale(midpoints: np.ndarray, centers: np.ndarray) -> float:
    if len(centers) == 1:
        distances = np.linalg.norm(midpoints - centers[0], axis=1)
        positive = distances[distances > 0.0]
    else:
        pairwise = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :],
            axis=2,
        )
        pairwise[pairwise == 0.0] = np.inf
        positive = np.min(pairwise, axis=1)
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if len(positive) == 0:
        extent = np.ptp(midpoints, axis=0)
        fallback = float(np.linalg.norm(extent))
        if fallback <= 0.0:
            fallback = 1.0
        return fallback
    return float(np.median(positive))
