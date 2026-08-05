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


@dataclass(frozen=True)
class CanonicalTriplaneSpringBasis:
    """Sparse bilinear map from canonical tri-planes to spring log scales."""

    parameter_indices: np.ndarray
    interpolation_weights: np.ndarray
    resolution: int
    parameter_count: int
    controller_parameter_index: int | None
    canonical_center: np.ndarray
    canonical_rotation: np.ndarray
    canonical_minimum: np.ndarray
    canonical_maximum: np.ndarray


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


def build_canonical_triplane_spring_basis(
    vertices: np.ndarray,
    springs: np.ndarray,
    *,
    num_object_springs: int,
    resolution: int | None = None,
) -> CanonicalTriplaneSpringBasis:
    """Build a sparse canonical tri-plane interpolation map.

    Each object spring queries four bilinear neighbours on each of the XY, YZ,
    and XZ planes. The three plane values are averaged, yielding twelve sparse
    coefficients per spring regardless of plane resolution. Controller springs
    use one separate scalar and cannot absorb object-field evidence. A zero
    coefficient vector reproduces the reference spring field exactly.
    """

    points, edges = _validated_graph(vertices, springs, num_object_springs)
    if resolution is None:
        plane_resolution = max(2, int(round(0.85 * np.sqrt(num_object_springs))))
    else:
        plane_resolution = int(resolution)
    if plane_resolution < 2:
        raise ValueError("triplane resolution must be at least two")

    object_edges = edges[:num_object_springs]
    midpoints = 0.5 * (points[object_edges[:, 0]] + points[object_edges[:, 1]])
    canonical, center, rotation = _canonical_coordinates(midpoints)
    minimum = np.min(canonical, axis=0)
    maximum = np.max(canonical, axis=0)
    extent = maximum - minimum
    nondegenerate = extent > np.finfo(np.float64).eps
    normalized = np.full_like(canonical, 0.5)
    normalized[:, nondegenerate] = (
        canonical[:, nondegenerate] - minimum[nondegenerate]
    ) / extent[nondegenerate]
    normalized = np.clip(normalized, 0.0, 1.0)

    support_count = 12
    plane_size = plane_resolution * plane_resolution
    has_controller = num_object_springs < len(edges)
    controller_parameter_index = 3 * plane_size if has_controller else None
    parameter_count = 3 * plane_size + int(has_controller)
    indices = np.zeros((len(edges), support_count), dtype=np.int32)
    weights = np.zeros((len(edges), support_count), dtype=np.float32)

    for plane, (first_axis, second_axis) in enumerate(((0, 1), (1, 2), (0, 2))):
        first = normalized[:, first_axis] * (plane_resolution - 1)
        second = normalized[:, second_axis] * (plane_resolution - 1)
        first_lower = np.floor(first).astype(np.int64)
        second_lower = np.floor(second).astype(np.int64)
        first_upper = np.minimum(first_lower + 1, plane_resolution - 1)
        second_upper = np.minimum(second_lower + 1, plane_resolution - 1)
        first_fraction = first - first_lower
        second_fraction = second - second_lower
        plane_offset = plane * plane_size
        slot = 4 * plane
        plane_indices = np.stack(
            (
                second_lower * plane_resolution + first_lower,
                second_lower * plane_resolution + first_upper,
                second_upper * plane_resolution + first_lower,
                second_upper * plane_resolution + first_upper,
            ),
            axis=1,
        )
        plane_weights = np.stack(
            (
                (1.0 - first_fraction) * (1.0 - second_fraction),
                first_fraction * (1.0 - second_fraction),
                (1.0 - first_fraction) * second_fraction,
                first_fraction * second_fraction,
            ),
            axis=1,
        )
        indices[:num_object_springs, slot : slot + 4] = (
            plane_indices + plane_offset
        ).astype(np.int32)
        weights[:num_object_springs, slot : slot + 4] = (
            plane_weights / 3.0
        ).astype(np.float32)

    if controller_parameter_index is not None:
        indices[num_object_springs:] = controller_parameter_index
        weights[num_object_springs:, 0] = 1.0

    return CanonicalTriplaneSpringBasis(
        parameter_indices=np.ascontiguousarray(indices),
        interpolation_weights=np.ascontiguousarray(weights),
        resolution=plane_resolution,
        parameter_count=parameter_count,
        controller_parameter_index=controller_parameter_index,
        canonical_center=np.ascontiguousarray(center.astype(np.float32)),
        canonical_rotation=np.ascontiguousarray(rotation.astype(np.float32)),
        canonical_minimum=np.ascontiguousarray(minimum.astype(np.float32)),
        canonical_maximum=np.ascontiguousarray(maximum.astype(np.float32)),
    )


def _validated_graph(
    vertices: np.ndarray,
    springs: np.ndarray,
    num_object_springs: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    return points, edges


def _canonical_coordinates(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic right-handed PCA frame and its coordinates."""

    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(centered), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    rotation = eigenvectors[:, ::-1].copy()
    for axis in range(3):
        projections = centered @ rotation[:, axis]
        extreme = int(np.argmax(np.abs(projections)))
        if projections[extreme] < 0.0:
            rotation[:, axis] *= -1.0
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    return centered @ rotation, center, rotation


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
