"""Conservative source-depth and redundant-view observation fusion.

The source RGB-D channel retains authority over local surface-normal motion.
Redundant multiview triangulation may alter only the tangent component, and
only where both channels already provide a valid observation. The operation
therefore cannot manufacture graph support or confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultiviewTangentFusion:
    """Result of residual-independent tangent-only observation fusion."""

    points_world_m: np.ndarray
    valid: np.ndarray
    priority_identities: np.ndarray
    multiview_availability: np.ndarray
    tangent_projectors: np.ndarray
    fused_update: np.ndarray


def _nearest_neighbor_indices(
    points: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        return np.asarray(cKDTree(points).query(points, k=count)[1], dtype=int)
    except (ImportError, ValueError):
        pass

    indices = np.empty((len(points), count), dtype=int)
    chunk_size = max(1, min(512, 2_000_000 // max(len(points), 1)))
    squared_norm = np.einsum("ij,ij->i", points, points)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        distance_squared = (
            squared_norm[start:stop, None]
            + squared_norm[None, :]
            - 2.0 * points[start:stop] @ points.T
        )
        nearest = np.argpartition(
            distance_squared,
            kth=count - 1,
            axis=1,
        )[:, :count]
        indices[start:stop] = nearest
    return indices


def local_surface_tangent_projectors(
    initial_points_world_m: np.ndarray,
    *,
    neighbor_count: int = 16,
) -> np.ndarray:
    """Estimate one deterministic local surface-tangent projector per point."""

    points = np.asarray(initial_points_world_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("initial points must have shape (point, 3)")
    if len(points) < 3:
        raise ValueError("at least three initial points are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("initial points must be finite")
    if neighbor_count < 3:
        raise ValueError("neighbor count must be at least three")

    count = min(int(neighbor_count), len(points))
    indices = _nearest_neighbor_indices(points, count=count)
    if indices.ndim == 1:
        indices = indices[:, None]

    projectors = np.empty((len(points), 3, 3), dtype=float)
    identity = np.eye(3, dtype=float)
    for point_index, neighbors in enumerate(indices):
        local = points[np.asarray(neighbors, dtype=int)]
        centered = local - np.mean(local, axis=0, keepdims=True)
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        projectors[point_index] = identity - np.outer(normal, normal)
    return projectors


def fuse_source_normal_multiview_tangent(
    source_points_world_m: np.ndarray,
    source_valid: np.ndarray,
    multiview_points_world_m: np.ndarray,
    multiview_valid: np.ndarray,
    initial_points_world_m: np.ndarray,
    *,
    minimum_multiview_availability_fraction: float,
    neighbor_count: int = 16,
) -> MultiviewTangentFusion:
    """Fuse tangent updates without changing the source observation support."""

    source = np.asarray(source_points_world_m)
    multiview = np.asarray(multiview_points_world_m)
    source_mask = np.asarray(source_valid, dtype=bool)
    multiview_mask = np.asarray(multiview_valid, dtype=bool)
    initial = np.asarray(initial_points_world_m, dtype=float)

    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source points must have shape (frame, point, 3)")
    if multiview.shape != source.shape:
        raise ValueError("multiview points must match source points")
    if source_mask.shape != source.shape[:2]:
        raise ValueError("source validity must match source point axes")
    if multiview_mask.shape != source.shape[:2]:
        raise ValueError("multiview validity must match source point axes")
    if initial.shape != source.shape[1:]:
        raise ValueError("initial points must match the point axis")
    if not 0.0 <= minimum_multiview_availability_fraction <= 1.0:
        raise ValueError("minimum multiview availability must lie in [0, 1]")
    if np.any(source_mask & ~np.isfinite(source).all(axis=2)):
        raise ValueError("valid source observations must be finite")
    if np.any(multiview_mask & ~np.isfinite(multiview).all(axis=2)):
        raise ValueError("valid multiview observations must be finite")

    tangent_projectors = local_surface_tangent_projectors(
        initial,
        neighbor_count=neighbor_count,
    )
    multiview_availability = np.mean(multiview_mask, axis=0)
    priority_identities = (
        multiview_availability
        >= minimum_multiview_availability_fraction
    )
    fused_update = (
        source_mask
        & multiview_mask
        & priority_identities[None, :]
    )

    fused = source.copy()
    if np.any(fused_update):
        tangent_delta = np.einsum(
            "nij,tnj->tni",
            tangent_projectors,
            multiview - source,
        )
        fused[fused_update] = (
            source[fused_update] + tangent_delta[fused_update]
        )

    return MultiviewTangentFusion(
        points_world_m=fused,
        valid=source_mask.copy(),
        priority_identities=priority_identities,
        multiview_availability=multiview_availability,
        tangent_projectors=tangent_projectors,
        fused_update=fused_update,
    )
