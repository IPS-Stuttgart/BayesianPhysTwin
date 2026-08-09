"""Stable geometry and residual-lifting primitives for PhysTwin integrations."""

from __future__ import annotations

import numpy as np


def target_validity(visible: np.ndarray, motion_valid: np.ndarray) -> np.ndarray:
    """Align visibility and transition validity to target-state frames."""

    visible_array = np.asarray(visible, dtype=bool)
    motion_array = np.asarray(motion_valid, dtype=bool)
    if visible_array.ndim != 2 or visible_array.shape[0] < 1:
        raise ValueError("visible must have shape (T>=1, N)")
    frame_count, track_count = visible_array.shape
    if motion_array.shape not in {
        (frame_count, track_count),
        (frame_count - 1, track_count),
    }:
        raise ValueError("motion_valid has an incompatible shape")
    valid = np.zeros_like(visible_array, dtype=bool)
    valid[0] = visible_array[0]
    valid[1:] = motion_array[: frame_count - 1]
    return valid


def build_lift_map(
    initial_vertices: np.ndarray,
    original_count: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic inverse-distance kNN map for untracked state nodes."""

    vertices = np.asarray(initial_vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 1:
        raise ValueError("initial_vertices must have shape (N>=1, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("initial_vertices must be finite")
    original_count = int(original_count)
    neighbors = int(neighbors)
    if not 1 <= original_count <= len(vertices):
        raise ValueError("original_count must lie within initial_vertices")
    if not 1 <= neighbors <= original_count:
        raise ValueError("neighbors exceeds the original point count")

    extra = vertices[original_count:]
    if len(extra) == 0:
        return (
            np.empty((0, neighbors), dtype=np.int64),
            np.empty((0, neighbors), dtype=float),
        )
    original = vertices[:original_count]
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(original).query(extra, k=neighbors)
        distances = np.asarray(distances, dtype=float).reshape(len(extra), neighbors)
        indices = np.asarray(indices, dtype=np.int64).reshape(len(extra), neighbors)
    except (ImportError, OSError, ValueError):
        indices = np.empty((len(extra), neighbors), dtype=np.int64)
        distances = np.empty((len(extra), neighbors), dtype=float)
        for start in range(0, len(extra), 128):
            stop = min(start + 128, len(extra))
            squared = np.sum(
                np.square(extra[start:stop, None] - original[None, :]),
                axis=2,
            )
            local = np.argpartition(squared, neighbors - 1, axis=1)[:, :neighbors]
            indices[start:stop] = local
            distances[start:stop] = np.sqrt(np.take_along_axis(squared, local, axis=1))
    inverse = 1.0 / np.maximum(distances, 1e-6)
    weights = inverse / np.sum(inverse, axis=1, keepdims=True)
    return indices, weights


def clip_residual(values: np.ndarray, maximum_norm: float) -> np.ndarray:
    """Clip every three-dimensional residual vector to ``maximum_norm``."""

    residual = np.asarray(values, dtype=float)
    maximum_norm = float(maximum_norm)
    if residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("values must have shape (T, N, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("values must be finite")
    if not np.isfinite(maximum_norm) or maximum_norm <= 0.0:
        raise ValueError("maximum_norm must be positive and finite")
    norms = np.linalg.norm(residual, axis=2, keepdims=True)
    scale = np.minimum(1.0, maximum_norm / np.maximum(norms, 1e-12))
    return residual * scale


def lift_residual(
    tracked_residual: np.ndarray,
    state_count: int,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    maximum_norm: float,
) -> np.ndarray:
    """Lift tracked residuals onto the complete PhysTwin state."""

    tracked = np.asarray(tracked_residual, dtype=float)
    if tracked.ndim != 3 or tracked.shape[2] != 3:
        raise ValueError("tracked_residual must have shape (T, N, 3)")
    if not np.all(np.isfinite(tracked)):
        raise ValueError("tracked_residual must be finite")
    state_count = int(state_count)
    original_count = tracked.shape[1]
    if state_count < original_count:
        raise ValueError("state_count cannot be below the tracked point count")
    neighbor_indices = np.asarray(indices, dtype=np.int64)
    neighbor_weights = np.asarray(weights, dtype=float)
    extra_count = state_count - original_count
    if (
        neighbor_indices.ndim != 2
        or neighbor_indices.shape != neighbor_weights.shape
        or neighbor_indices.shape[0] != extra_count
    ):
        raise ValueError("lift map must identify every untracked state node")
    if not np.all(np.isfinite(neighbor_weights)) or np.any(neighbor_weights < 0.0):
        raise ValueError("lift weights must be finite and nonnegative")
    if extra_count:
        if neighbor_indices.shape[1] < 1:
            raise ValueError("lift map must contain at least one neighbor")
        if np.any(neighbor_indices < 0) or np.any(neighbor_indices >= original_count):
            raise ValueError("lift map references an unavailable tracked node")
        if not np.allclose(np.sum(neighbor_weights, axis=1), 1.0):
            raise ValueError("lift weights must sum to one")

    lifted: np.ndarray = np.zeros((len(tracked), state_count, 3), dtype=float)
    lifted[:, :original_count] = tracked
    if extra_count:
        lifted[:, original_count:] = np.sum(
            tracked[:, neighbor_indices] * neighbor_weights[None, :, :, None],
            axis=2,
        )
    return clip_residual(lifted, maximum_norm)


__all__ = [
    "build_lift_map",
    "clip_residual",
    "lift_residual",
    "target_validity",
]
