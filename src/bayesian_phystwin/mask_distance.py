"""Canonical border-aware Euclidean distance from mask interior to background."""

from __future__ import annotations

import numpy as np


def _squared_distance_transform_1d(cost: np.ndarray) -> np.ndarray:
    """Return the exact squared 1-D Euclidean distance transform."""

    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cost must be a nonempty vector")
    size = values.size
    locations = np.empty(size, dtype=np.int64)
    boundaries = np.empty(size + 1, dtype=np.float64)
    output = np.empty(size, dtype=np.float64)
    envelope = 0
    locations[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for query in range(1, size):
        while True:
            previous = int(locations[envelope])
            separation = (
                values[query]
                + float(query * query)
                - values[previous]
                - float(previous * previous)
            ) / (2.0 * (query - previous))
            if separation > boundaries[envelope]:
                break
            envelope -= 1
        envelope += 1
        locations[envelope] = query
        boundaries[envelope] = separation
        boundaries[envelope + 1] = np.inf

    envelope = 0
    for query in range(size):
        while boundaries[envelope + 1] < query:
            envelope += 1
        previous = int(locations[envelope])
        delta = query - previous
        output[query] = float(delta * delta) + values[previous]
    return output


def _interior_mask_distance_fallback(mask: np.ndarray) -> np.ndarray:
    """Pure-NumPy exact EDT used when SciPy is unavailable."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or min(values.shape, default=0) == 0:
        raise ValueError("mask must be a nonempty 2-D array")
    padded = np.pad(values, 1, constant_values=False)
    maximum_squared_distance = float(sum(padded.shape) ** 2)
    squared = np.where(padded, maximum_squared_distance, 0.0)
    row_pass = np.empty_like(squared)
    for row in range(squared.shape[0]):
        row_pass[row] = _squared_distance_transform_1d(squared[row])
    column_pass = np.empty_like(row_pass)
    for column in range(row_pass.shape[1]):
        column_pass[:, column] = _squared_distance_transform_1d(row_pass[:, column])
    return np.sqrt(column_pass)[1:-1, 1:-1]


def interior_mask_distance(mask: np.ndarray) -> np.ndarray:
    """Return border-aware Euclidean distance for pixels inside ``mask``.

    The image exterior is background. This makes a foreground pixel on an
    image edge exactly one pixel from background and gives identical
    semantics with and without SciPy.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or min(values.shape, default=0) == 0:
        raise ValueError("mask must be a nonempty 2-D array")
    padded = np.pad(values, 1, constant_values=False)
    try:
        from scipy.ndimage import distance_transform_edt
    except (ImportError, OSError):
        return _interior_mask_distance_fallback(values)
    distance = np.asarray(distance_transform_edt(padded), dtype=np.float64)
    return distance[1:-1, 1:-1]


__all__ = ["interior_mask_distance"]
