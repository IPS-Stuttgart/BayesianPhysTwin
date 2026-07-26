"""Private numerical helpers for the process-discrepancy model."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


def readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def json_data(values: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def solve_spd(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError("matrix must be symmetric positive definite") from error
    intermediate = np.linalg.solve(factor, right_hand_side)
    return np.linalg.solve(factor.T, intermediate)


def logdet_spd(matrix: np.ndarray) -> float:
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError("matrix must be symmetric positive definite") from error
    return float(2.0 * np.sum(np.log(np.diag(factor))))


def cross_matrix(vector: np.ndarray) -> np.ndarray:
    x_coordinate, y_coordinate, z_coordinate = np.asarray(vector, dtype=float)
    return np.asarray(
        (
            (0.0, -z_coordinate, y_coordinate),
            (z_coordinate, 0.0, -x_coordinate),
            (-y_coordinate, x_coordinate, 0.0),
        )
    )


def expanded_coordinate_values(
    values: float | np.ndarray,
    *,
    node_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full((node_count, 3), float(array), dtype=float)
    if array.shape == (node_count,):
        return np.repeat(array[:, None], 3, axis=1)
    if array.shape == (node_count, 3):
        return array.copy()
    raise ValueError(f"{name} must be scalar, shape (node,), or shape (node, 3)")


def expanded_coordinate_mask(
    values: np.ndarray | None,
    *,
    node_count: int,
) -> np.ndarray:
    if values is None:
        return np.ones((node_count, 3), dtype=bool)
    array = np.asarray(values, dtype=bool)
    if array.shape == (node_count,):
        return np.repeat(array[:, None], 3, axis=1)
    if array.shape == (node_count, 3):
        return array.copy()
    raise ValueError("observed must have shape (node,) or shape (node, 3)")
