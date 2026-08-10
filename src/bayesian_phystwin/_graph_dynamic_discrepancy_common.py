"""Numerical and validation primitives for graph discrepancy dynamics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Final

import numpy as np

GRAPH_DYNAMIC_DISCREPANCY_SCHEMA: Final = (
    "bayesian_phystwin.graph_dynamic_discrepancy"
)
GRAPH_DYNAMIC_DISCREPANCY_VERSION: Final = 1
GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY: Final = (
    "Predictive readout/model-discrepancy belief only. It is not a latent "
    "physical-state correction, calibrated deployment uncertainty, or evidence "
    "of physical-query benefit."
)
_DEFAULT_MAXIMUM_COVARIANCE_BYTES: Final = 256 * 1024 * 1024


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    _require(np.isfinite(result), f"{name} must be finite")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    _require(result >= minimum, f"{name} must be at least {minimum}")
    return result


def _readonly(
    value: object,
    *,
    name: str,
    dtype: Any = np.float64,
    finite: bool = True,
) -> np.ndarray:
    raw = np.asarray(value)
    target = np.dtype(dtype)
    if np.issubdtype(target, np.bool_):
        _require(raw.dtype.kind == "b", f"{name} must be Boolean")
    elif np.issubdtype(target, np.integer):
        _require(
            np.issubdtype(raw.dtype, np.integer) and raw.dtype.kind != "b",
            f"{name} must contain integers",
        )
    else:
        _require(
            raw.dtype.kind in {"i", "u", "f"},
            f"{name} must be real numeric",
        )
    result = np.asarray(raw, dtype=target).copy()
    if finite:
        _require(np.all(np.isfinite(result)), f"{name} must be finite")
    result.setflags(write=False)
    return result


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _positive_definite_precision(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1],
        f"{name} must be square",
    )
    _require(np.all(np.isfinite(matrix)), f"{name} must be finite")
    _require(
        np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12),
        f"{name} must be symmetric",
    )
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    identity = np.eye(len(matrix), dtype=np.float64)
    inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
    return _symmetric(inverse)


def _covariance_from_precision(value: np.ndarray, *, name: str) -> np.ndarray:
    precision = np.asarray(value, dtype=np.float64)
    _require(
        precision.ndim == 2 and precision.shape[0] == precision.shape[1],
        f"{name} must be square",
    )
    _require(np.all(np.isfinite(precision)), f"{name} must be finite")
    _require(
        np.allclose(precision, precision.T, atol=1e-12, rtol=1e-12),
        f"{name} must be symmetric",
    )
    try:
        factor = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    identity = np.eye(len(precision), dtype=np.float64)
    covariance = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
    return _symmetric(covariance)


def _positive_semidefinite_root(
    value: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1],
        f"{name} must be square",
    )
    _require(np.all(np.isfinite(matrix)), f"{name} must be finite")
    _require(
        np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12),
        f"{name} must be symmetric",
    )
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(matrix))
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    tolerance = len(matrix) * np.finfo(np.float64).eps * scale
    _require(
        float(np.min(eigenvalues, initial=0.0)) >= -100.0 * tolerance,
        f"{name} must be positive semidefinite",
    )
    active = eigenvalues > tolerance
    if not np.any(active):
        return np.zeros((len(matrix), 0), dtype=np.float64)
    return eigenvectors[:, active] * np.sqrt(eigenvalues[active])


def _validate_graph_basis(value: object) -> np.ndarray:
    basis = _readonly(value, name="graph_basis")
    _require(basis.ndim == 2, "graph_basis must be a matrix")
    node_count, rank = basis.shape
    _require(node_count >= 1, "graph_basis must contain at least one node")
    _require(1 <= rank <= node_count, "graph_basis rank is invalid")
    _require(
        np.allclose(
            basis.T @ basis,
            np.eye(rank, dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        ),
        "graph_basis columns must be orthonormal",
    )
    return basis


def _state_vector(value: np.ndarray) -> np.ndarray:
    return np.concatenate((value[0].reshape(-1), value[1].reshape(-1)))


def _state_array(value: np.ndarray, rank: int) -> np.ndarray:
    position_size = 3 * rank
    result = np.empty((2, rank, 3), dtype=np.float64)
    result[0] = value[:position_size].reshape(rank, 3)
    result[1] = value[position_size:].reshape(rank, 3)
    return result


def _transition_and_noise(
    rank: int,
    *,
    frame_dt_s: float,
    velocity_retention: float,
    process_position_std_m: float,
    process_acceleration_std_mps2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate_count = 3 * rank
    identity = np.eye(coordinate_count, dtype=np.float64)
    transition = np.block(
        [
            [identity, frame_dt_s * identity],
            [np.zeros_like(identity), velocity_retention * identity],
        ]
    )
    control = np.concatenate(
        (
            0.5 * frame_dt_s**2 * identity,
            frame_dt_s * identity,
        ),
        axis=0,
    )
    acceleration_variance = process_acceleration_std_mps2**2
    position_variance = process_position_std_m**2
    process_noise = acceleration_variance * np.block(
        [
            [
                0.25 * frame_dt_s**4 * identity,
                0.5 * frame_dt_s**3 * identity,
            ],
            [
                0.5 * frame_dt_s**3 * identity,
                frame_dt_s**2 * identity,
            ],
        ]
    )
    process_noise[:coordinate_count, :coordinate_count] += (
        position_variance * identity
    )
    return transition, _symmetric(process_noise), control


@dataclass(frozen=True, slots=True)
class GraphDynamicDiscrepancyConfigV1:
    """Priors, dynamics, robust likelihood, and admission controls."""

    initial_position_std_m: float = 0.020
    initial_velocity_std_mps: float = 0.050
    process_position_std_m: float = 0.001
    process_acceleration_std_mps2: float = 0.100
    velocity_retention: float = 0.95
    observation_std_m: float = 0.005
    degrees_of_freedom: float = 5.0
    effective_samples_per_correlation_group: float = 64.0
    minimum_robust_weight: float = 0.0
    maximum_iterations: int = 20
    convergence_tolerance: float = 1e-8
    maximum_condition_number: float = 1e12
    maximum_node_position_m: float = 0.10
    maximum_node_velocity_mps: float = 1.0

    def __post_init__(self) -> None:
        positive_names = (
            "initial_position_std_m",
            "observation_std_m",
            "effective_samples_per_correlation_group",
            "convergence_tolerance",
            "maximum_condition_number",
            "maximum_node_position_m",
            "maximum_node_velocity_mps",
        )
        for name in positive_names:
            value = _real(getattr(self, name), name=name)
            _require(value > 0.0, f"{name} must be positive")
            object.__setattr__(self, name, value)
        nonnegative_names = (
            "initial_velocity_std_mps",
            "process_position_std_m",
            "process_acceleration_std_mps2",
            "minimum_robust_weight",
        )
        for name in nonnegative_names:
            value = _real(getattr(self, name), name=name)
            _require(value >= 0.0, f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        retention = _real(self.velocity_retention, name="velocity_retention")
        _require(
            0.0 <= retention <= 1.0,
            "velocity_retention must lie in [0, 1]",
        )
        object.__setattr__(self, "velocity_retention", retention)
        degrees = _real(self.degrees_of_freedom, name="degrees_of_freedom")
        _require(
            degrees > 2.0,
            "degrees_of_freedom must exceed two when inputs are covariances",
        )
        object.__setattr__(self, "degrees_of_freedom", degrees)
        _require(
            self.minimum_robust_weight <= 1.0,
            "minimum_robust_weight must not exceed one",
        )
        _require(
            self.maximum_condition_number >= 1.0,
            "maximum_condition_number must be at least one",
        )
        iterations = _integer(
            self.maximum_iterations,
            name="maximum_iterations",
            minimum=1,
        )
        object.__setattr__(self, "maximum_iterations", iterations)


__all__ = [
    "GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY",
    "GRAPH_DYNAMIC_DISCREPANCY_SCHEMA",
    "GRAPH_DYNAMIC_DISCREPANCY_VERSION",
    "GraphDynamicDiscrepancyConfigV1",
]
