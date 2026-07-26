"""Contracts and numerical helpers for gauge-aware Bayesian updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(value: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _finite_array(value: np.ndarray, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _positive_definite_whitener(value: np.ndarray, name: str) -> np.ndarray:
    matrix = _finite_array(value, name, 2)
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(np.allclose(matrix, matrix.T, atol=1e-12), f"{name} must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(matrix))
    _require(np.all(eigenvalues > 0.0), f"{name} must be positive definite")
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def _regularized_precision(
    value: np.ndarray, name: str, *, eigenvalue_floor: float
) -> np.ndarray:
    matrix = _finite_array(value, name, 2)
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(np.allclose(matrix, matrix.T, atol=1e-12), f"{name} must be symmetric")
    if not len(matrix):
        return matrix.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(matrix))
    _require(
        np.all(eigenvalues >= -eigenvalue_floor),
        f"{name} must be positive semidefinite",
    )
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    return (eigenvectors * (1.0 / clipped)) @ eigenvectors.T


def _positive_semidefinite_square_root(
    value: np.ndarray, name: str, *, eigenvalue_floor: float
) -> np.ndarray:
    matrix = _finite_array(value, name, 2)
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(np.allclose(matrix, matrix.T, atol=1e-12), f"{name} must be symmetric")
    if not len(matrix):
        return matrix.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(matrix))
    _require(
        np.all(eigenvalues >= -eigenvalue_floor),
        f"{name} must be positive semidefinite",
    )
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


def _block_diagonal(values: list[np.ndarray]) -> np.ndarray:
    dimension = sum(value.shape[0] for value in values)
    result = np.zeros((dimension, dimension), dtype=np.float64)
    offset = 0
    for value in values:
        width = value.shape[0]
        result[offset : offset + width, offset : offset + width] = value
        offset += width
    return result


def _orthonormal_column_space(value: np.ndarray) -> np.ndarray:
    if value.shape[1] == 0:
        return np.zeros((len(value), 0), dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(value, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.zeros((len(value), 0), dtype=np.float64)
    tolerance = max(value.shape) * np.finfo(np.float64).eps * singular_values[0]
    return left[:, singular_values > tolerance]


def _subspace_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first_basis = _orthonormal_column_space(first)
    second_basis = _orthonormal_column_space(second)
    if first_basis.shape[1] == 0 or second_basis.shape[1] == 0:
        return 0.0
    return float(np.linalg.svd(first_basis.T @ second_basis, compute_uv=False)[0])


def _student_t_weights(
    squared_mahalanobis: np.ndarray,
    *,
    dimension: int,
    degrees_of_freedom: float,
    minimum: float,
) -> np.ndarray:
    weights = (degrees_of_freedom + dimension) / (
        degrees_of_freedom + squared_mahalanobis
    )
    return np.clip(weights, minimum, 1.0)


@dataclass(frozen=True)
class GaugeAwareBeliefConfig:
    """Prior, robustness, identifiability, and plausibility settings."""

    state_prior_std_m: float = 0.020
    shared_bias_prior_std_m: float = 0.020
    view_bias_prior_std_m: float = 0.010
    effective_samples_per_correlation_group: float = 64.0
    degrees_of_freedom: float = 4.0
    minimum_robust_weight: float = 0.02
    maximum_iterations: int = 12
    convergence_tolerance: float = 1e-9
    maximum_condition_number: float = 1e14
    minimum_identifiable_fraction: float = 0.10
    minimum_query_sensitivity_fraction: float = 1e-3
    maximum_state_update_m: float = 0.10
    maximum_update_to_physical_response_ratio: float = 2.0
    prior_eigenvalue_floor: float = 1e-12

    def __post_init__(self) -> None:
        positive = (
            self.state_prior_std_m,
            self.shared_bias_prior_std_m,
            self.view_bias_prior_std_m,
            self.effective_samples_per_correlation_group,
            self.degrees_of_freedom,
            self.convergence_tolerance,
            self.maximum_condition_number,
            self.maximum_state_update_m,
            self.maximum_update_to_physical_response_ratio,
            self.prior_eigenvalue_floor,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "gauge-aware configuration scales must be positive",
        )
        _require(
            0.0 < self.minimum_robust_weight <= 1.0,
            "minimum robust weight must lie in (0, 1]",
        )
        _require(self.maximum_iterations >= 1, "maximum_iterations must be positive")
        _require(
            0.0 < self.minimum_identifiable_fraction <= 1.0,
            "minimum identifiable fraction must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_query_sensitivity_fraction <= 1.0,
            "minimum query sensitivity fraction must lie in [0, 1]",
        )


@dataclass(frozen=True)
class GaugeAwareObservationBatch:
    """Linearized unfused factors and optional nuisance-free anchor evidence."""

    innovation_m: np.ndarray
    observation_covariance_m2: np.ndarray
    state_jacobian: np.ndarray
    gauge_jacobian: np.ndarray
    shared_bias_jacobian: np.ndarray
    view_bias_jacobian: np.ndarray
    query_state_jacobian: np.ndarray
    gauge_prior_covariance: np.ndarray
    correlation_group_ids: tuple[str, ...]
    prior_reliability: np.ndarray
    physical_response_scale_m: float
    state_prior_covariance_m2: np.ndarray | None = None
    anchor_innovation_m: np.ndarray | None = None
    anchor_covariance_m2: np.ndarray | None = None
    anchor_state_jacobian: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        innovation = _finite_array(self.innovation_m, "innovation_m", 2)
        covariance = _finite_array(
            self.observation_covariance_m2,
            "observation_covariance_m2",
            3,
        )
        state = _finite_array(self.state_jacobian, "state_jacobian", 3)
        gauge = _finite_array(self.gauge_jacobian, "gauge_jacobian", 3)
        shared = _finite_array(
            self.shared_bias_jacobian, "shared_bias_jacobian", 3
        )
        view = _finite_array(self.view_bias_jacobian, "view_bias_jacobian", 3)
        query = _finite_array(
            self.query_state_jacobian, "query_state_jacobian", 3
        )
        count = len(innovation)
        _require(innovation.shape == (count, 3), "innovation_m must have shape (M, 3)")
        _require(
            covariance.shape == (count, 3, 3),
            "observation_covariance_m2 must have shape (M, 3, 3)",
        )
        _require(
            state.shape[:2] == (count, 3) and state.shape[2] >= 1,
            "state_jacobian must have shape (M, 3, S) with S >= 1",
        )
        state_count = state.shape[2]
        for name, value in (
            ("gauge_jacobian", gauge),
            ("shared_bias_jacobian", shared),
            ("view_bias_jacobian", view),
        ):
            _require(value.shape[:2] == (count, 3), f"{name} row shape changed")
        _require(
            query.ndim == 3 and query.shape[1:] == (3, state_count) and len(query),
            "query_state_jacobian must have shape (Q, 3, S)",
        )
        gauge_count = gauge.shape[2]
        gauge_prior = _finite_array(
            self.gauge_prior_covariance, "gauge_prior_covariance", 2
        )
        _require(
            gauge_prior.shape == (gauge_count, gauge_count),
            "gauge prior covariance has changed shape",
        )
        _regularized_precision(
            gauge_prior,
            "gauge prior covariance",
            eigenvalue_floor=1e-12,
        )
        for index, matrix in enumerate(covariance):
            _positive_definite_whitener(matrix, f"observation covariance {index}")

        groups = tuple(map(str, self.correlation_group_ids))
        _require(len(groups) == count, "correlation_group_ids length changed")
        _require(all(groups), "correlation group IDs must not be empty")
        reliability = np.asarray(self.prior_reliability, dtype=np.float64)
        _require(reliability.shape == (count,), "prior_reliability must have shape (M,)")
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior_reliability must lie in [0, 1]",
        )
        _require(
            np.isfinite(self.physical_response_scale_m)
            and self.physical_response_scale_m > 0.0,
            "physical_response_scale_m must be positive",
        )

        state_prior = None
        if self.state_prior_covariance_m2 is not None:
            state_prior = _finite_array(
                self.state_prior_covariance_m2,
                "state_prior_covariance_m2",
                2,
            )
            _require(
                state_prior.shape == (state_count, state_count),
                "state prior covariance has changed shape",
            )
            _regularized_precision(
                state_prior,
                "state prior covariance",
                eigenvalue_floor=1e-12,
            )

        anchor_values = (
            self.anchor_innovation_m,
            self.anchor_covariance_m2,
            self.anchor_state_jacobian,
        )
        has_anchor = any(value is not None for value in anchor_values)
        _require(
            not has_anchor or all(value is not None for value in anchor_values),
            "all anchor arrays must be supplied together",
        )
        anchor_innovation = None
        anchor_covariance = None
        anchor_state = None
        if has_anchor:
            anchor_innovation = _finite_array(
                self.anchor_innovation_m, "anchor_innovation_m", 2
            )
            anchor_covariance = _finite_array(
                self.anchor_covariance_m2, "anchor_covariance_m2", 3
            )
            anchor_state = _finite_array(
                self.anchor_state_jacobian, "anchor_state_jacobian", 3
            )
            anchor_count = len(anchor_innovation)
            _require(
                anchor_innovation.shape == (anchor_count, 3),
                "anchor_innovation_m must have shape (A, 3)",
            )
            _require(
                anchor_covariance.shape == (anchor_count, 3, 3),
                "anchor_covariance_m2 must have shape (A, 3, 3)",
            )
            _require(
                anchor_state.shape == (anchor_count, 3, state_count),
                "anchor_state_jacobian must have shape (A, 3, S)",
            )
            for index, matrix in enumerate(anchor_covariance):
                _positive_definite_whitener(matrix, f"anchor covariance {index}")

        for name, value in (
            ("innovation_m", innovation),
            ("observation_covariance_m2", covariance),
            ("state_jacobian", state),
            ("gauge_jacobian", gauge),
            ("shared_bias_jacobian", shared),
            ("view_bias_jacobian", view),
            ("query_state_jacobian", query),
            ("gauge_prior_covariance", gauge_prior),
            ("prior_reliability", reliability),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "correlation_group_ids", groups)
        object.__setattr__(self, "physical_response_scale_m", float(self.physical_response_scale_m))
        object.__setattr__(
            self,
            "state_prior_covariance_m2",
            None if state_prior is None else _readonly(state_prior),
        )
        object.__setattr__(
            self,
            "anchor_innovation_m",
            None if anchor_innovation is None else _readonly(anchor_innovation),
        )
        object.__setattr__(
            self,
            "anchor_covariance_m2",
            None if anchor_covariance is None else _readonly(anchor_covariance),
        )
        object.__setattr__(
            self,
            "anchor_state_jacobian",
            None if anchor_state is None else _readonly(anchor_state),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class GaugeAwareBeliefResult:
    """Posterior coefficients and diagnostics with exact-zero rejection state."""

    accepted: bool
    reason: str
    state_coefficients: np.ndarray
    gauge_delta: np.ndarray
    shared_bias_coefficients: np.ndarray
    view_bias_coefficients: np.ndarray
    posterior_covariance: np.ndarray
    identifiable_state_transform: np.ndarray
    identifiable_fractions: np.ndarray
    query_sensitivity_fractions: np.ndarray
    robust_weights: np.ndarray
    anchor_robust_weights: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        state = np.asarray(self.state_coefficients, dtype=np.float64)
        gauge = np.asarray(self.gauge_delta, dtype=np.float64)
        shared = np.asarray(self.shared_bias_coefficients, dtype=np.float64)
        view = np.asarray(self.view_bias_coefficients, dtype=np.float64)
        covariance = np.asarray(self.posterior_covariance, dtype=np.float64)
        transform = np.asarray(self.identifiable_state_transform, dtype=np.float64)
        fractions = np.asarray(self.identifiable_fractions, dtype=np.float64)
        query_fractions = np.asarray(
            self.query_sensitivity_fractions, dtype=np.float64
        )
        robust = np.asarray(self.robust_weights, dtype=np.float64)
        anchor_robust = np.asarray(self.anchor_robust_weights, dtype=np.float64)
        _require(
            state.ndim == gauge.ndim == shared.ndim == view.ndim == 1,
            "coefficient arrays must be vectors",
        )
        dimension = len(state) + len(gauge) + len(shared) + len(view)
        _require(
            covariance.shape == (dimension, dimension),
            "posterior covariance has changed shape",
        )
        _require(
            transform.ndim == 2 and transform.shape[0] == len(state),
            "identifiable state transform has changed shape",
        )
        _require(
            fractions.shape == query_fractions.shape == (transform.shape[1],),
            "identifiability diagnostics have changed shape",
        )
        for value in (
            state,
            gauge,
            shared,
            view,
            covariance,
            transform,
            fractions,
            query_fractions,
            robust,
            anchor_robust,
        ):
            _require(np.all(np.isfinite(value)), "gauge-aware result is non-finite")
        for name, value in (
            ("state_coefficients", state),
            ("gauge_delta", gauge),
            ("shared_bias_coefficients", shared),
            ("view_bias_coefficients", view),
            ("posterior_covariance", covariance),
            ("identifiable_state_transform", transform),
            ("identifiable_fractions", fractions),
            ("query_sensitivity_fractions", query_fractions),
            ("robust_weights", robust),
            ("anchor_robust_weights", anchor_robust),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class GaugeAwareSelection:
    """Candidate selection with byte-exact baseline fallback."""

    candidate_accepted: bool
    reason: str
    selected_value: np.ndarray

    def __post_init__(self) -> None:
        selected = np.asarray(self.selected_value).copy()
        selected.setflags(write=False)
        object.__setattr__(self, "selected_value", selected)


def _fallback_result(
    batch: GaugeAwareObservationBatch,
    reason: str,
    diagnostics: Mapping[str, Any],
) -> GaugeAwareBeliefResult:
    state_count = batch.state_jacobian.shape[2]
    gauge_count = batch.gauge_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    dimension = state_count + gauge_count + shared_count + view_count
    anchor_count = 0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
    return GaugeAwareBeliefResult(
        accepted=False,
        reason=reason,
        state_coefficients=np.zeros(state_count),
        gauge_delta=np.zeros(gauge_count),
        shared_bias_coefficients=np.zeros(shared_count),
        view_bias_coefficients=np.zeros(view_count),
        posterior_covariance=np.zeros((dimension, dimension)),
        identifiable_state_transform=np.zeros((state_count, 0)),
        identifiable_fractions=np.zeros(0),
        query_sensitivity_fractions=np.zeros(0),
        robust_weights=np.zeros(len(batch.innovation_m)),
        anchor_robust_weights=np.zeros(anchor_count),
        diagnostics=diagnostics,
    )
