"""Numerical helpers for prior-aware gauge and bias inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
    _block_diagonal,
    _positive_definite_whitener,
    _positive_semidefinite_square_root,
    _regularized_precision,
    _require,
)


@dataclass(frozen=True)
class PriorAwareGaugeConfigV1:
    """Prior-aware identifiability, robust mixture, and trust-region settings."""

    state_prior_std_m: float = 0.020
    shared_bias_prior_std_m: float = 0.020
    view_bias_prior_std_m: float = 0.010
    effective_samples_per_correlation_group: float = 64.0
    effective_samples_per_anchor_correlation_group: float = 16.0
    degrees_of_freedom: float = 5.0
    outlier_covariance_multiplier: float = 25.0
    probability_floor: float = 1e-6
    minimum_robust_precision: float = 0.0
    maximum_iterations: int = 12
    convergence_tolerance: float = 1e-9
    maximum_condition_number: float = 1e14
    minimum_conditional_information_fraction: float = 1e-4
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
            self.effective_samples_per_anchor_correlation_group,
            self.outlier_covariance_multiplier,
            self.convergence_tolerance,
            self.maximum_condition_number,
            self.maximum_state_update_m,
            self.maximum_update_to_physical_response_ratio,
            self.prior_eigenvalue_floor,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "prior-aware configuration scales must be positive",
        )
        _require(
            np.isfinite(self.degrees_of_freedom) and self.degrees_of_freedom > 2.0,
            "degrees_of_freedom must exceed two when inputs are covariances",
        )
        _require(
            self.outlier_covariance_multiplier > 1.0,
            "outlier_covariance_multiplier must exceed one",
        )
        _require(
            0.0 < self.probability_floor < 0.5,
            "probability_floor must lie in (0, 0.5)",
        )
        _require(
            np.isfinite(self.minimum_robust_precision)
            and self.minimum_robust_precision >= 0.0,
            "minimum_robust_precision must be finite and nonnegative",
        )
        raw_maximum_iterations = np.asarray(self.maximum_iterations)
        _require(
            all(
                (
                    raw_maximum_iterations.ndim == 0,
                    np.issubdtype(raw_maximum_iterations.dtype, np.integer),
                    raw_maximum_iterations.dtype.kind != "b",
                )
            ),
            "maximum_iterations must be an integer",
        )
        maximum_iterations = int(raw_maximum_iterations.item())
        _require(maximum_iterations >= 1, "maximum_iterations must be positive")
        object.__setattr__(self, "maximum_iterations", maximum_iterations)
        _require(
            0.0 <= self.minimum_conditional_information_fraction <= 1.0,
            "minimum_conditional_information_fraction must lie in [0, 1]",
        )
        _require(
            0.0 < self.minimum_identifiable_fraction <= 1.0,
            "minimum_identifiable_fraction must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_query_sensitivity_fraction <= 1.0,
            "minimum_query_sensitivity_fraction must lie in [0, 1]",
        )


@dataclass(frozen=True)
class _StudentTMixtureStatistics:
    """Density, responsibility, score, and curvature for one residual group."""

    log_mixture_density: float
    posterior_nominal_probability: float
    expected_precision: float
    expected_precision_derivative: float
    unfloored_expected_precision: float
    unfloored_expected_precision_derivative: float
    nominal_precision: float
    outlier_precision: float
    log_nominal_density: float
    log_outlier_density: float
    effective_prior_nominal_probability: float
    precision_floor_active: bool


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _whiten(
    target: np.ndarray,
    covariance: np.ndarray,
    designs: tuple[np.ndarray, ...],
    *,
    name: str,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    white_target = np.empty_like(target, dtype=np.float64)
    white_designs = tuple(np.empty_like(value) for value in designs)
    whiteners = np.empty_like(covariance, dtype=np.float64)
    for index, matrix in enumerate(covariance):
        whitener = _positive_definite_whitener(matrix, f"{name} covariance {index}")
        whiteners[index] = whitener
        white_target[index] = whitener @ target[index]
        for output, design in zip(white_designs, designs, strict=True):
            output[index] = whitener @ design[index]
    return white_target, white_designs, whiteners


def _group_layout(
    labels: tuple[str, ...],
    reliability: np.ndarray,
    nominal: np.ndarray,
    composite: np.ndarray,
    cap: float,
    *,
    composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
) -> tuple[
    tuple[str, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return row information weights and one generalized-Bayes power per group."""

    ordered = tuple(dict.fromkeys(map(str, labels)))
    indices = tuple(
        np.flatnonzero(np.asarray(labels, dtype=object) == group) for group in ordered
    )
    base: np.ndarray = np.zeros(len(labels), dtype=np.float64)
    prior: np.ndarray = np.empty(len(ordered), dtype=np.float64)
    group_power: np.ndarray = np.zeros(len(ordered), dtype=np.float64)
    for position, selected in enumerate(indices):
        _require(
            np.allclose(nominal[selected], nominal[selected[0]], atol=1e-12),
            "prior nominal probability must be constant within a group",
        )
        _require(
            np.allclose(composite[selected], composite[selected[0]], atol=1e-12),
            "composite weight must be constant within a group",
        )
        prior[position] = float(nominal[selected[0]])
        active = selected[reliability[selected] > 0.0]
        if len(active):
            consumer_scale = (
                1.0
                if composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
                else min(cap, float(len(active))) / len(active)
            )
            group_power[position] = float(composite[selected[0]]) * consumer_scale
            base[active] = reliability[active] * group_power[position]
    return ordered, indices, base, prior, group_power


def _student_t_mixture_statistics(
    squared_mahalanobis: float,
    dimension: int,
    prior_nominal: float,
    config: PriorAwareGaugeConfigV1,
) -> _StudentTMixtureStatistics:
    """Return the normalized mixture density and exact IRLS score precision."""

    _require(
        np.isfinite(squared_mahalanobis) and squared_mahalanobis >= 0.0,
        "squared Mahalanobis distance must be finite and nonnegative",
    )
    _require(dimension >= 1, "mixture dimension must be positive")
    _require(
        np.isfinite(prior_nominal),
        "prior nominal probability must be finite",
    )
    rho = float(
        np.clip(
            prior_nominal,
            config.probability_floor,
            1.0 - config.probability_floor,
        )
    )
    covariance_to_scale = (config.degrees_of_freedom - 2.0) / config.degrees_of_freedom
    common_log_normalizer = (
        math.lgamma(0.5 * (config.degrees_of_freedom + dimension))
        - math.lgamma(0.5 * config.degrees_of_freedom)
        - 0.5 * dimension * math.log(config.degrees_of_freedom * math.pi)
    )

    def component(multiplier: float) -> tuple[float, float]:
        scale = covariance_to_scale * multiplier
        log_density = (
            common_log_normalizer
            - 0.5 * dimension * math.log(scale)
            - 0.5
            * (config.degrees_of_freedom + dimension)
            * math.log1p(squared_mahalanobis / (config.degrees_of_freedom * scale))
        )
        precision = (config.degrees_of_freedom + dimension) / (
            config.degrees_of_freedom * scale + squared_mahalanobis
        )
        return log_density, precision

    log_nominal, precision_nominal = component(1.0)
    log_outlier, precision_outlier = component(config.outlier_covariance_multiplier)
    weighted_nominal = math.log(rho) + log_nominal
    weighted_outlier = math.log1p(-rho) + log_outlier
    log_mixture = float(np.logaddexp(weighted_nominal, weighted_outlier))
    responsibility = math.exp(weighted_nominal - log_mixture)
    unfloored_precision = (
        responsibility * precision_nominal + (1.0 - responsibility) * precision_outlier
    )
    precision_difference = precision_nominal - precision_outlier
    responsibility_derivative = (
        -0.5 * responsibility * (1.0 - responsibility) * precision_difference
    )
    unfloored_precision_derivative = (
        responsibility_derivative * precision_difference
        - (
            responsibility * precision_nominal**2
            + (1.0 - responsibility) * precision_outlier**2
        )
        / (config.degrees_of_freedom + dimension)
    )
    precision_floor_active = config.minimum_robust_precision > unfloored_precision
    precision = (
        config.minimum_robust_precision
        if precision_floor_active
        else unfloored_precision
    )
    precision_derivative = (
        0.0 if precision_floor_active else unfloored_precision_derivative
    )
    return _StudentTMixtureStatistics(
        log_mixture_density=log_mixture,
        posterior_nominal_probability=responsibility,
        expected_precision=precision,
        expected_precision_derivative=precision_derivative,
        unfloored_expected_precision=unfloored_precision,
        unfloored_expected_precision_derivative=(unfloored_precision_derivative),
        nominal_precision=precision_nominal,
        outlier_precision=precision_outlier,
        log_nominal_density=log_nominal,
        log_outlier_density=log_outlier,
        effective_prior_nominal_probability=rho,
        precision_floor_active=precision_floor_active,
    )


def _mixture_precision(
    mahalanobis: float,
    dimension: int,
    prior_nominal: float,
    config: PriorAwareGaugeConfigV1,
) -> tuple[float, float]:
    statistics = _student_t_mixture_statistics(
        mahalanobis,
        dimension,
        prior_nominal,
        config,
    )
    return (
        statistics.expected_precision,
        statistics.posterior_nominal_probability,
    )


def _cholesky_factor(normal: np.ndarray) -> np.ndarray:
    matrix = np.asarray(normal, dtype=np.float64)
    _require(matrix.ndim == 2, "normal matrix must have two dimensions")
    _require(
        matrix.shape[0] == matrix.shape[1],
        "normal matrix must be square",
    )
    _require(np.all(np.isfinite(matrix)), "normal matrix must be finite")
    _require(
        np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10),
        "normal matrix must be symmetric",
    )
    return np.linalg.cholesky(_symmetric(matrix))


def _solve_spd_system(normal: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Solve a symmetric positive-definite normal system by Cholesky."""

    factor = _cholesky_factor(normal)
    vector = np.asarray(right, dtype=np.float64)
    _require(vector.shape == (len(factor),), "right-hand side shape changed")
    _require(np.all(np.isfinite(vector)), "right-hand side must be finite")
    intermediate = np.linalg.solve(factor, vector)
    return np.linalg.solve(factor.T, intermediate)


def _spd_covariance(normal: np.ndarray) -> np.ndarray:
    """Invert an SPD normal matrix with triangular solves against identity."""

    factor = _cholesky_factor(normal)
    inverse_factor = np.linalg.solve(
        factor,
        np.eye(len(factor), dtype=np.float64),
    )
    return _symmetric(inverse_factor.T @ inverse_factor)


def _solve_spd_posterior(
    normal: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a normal system and form its symmetric covariance without inv()."""

    return _solve_spd_system(normal, right), _spd_covariance(normal)


def _prior_covariances(
    batch: GaugeAwareObservationBatch,
    config: PriorAwareGaugeConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_count = batch.state_jacobian.shape[2]
    state = (
        np.eye(state_count) * config.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2)
    )
    nuisance = [np.asarray(batch.gauge_prior_covariance)]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    if shared_count:
        nuisance.append(np.eye(shared_count) * config.shared_bias_prior_std_m**2)
    if view_count:
        nuisance.append(np.eye(view_count) * config.view_bias_prior_std_m**2)
    if batch.anchor_bias_prior_covariance is not None:
        nuisance.append(np.asarray(batch.anchor_bias_prior_covariance))
    nuisance_covariance = _block_diagonal(nuisance)
    return state, nuisance_covariance, _block_diagonal([state, nuisance_covariance])


def _prior_aware_basis(
    state_design: np.ndarray,
    nuisance_design: np.ndarray,
    anchor_state: np.ndarray,
    anchor_nuisance: np.ndarray,
    state_prior: np.ndarray,
    nuisance_prior: np.ndarray,
    observation_weight: np.ndarray,
    anchor_weight: np.ndarray,
    query: np.ndarray,
    config: PriorAwareGaugeConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    state_sqrt = _positive_semidefinite_square_root(
        state_prior,
        "state prior covariance",
        eigenvalue_floor=config.prior_eigenvalue_floor,
    )
    hx = np.concatenate(
        (
            np.sqrt(observation_weight)[:, None, None] * state_design,
            np.sqrt(anchor_weight)[:, None, None] * anchor_state,
        )
    ).reshape(-1, state_design.shape[2])
    nuisance_count = nuisance_design.shape[2]
    hn = (
        np.zeros((hx.shape[0], 0), dtype=np.float64)
        if nuisance_count == 0
        else np.concatenate(
            (
                np.sqrt(observation_weight)[:, None, None] * nuisance_design,
                np.sqrt(anchor_weight)[:, None, None] * anchor_nuisance,
            )
        ).reshape(-1, nuisance_count)
    )
    known = hx.T @ hx
    if hn.shape[1]:
        cross = hx.T @ hn
        nuisance_information = (
            _regularized_precision(
                nuisance_prior,
                "nuisance prior covariance",
                eigenvalue_floor=config.prior_eigenvalue_floor,
            )
            + hn.T @ hn
        )
        conditional = known - cross @ np.linalg.solve(nuisance_information, cross.T)
    else:
        conditional = known
    standardized = _symmetric(state_sqrt.T @ conditional @ state_sqrt)
    eigenvalues, eigenvectors = np.linalg.eigh(standardized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum_information = float(np.max(eigenvalues, initial=0.0))
    query_flat = query.reshape(-1, query.shape[2])
    candidates: list[tuple[np.ndarray, float, float, float]] = []
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= config.prior_eigenvalue_floor:
            continue
        direction = state_sqrt @ eigenvectors[:, index]
        known_value = float(direction @ known @ direction)
        conditional_value = float(direction @ conditional @ direction)
        identifiable = conditional_value / max(
            known_value, config.prior_eigenvalue_floor
        )
        candidates.append(
            (
                direction,
                float(np.linalg.norm(query_flat @ direction)),
                float(eigenvalue),
                identifiable,
            )
        )
    maximum_query = max((item[1] for item in candidates), default=0.0)
    retained: list[np.ndarray] = []
    identifiable_fractions: list[float] = []
    query_fractions: list[float] = []
    for direction, query_norm, information, identifiable in candidates:
        information_fraction = information / max(
            maximum_information, config.prior_eigenvalue_floor
        )
        query_fraction = query_norm / maximum_query if maximum_query else 0.0
        if (
            information_fraction >= config.minimum_conditional_information_fraction
            and identifiable >= config.minimum_identifiable_fraction
            and query_fraction >= config.minimum_query_sensitivity_fraction
        ):
            retained.append(direction)
            identifiable_fractions.append(min(1.0, max(0.0, identifiable)))
            query_fractions.append(min(1.0, max(0.0, query_fraction)))
    mapping = (
        np.column_stack(retained) if retained else np.zeros((state_design.shape[2], 0))
    )
    return (
        mapping,
        np.asarray(identifiable_fractions),
        np.asarray(query_fractions),
        {
            "maximum_conditional_information_eigenvalue": maximum_information,
            "maximum_query_sensitivity_norm": maximum_query,
        },
    )


def _full_covariance(
    state_prior: np.ndarray,
    state_mapping: np.ndarray,
    reduced: np.ndarray,
    nuisance_count: int,
) -> np.ndarray:
    retained = state_mapping.shape[1]
    state_count = len(state_prior)
    result = np.zeros((state_count + nuisance_count,) * 2)
    result[:state_count, :state_count] = state_prior
    result[:state_count, :state_count] += (
        state_mapping
        @ (reduced[:retained, :retained] - np.eye(retained))
        @ state_mapping.T
    )
    if nuisance_count:
        cross = state_mapping @ reduced[:retained, retained:]
        result[:state_count, state_count:] = cross
        result[state_count:, :state_count] = cross.T
        result[state_count:, state_count:] = reduced[retained:, retained:]
    return _symmetric(result)


__all__ = [
    "PriorAwareGaugeConfigV1",
    "_StudentTMixtureStatistics",
    "_full_covariance",
    "_group_layout",
    "_mixture_precision",
    "_prior_aware_basis",
    "_prior_covariances",
    "_solve_spd_posterior",
    "_solve_spd_system",
    "_spd_covariance",
    "_student_t_mixture_statistics",
    "_whiten",
]
