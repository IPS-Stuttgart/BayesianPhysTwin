"""Internal SPD-safe robust linear updates for the directional endpoint v2."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .spd_system import SPDSolveError, SPDSystem, SPDSystemError


class DirectionalEndpointNumericalError(RuntimeError):
    """Raised when v2 cannot admit an update without numerical regularization."""


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def immutable_float64(value: object) -> np.ndarray:
    """Return a C-contiguous float64 array backed by immutable bytes."""

    array = np.ascontiguousarray(value, dtype=np.float64)
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64).reshape(array.shape)


def immutable_int64(value: object) -> np.ndarray:
    """Return a C-contiguous int64 array backed by immutable bytes."""

    array = np.ascontiguousarray(value, dtype=np.int64)
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=np.int64).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class DirectionalEndpointConfigV2:
    """Numerical admission policy for the prospective directional endpoint."""

    maximum_condition_number: float = 1e12
    symmetry_absolute_tolerance: float = 1e-12
    symmetry_relative_tolerance: float = 1e-10
    solve_residual_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        maximum_condition_number = _finite_positive(
            self.maximum_condition_number,
            name="maximum_condition_number",
        )
        if maximum_condition_number < 1.0:
            raise ValueError("maximum_condition_number must be at least one")
        symmetry_absolute_tolerance = _finite_nonnegative(
            self.symmetry_absolute_tolerance,
            name="symmetry_absolute_tolerance",
        )
        symmetry_relative_tolerance = _finite_nonnegative(
            self.symmetry_relative_tolerance,
            name="symmetry_relative_tolerance",
        )
        solve_residual_tolerance = _finite_positive(
            self.solve_residual_tolerance,
            name="solve_residual_tolerance",
        )
        object.__setattr__(
            self,
            "maximum_condition_number",
            maximum_condition_number,
        )
        object.__setattr__(
            self,
            "symmetry_absolute_tolerance",
            symmetry_absolute_tolerance,
        )
        object.__setattr__(
            self,
            "symmetry_relative_tolerance",
            symmetry_relative_tolerance,
        )
        object.__setattr__(
            self,
            "solve_residual_tolerance",
            solve_residual_tolerance,
        )


def validate_filter_parameters(
    *,
    end_frame: int,
    frame_count: int,
    process_variance: float,
    observation_variance: float,
    initial_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
) -> tuple[float, float, float, float, float]:
    """Validate and normalize the scalar filter parameters."""

    if isinstance(end_frame, (bool, np.bool_)) or not isinstance(
        end_frame, (int, np.integer)
    ):
        raise TypeError("end_frame must be an integer")
    if not 0 < int(end_frame) <= frame_count:
        raise ValueError("end_frame must lie inside the residual sequence")
    process = _finite_nonnegative(process_variance, name="process_variance")
    observation = _finite_positive(
        observation_variance,
        name="observation_variance",
    )
    initial = _finite_positive(initial_variance, name="initial_variance")
    inlier = _finite_positive(inlier_prior, name="inlier_prior")
    if inlier >= 1.0:
        raise ValueError("inlier_prior must lie in (0, 1)")
    outlier = _finite_positive(
        outlier_variance_multiplier,
        name="outlier_variance_multiplier",
    )
    if outlier <= 1.0:
        raise ValueError("outlier_variance_multiplier must exceed one")
    if not np.isfinite(observation * outlier):
        raise ValueError("outlier observation variance must remain finite")
    return process, observation, initial, inlier, outlier


def admit_spd_system(
    value: object,
    *,
    name: str,
    config: DirectionalEndpointConfigV2,
) -> SPDSystem:
    """Validate one SPD system under the declared prospective policy."""

    return SPDSystem.from_matrix(
        value,
        name=name,
        maximum_condition_number=config.maximum_condition_number,
        symmetry_absolute_tolerance=config.symmetry_absolute_tolerance,
        symmetry_relative_tolerance=config.symmetry_relative_tolerance,
        solve_residual_tolerance=config.solve_residual_tolerance,
    )


def _component_update(
    *,
    mean: np.ndarray,
    prior: SPDSystem,
    innovation: np.ndarray,
    projected_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    observation_variance: float,
    name: str,
    config: DirectionalEndpointConfigV2,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    observation_dimension = innovation.shape[0]
    innovation_covariance = projected_covariance + (
        observation_variance * np.eye(observation_dimension, dtype=np.float64)
    )
    innovation_system = admit_spd_system(
        innovation_covariance,
        name=f"{name} innovation covariance",
        config=config,
    )
    prior_covariance = np.asarray(prior.matrix)
    cross_covariance = prior_covariance @ observation_matrix.T
    gain = np.asarray(innovation_system.solve(cross_covariance.T)).T
    updated_mean = mean + gain @ innovation
    if not np.all(np.isfinite(updated_mean)):
        raise SPDSolveError(f"{name} mean update produced non-finite values")

    residual_operator = np.eye(prior.dimension, dtype=np.float64) - (
        gain @ observation_matrix
    )
    updated_covariance = (
        residual_operator @ prior_covariance @ residual_operator.T
        + observation_variance * (gain @ gain.T)
    )
    posterior = admit_spd_system(
        updated_covariance,
        name=f"{name} posterior covariance",
        config=config,
    )
    quadratic = innovation_system.quadratic_form(innovation)
    return (
        np.asarray(updated_mean, dtype=np.float64),
        np.asarray(posterior.matrix),
        quadratic,
        innovation_system.log_determinant,
        innovation_system.condition_number,
        posterior.condition_number,
    )


def robust_linear_update_v2(
    mean: np.ndarray,
    covariance: np.ndarray,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    *,
    observation_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
    name: str,
    config: DirectionalEndpointConfigV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one robust mixture update to an independent batch of states."""

    count = len(mean)
    dimension = observation.shape[1]
    updated_mean = np.empty_like(mean, dtype=np.float64)
    updated_covariance = np.empty_like(covariance, dtype=np.float64)
    probability = np.empty(count, dtype=np.float64)
    innovation_condition = np.empty(count, dtype=np.float64)
    posterior_condition = np.empty(count, dtype=np.float64)
    normalizer = dimension * np.log(2.0 * np.pi)
    outlier_variance = observation_variance * outlier_variance_multiplier
    if not np.isfinite(outlier_variance):
        raise DirectionalEndpointNumericalError(
            "outlier observation variance overflowed"
        )

    for index in range(count):
        row_name = f"{name} row {index}"
        try:
            prior = admit_spd_system(
                covariance[index],
                name=f"{row_name} prior covariance",
                config=config,
            )
            row_matrix = observation_matrix[index]
            innovation = observation[index] - row_matrix @ mean[index]
            if not np.all(np.isfinite(innovation)):
                raise SPDSolveError(
                    f"{row_name} innovation produced non-finite values"
                )
            projected_covariance = (
                row_matrix @ np.asarray(prior.matrix) @ row_matrix.T
            )
            (
                inlier_mean,
                inlier_covariance,
                inlier_quadratic,
                inlier_log_determinant,
                inlier_innovation_condition,
                inlier_posterior_condition,
            ) = _component_update(
                mean=mean[index],
                prior=prior,
                innovation=innovation,
                projected_covariance=projected_covariance,
                observation_matrix=row_matrix,
                observation_variance=observation_variance,
                name=f"{row_name} inlier",
                config=config,
            )
            (
                outlier_mean,
                outlier_covariance,
                outlier_quadratic,
                outlier_log_determinant,
                outlier_innovation_condition,
                outlier_posterior_condition,
            ) = _component_update(
                mean=mean[index],
                prior=prior,
                innovation=innovation,
                projected_covariance=projected_covariance,
                observation_matrix=row_matrix,
                observation_variance=outlier_variance,
                name=f"{row_name} outlier",
                config=config,
            )
        except SPDSystemError as error:
            raise DirectionalEndpointNumericalError(
                f"{row_name} failed SPD admission"
            ) from error

        log_inlier = np.log(inlier_prior) - 0.5 * (
            normalizer + inlier_log_determinant + inlier_quadratic
        )
        log_outlier = np.log1p(-inlier_prior) - 0.5 * (
            normalizer + outlier_log_determinant + outlier_quadratic
        )
        denominator = np.logaddexp(log_inlier, log_outlier)
        inlier_probability = float(np.exp(log_inlier - denominator))
        if not np.isfinite(inlier_probability):
            raise DirectionalEndpointNumericalError(
                f"{row_name} produced a non-finite mixture probability"
            )
        mixture_mean = (
            inlier_probability * inlier_mean
            + (1.0 - inlier_probability) * outlier_mean
        )
        if not np.all(np.isfinite(mixture_mean)):
            raise DirectionalEndpointNumericalError(
                f"{row_name} produced a non-finite mixture mean"
            )
        inlier_offset = inlier_mean - mixture_mean
        outlier_offset = outlier_mean - mixture_mean
        mixture_covariance = (
            inlier_probability
            * (inlier_covariance + np.outer(inlier_offset, inlier_offset))
            + (1.0 - inlier_probability)
            * (outlier_covariance + np.outer(outlier_offset, outlier_offset))
        )
        try:
            mixture = admit_spd_system(
                mixture_covariance,
                name=f"{row_name} mixture covariance",
                config=config,
            )
        except SPDSystemError as error:
            raise DirectionalEndpointNumericalError(
                f"{row_name} mixture failed SPD admission"
            ) from error

        updated_mean[index] = mixture_mean
        updated_covariance[index] = mixture.matrix
        probability[index] = inlier_probability
        innovation_condition[index] = max(
            inlier_innovation_condition,
            outlier_innovation_condition,
        )
        posterior_condition[index] = max(
            prior.condition_number,
            inlier_posterior_condition,
            outlier_posterior_condition,
            mixture.condition_number,
        )

    return (
        updated_mean,
        updated_covariance,
        probability,
        innovation_condition,
        posterior_condition,
    )


__all__ = [
    "DirectionalEndpointConfigV2",
    "DirectionalEndpointNumericalError",
    "admit_spd_system",
    "immutable_float64",
    "immutable_int64",
    "robust_linear_update_v2",
    "validate_filter_parameters",
]
