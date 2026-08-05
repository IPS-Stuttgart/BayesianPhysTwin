"""Reliability-conditioned robust pseudo-measurement likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pseudo_measurements import (
    PseudoMeasurementBatch,
    ReliabilityConfig,
    measurement_variance,
    score_reliability,
)


def _finite_real_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real number")
    return scalar


def _owned_float_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    return np.array(raw, dtype=np.float64, copy=True, order="C")


@dataclass(frozen=True)
class RobustLikelihoodConfig:
    """Configuration for a Gaussian inlier/broad-Gaussian outlier mixture."""

    outlier_variance_multiplier: float = 100.0
    model_discrepancy_variance: float = 0.0
    probability_floor: float = 1e-6

    def __post_init__(self) -> None:
        multiplier = _finite_real_scalar(
            self.outlier_variance_multiplier,
            name="outlier_variance_multiplier",
        )
        discrepancy = _finite_real_scalar(
            self.model_discrepancy_variance,
            name="model_discrepancy_variance",
        )
        floor = _finite_real_scalar(
            self.probability_floor,
            name="probability_floor",
        )
        if multiplier <= 1.0:
            raise ValueError("outlier_variance_multiplier must be greater than 1")
        if discrepancy < 0.0:
            raise ValueError("model_discrepancy_variance must be nonnegative")
        if not 0.0 < floor < 0.5:
            raise ValueError("probability_floor must be in (0, 0.5)")
        object.__setattr__(self, "outlier_variance_multiplier", multiplier)
        object.__setattr__(self, "model_discrepancy_variance", discrepancy)
        object.__setattr__(self, "probability_floor", floor)


@dataclass(frozen=True)
class RobustLikelihoodResult:
    """Per-measurement likelihood values and inferred inlier probabilities."""

    negative_log_likelihood: np.ndarray
    posterior_inlier_probability: np.ndarray
    log_inlier_density: np.ndarray
    log_outlier_density: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "negative_log_likelihood": _owned_float_array(
                self.negative_log_likelihood,
                name="negative_log_likelihood",
            ),
            "posterior_inlier_probability": _owned_float_array(
                self.posterior_inlier_probability,
                name="posterior_inlier_probability",
            ),
            "log_inlier_density": _owned_float_array(
                self.log_inlier_density,
                name="log_inlier_density",
            ),
            "log_outlier_density": _owned_float_array(
                self.log_outlier_density,
                name="log_outlier_density",
            ),
        }
        shape = arrays["negative_log_likelihood"].shape
        if len(shape) != 1 or not shape or shape[0] == 0:
            raise ValueError("likelihood results must be nonempty vectors")
        for name, array in arrays.items():
            if array.shape != shape:
                raise ValueError("likelihood result vectors must have equal shape")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain finite values")
        posterior = arrays["posterior_inlier_probability"]
        if np.any((posterior < 0.0) | (posterior > 1.0)):
            raise ValueError("posterior_inlier_probability must lie in [0, 1]")

        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def mean_negative_log_likelihood(self) -> float:
        return float(np.mean(self.negative_log_likelihood))


def _diagonal_gaussian_log_density(
    residual: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        density = -0.5 * np.sum(
            np.log(2.0 * np.pi * variance) + np.square(residual) / variance,
            axis=1,
        )
    if not np.all(np.isfinite(density)):
        raise ValueError("Gaussian log density is not representable as a finite value")
    return density


def robust_mixture_likelihood(
    batch: PseudoMeasurementBatch,
    *,
    prior_reliability: np.ndarray | None = None,
    reliability_config: ReliabilityConfig | None = None,
    config: RobustLikelihoodConfig | None = None,
) -> RobustLikelihoodResult:
    """Evaluate a reliability-conditioned inlier/outlier mixture.

    Reliability is the prior inlier probability. The returned posterior inlier
    probability additionally conditions on the observed residual. Keeping
    these quantities separate avoids using the same residual twice.
    """

    cfg = config if config is not None else RobustLikelihoodConfig()
    if not isinstance(cfg, RobustLikelihoodConfig):
        raise TypeError("config must be a RobustLikelihoodConfig")

    observed, predicted = batch.arrays()
    with np.errstate(over="ignore", invalid="ignore"):
        residual = observed - predicted
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual must be representable as finite values")
    observation_variance = measurement_variance(batch)
    with np.errstate(over="ignore", invalid="ignore"):
        variance = observation_variance + cfg.model_discrepancy_variance
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("combined inlier variance must be finite and positive")
    n = observed.shape[0]

    if prior_reliability is None:
        prior = score_reliability(batch, reliability_config).weights
    else:
        raw_prior = np.asarray(prior_reliability)
        if raw_prior.dtype.kind not in "iuf":
            raise ValueError("prior_reliability must contain real numeric values")
        prior = np.asarray(raw_prior, dtype=float)
        if prior.shape != (n,):
            raise ValueError(
                f"prior_reliability must have shape ({n},), got {prior.shape}"
            )
        if not np.all(np.isfinite(prior)):
            raise ValueError("prior_reliability must contain finite values")
        if np.any((prior < 0.0) | (prior > 1.0)):
            raise ValueError("prior_reliability must lie in [0, 1]")

    prior = np.clip(prior, cfg.probability_floor, 1.0 - cfg.probability_floor)
    with np.errstate(over="ignore", invalid="ignore"):
        outlier_variance = variance * cfg.outlier_variance_multiplier
    if not np.all(np.isfinite(outlier_variance)):
        raise ValueError("combined outlier variance must be finite")
    log_inlier = _diagonal_gaussian_log_density(residual, variance)
    log_outlier = _diagonal_gaussian_log_density(
        residual,
        outlier_variance,
    )

    log_inlier_component = np.log(prior) + log_inlier
    log_outlier_component = np.log1p(-prior) + log_outlier
    log_mixture = np.logaddexp(log_inlier_component, log_outlier_component)
    posterior_inlier = np.exp(log_inlier_component - log_mixture)

    return RobustLikelihoodResult(
        negative_log_likelihood=-log_mixture,
        posterior_inlier_probability=posterior_inlier,
        log_inlier_density=log_inlier,
        log_outlier_density=log_outlier,
    )
