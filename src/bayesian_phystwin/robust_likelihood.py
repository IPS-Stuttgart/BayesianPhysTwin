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


@dataclass(frozen=True)
class RobustLikelihoodConfig:
    """Configuration for a Gaussian inlier/broad-Gaussian outlier mixture."""

    outlier_variance_multiplier: float = 100.0
    model_discrepancy_variance: float = 0.0
    probability_floor: float = 1e-6


@dataclass(frozen=True)
class RobustLikelihoodResult:
    """Per-measurement likelihood values and inferred inlier probabilities."""

    negative_log_likelihood: np.ndarray
    posterior_inlier_probability: np.ndarray
    log_inlier_density: np.ndarray
    log_outlier_density: np.ndarray

    @property
    def mean_negative_log_likelihood(self) -> float:
        return float(np.mean(self.negative_log_likelihood))


def _diagonal_gaussian_log_density(
    residual: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + np.square(residual) / variance,
        axis=1,
    )


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

    cfg = config or RobustLikelihoodConfig()
    if cfg.outlier_variance_multiplier <= 1.0:
        raise ValueError("outlier_variance_multiplier must be greater than 1")
    if cfg.model_discrepancy_variance < 0.0:
        raise ValueError("model_discrepancy_variance must be nonnegative")
    if not 0.0 < cfg.probability_floor < 0.5:
        raise ValueError("probability_floor must be in (0, 0.5)")

    observed, predicted = batch.arrays()
    residual = observed - predicted
    observation_variance = measurement_variance(batch)
    variance = observation_variance + cfg.model_discrepancy_variance
    n = observed.shape[0]

    if prior_reliability is None:
        prior = score_reliability(batch, reliability_config).weights
    else:
        prior = np.asarray(prior_reliability, dtype=float)
        if prior.shape != (n,):
            raise ValueError(f"prior_reliability must have shape ({n},), got {prior.shape}")
        if not np.all(np.isfinite(prior)):
            raise ValueError("prior_reliability must contain finite values")

    prior = np.clip(prior, cfg.probability_floor, 1.0 - cfg.probability_floor)
    log_inlier = _diagonal_gaussian_log_density(residual, variance)
    log_outlier = _diagonal_gaussian_log_density(
        residual,
        variance * cfg.outlier_variance_multiplier,
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
