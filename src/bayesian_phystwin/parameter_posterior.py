"""Small ensemble posterior utilities for low-dimensional physical parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParameterEnsemble:
    """Weighted particles for low-dimensional material/contact parameters."""

    particles: np.ndarray
    log_weights: np.ndarray

    @classmethod
    def from_prior_samples(cls, particles: np.ndarray) -> "ParameterEnsemble":
        particles = np.asarray(particles, dtype=float)
        if particles.ndim != 2:
            raise ValueError(f"particles must have shape (n, d), got {particles.shape}")
        n = particles.shape[0]
        return cls(particles=particles, log_weights=np.full(n, -np.log(n)))

    @property
    def weights(self) -> np.ndarray:
        normalized = self.log_weights - np.max(self.log_weights)
        weights = np.exp(normalized)
        total = np.sum(weights)
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("invalid particle weights")
        return weights / total

    @property
    def effective_sample_size(self) -> float:
        weights = self.weights
        return float(1.0 / np.sum(np.square(weights)))

    def mean(self) -> np.ndarray:
        return np.average(self.particles, axis=0, weights=self.weights)

    def covariance(self) -> np.ndarray:
        mean = self.mean()
        centered = self.particles - mean
        weights = self.weights
        return (centered * weights[:, None]).T @ centered

    def update_from_residuals(
        self,
        residual_sums: np.ndarray,
        *,
        variance: float,
        reliability: np.ndarray | None = None,
    ) -> None:
        """Bayesian update from per-particle residual sums.

        Args:
            residual_sums: Nonnegative residual objective for each particle.
            variance: Observation-model variance used as likelihood scale.
            reliability: Optional per-particle reliability factor in ``[0, 1]``.
                Low reliability tempers the update instead of over-penalizing
                likely corrupted observations.
        """

        residual_sums = np.asarray(residual_sums, dtype=float)
        if residual_sums.shape != self.log_weights.shape:
            raise ValueError(
                "residual_sums must match log_weights shape, got "
                f"{residual_sums.shape} and {self.log_weights.shape}"
            )
        if variance <= 0.0:
            raise ValueError("variance must be positive")

        if reliability is None:
            reliability_arr = np.ones_like(residual_sums)
        else:
            reliability_arr = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
            if reliability_arr.shape != residual_sums.shape:
                raise ValueError(
                    "reliability must match residual_sums shape, got "
                    f"{reliability_arr.shape} and {residual_sums.shape}"
                )

        log_likelihood = -0.5 * reliability_arr * residual_sums / variance
        self.log_weights = self.log_weights + log_likelihood
        self._renormalize_log_weights()

    def systematic_resample(
        self,
        rng: np.random.Generator | None = None,
        *,
        jitter_std: float | np.ndarray = 0.0,
    ) -> None:
        """Systematic resampling with optional Gaussian jitter."""

        rng = rng or np.random.default_rng()
        n = self.particles.shape[0]
        positions = (rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(self.weights)
        indexes = np.searchsorted(cumulative, positions, side="right")
        indexes = np.minimum(indexes, n - 1)
        self.particles = np.array(self.particles[indexes], copy=True)

        jitter = np.asarray(jitter_std, dtype=float)
        if np.any(jitter > 0.0):
            self.particles = self.particles + rng.normal(
                loc=0.0,
                scale=jitter,
                size=self.particles.shape,
            )
        self.log_weights = np.full(n, -np.log(n))

    def _renormalize_log_weights(self) -> None:
        max_log_weight = np.max(self.log_weights)
        shifted = self.log_weights - max_log_weight
        log_total = max_log_weight + np.log(np.sum(np.exp(shifted)))
        self.log_weights = self.log_weights - log_total

