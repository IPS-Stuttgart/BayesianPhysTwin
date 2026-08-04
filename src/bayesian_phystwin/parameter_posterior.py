"""Small ensemble posterior utilities for low-dimensional physical parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParameterEnsemble:
    """Weighted particles for low-dimensional material/contact parameters.

    Arrays are defensively copied on construction. The class remains mutable so
    sequential updates and resampling preserve the historical API, but every
    public operation revalidates the particle/weight state and fails closed on
    non-finite or shape-corrupted values.
    """

    particles: np.ndarray
    log_weights: np.ndarray

    def __post_init__(self) -> None:
        particles = np.array(
            self.particles,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        log_weights = np.array(
            self.log_weights,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        self.particles = particles
        self.log_weights = log_weights
        self._validate_state()
        self._renormalize_log_weights()

    @classmethod
    def from_prior_samples(cls, particles: np.ndarray) -> ParameterEnsemble:
        values = np.asarray(particles)
        if values.ndim != 2:
            raise ValueError(f"particles must have shape (n, d), got {values.shape}")
        if values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("particles must contain at least one nonempty sample")
        if not np.all(np.isfinite(values)):
            raise ValueError("particles must contain only finite values")
        n = values.shape[0]
        return cls(particles=values, log_weights=np.full(n, -np.log(n)))

    @property
    def particle_count(self) -> int:
        self._validate_state()
        return int(self.particles.shape[0])

    @property
    def dimension(self) -> int:
        self._validate_state()
        return int(self.particles.shape[1])

    @property
    def weights(self) -> np.ndarray:
        self._validate_state()
        normalized = self.log_weights - np.max(self.log_weights)
        weights = np.exp(normalized)
        total = np.sum(weights)
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("invalid particle weights")
        result = weights / total
        result.setflags(write=False)
        return result

    @property
    def effective_sample_size(self) -> float:
        weights = self.weights
        return float(1.0 / np.sum(np.square(weights)))

    def mean(self) -> np.ndarray:
        result = np.average(self.particles, axis=0, weights=self.weights)
        result.setflags(write=False)
        return result

    def covariance(self) -> np.ndarray:
        mean = self.mean()
        centered = self.particles - mean
        weights = self.weights
        covariance = (centered * weights[:, None]).T @ centered
        covariance = 0.5 * (covariance + covariance.T)
        covariance.setflags(write=False)
        return covariance

    def update_from_residuals(
        self,
        residual_sums: np.ndarray,
        *,
        variance: float,
        reliability: np.ndarray | None = None,
    ) -> None:
        """Bayesian update from per-particle nonnegative residual objectives.

        Args:
            residual_sums: Nonnegative residual objective for each particle.
            variance: Finite positive observation-model variance.
            reliability: Optional per-particle likelihood power in ``[0, 1]``.
                Low reliability tempers the update instead of over-penalizing
                likely corrupted observations.
        """

        self._validate_state()
        residual_array = np.asarray(residual_sums, dtype=np.float64)
        if residual_array.shape != self.log_weights.shape:
            raise ValueError(
                "residual_sums must match log_weights shape, got "
                f"{residual_array.shape} and {self.log_weights.shape}"
            )
        if not np.all(np.isfinite(residual_array)):
            raise ValueError("residual_sums must contain only finite values")
        if np.any(residual_array < 0.0):
            raise ValueError("residual_sums must be nonnegative")
        variance_value = float(variance)
        if not np.isfinite(variance_value) or variance_value <= 0.0:
            raise ValueError("variance must be finite and positive")

        if reliability is None:
            reliability_array = np.ones_like(residual_array)
        else:
            reliability_array = np.asarray(reliability, dtype=np.float64)
            if reliability_array.shape != residual_array.shape:
                raise ValueError(
                    "reliability must match residual_sums shape, got "
                    f"{reliability_array.shape} and {residual_array.shape}"
                )
            if not np.all(np.isfinite(reliability_array)):
                raise ValueError("reliability must contain only finite values")
            if np.any((reliability_array < 0.0) | (reliability_array > 1.0)):
                raise ValueError("reliability must lie in [0, 1]")

        with np.errstate(over="raise", invalid="raise"):
            try:
                log_likelihood = (
                    -0.5 * reliability_array * residual_array / variance_value
                )
            except FloatingPointError as error:
                raise FloatingPointError(
                    "residual likelihood overflowed its finite contract"
                ) from error
        if not np.all(np.isfinite(log_likelihood)):
            raise FloatingPointError("residual likelihood is non-finite")
        self.log_weights = np.asarray(
            self.log_weights + log_likelihood,
            dtype=np.float64,
        )
        self._renormalize_log_weights()

    def systematic_resample(
        self,
        rng: np.random.Generator | None = None,
        *,
        jitter_std: float | np.ndarray = 0.0,
    ) -> None:
        """Systematic resampling with optional finite Gaussian jitter.

        ``jitter_std`` may be a scalar, one scale per parameter dimension, or
        one scale per particle and dimension. Negative, non-finite, or
        non-broadcast-compatible scales are rejected instead of being silently
        ignored.
        """

        self._validate_state()
        generator = rng or np.random.default_rng()
        n, dimension = self.particles.shape
        jitter = np.asarray(jitter_std, dtype=np.float64)
        allowed_shapes = {(), (dimension,), self.particles.shape}
        if jitter.shape not in allowed_shapes:
            raise ValueError(
                "jitter_std must be scalar, shape "
                f"({dimension},), or {self.particles.shape}"
            )
        if not np.all(np.isfinite(jitter)) or np.any(jitter < 0.0):
            raise ValueError("jitter_std must be finite and nonnegative")

        positions = (generator.random() + np.arange(n)) / n
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, positions, side="right")
        indexes = np.minimum(indexes, n - 1)
        particles = np.array(
            self.particles[indexes],
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if np.any(jitter > 0.0):
            particles += generator.normal(
                loc=0.0,
                scale=jitter,
                size=particles.shape,
            )
        if not np.all(np.isfinite(particles)):
            raise FloatingPointError("resampled particles are non-finite")
        self.particles = particles
        self.log_weights = np.full(n, -np.log(n), dtype=np.float64)

    def _validate_state(self) -> None:
        particles = np.asarray(self.particles)
        log_weights = np.asarray(self.log_weights)
        if particles.ndim != 2:
            raise ValueError(f"particles must have shape (n, d), got {particles.shape}")
        if particles.shape[0] < 1 or particles.shape[1] < 1:
            raise ValueError("particles must contain at least one nonempty sample")
        if log_weights.shape != (particles.shape[0],):
            raise ValueError(
                "log_weights must contain one value per particle, got "
                f"{log_weights.shape} for {particles.shape[0]} particles"
            )
        if not np.all(np.isfinite(particles)):
            raise ValueError("particles must contain only finite values")
        if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
            raise ValueError("log_weights must be finite or negative infinity")
        if not np.any(np.isfinite(log_weights)):
            raise ValueError("at least one particle must have finite log weight")

    def _renormalize_log_weights(self) -> None:
        self._validate_state()
        finite = np.isfinite(self.log_weights)
        max_log_weight = float(np.max(self.log_weights[finite]))
        shifted = self.log_weights - max_log_weight
        total = float(np.sum(np.exp(shifted[finite])))
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("invalid particle-weight normalization")
        log_total = max_log_weight + np.log(total)
        self.log_weights = np.asarray(
            self.log_weights - log_total,
            dtype=np.float64,
        )
