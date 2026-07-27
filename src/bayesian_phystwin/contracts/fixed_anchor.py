"""Immutable contracts for the fixed Bayesian discrepancy endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class FixedBayesianAnchorConfigV1:
    """Fully specified robust random-walk endpoint configuration."""

    process_std_m: float = 0.005
    observation_std_m: float = 0.001
    initial_std_m: float = 0.01
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0

    def __post_init__(self) -> None:
        process_std = float(self.process_std_m)
        observation_std = float(self.observation_std_m)
        initial_std = float(self.initial_std_m)
        inlier_prior = float(self.inlier_prior)
        outlier_multiplier = float(self.outlier_variance_multiplier)
        if not np.isfinite(process_std) or process_std < 0.0:
            raise ValueError("process_std_m must be finite and nonnegative")
        if not np.isfinite(observation_std) or observation_std <= 0.0:
            raise ValueError("observation_std_m must be finite and positive")
        if not np.isfinite(initial_std) or initial_std <= 0.0:
            raise ValueError("initial_std_m must be finite and positive")
        if not np.isfinite(inlier_prior) or not 0.0 < inlier_prior < 1.0:
            raise ValueError("inlier_prior must lie strictly between zero and one")
        if not np.isfinite(outlier_multiplier) or outlier_multiplier <= 1.0:
            raise ValueError("outlier_variance_multiplier must exceed one")
        object.__setattr__(self, "process_std_m", process_std)
        object.__setattr__(self, "observation_std_m", observation_std)
        object.__setattr__(self, "initial_std_m", initial_std)
        object.__setattr__(self, "inlier_prior", inlier_prior)
        object.__setattr__(
            self,
            "outlier_variance_multiplier",
            outlier_multiplier,
        )


DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1 = FixedBayesianAnchorConfigV1()


@dataclass(frozen=True, slots=True)
class RobustEndpointPosteriorV1:
    """Immutable per-track posterior at an exclusive causal frame cutoff."""

    mean_m: np.ndarray
    variance_m2: np.ndarray
    final_nominal_probability: np.ndarray
    update_count: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean_m, dtype=np.float64, copy=True, order="C")
        variance = np.array(self.variance_m2, dtype=np.float64, copy=True, order="C")
        probability = np.array(
            self.final_nominal_probability,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        raw_count = np.asarray(self.update_count)
        if not np.issubdtype(raw_count.dtype, np.integer):
            raise ValueError("update_count must contain integers")
        count = np.array(raw_count, dtype=np.int64, copy=True, order="C")
        if mean.ndim != 2 or mean.shape[1:] != (3,) or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        expected = (len(mean),)
        if variance.shape != expected:
            raise ValueError("variance_m2 must contain one scalar per track")
        if probability.shape != expected:
            raise ValueError(
                "final_nominal_probability must contain one value per track"
            )
        if count.shape != expected:
            raise ValueError("update_count must contain one value per track")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean_m must contain only finite values")
        if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("variance_m2 must be finite and nonnegative")
        if not np.all(np.isfinite(probability)) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError("final_nominal_probability must lie in [0, 1]")
        if np.any(count < 0):
            raise ValueError("update_count must be nonnegative")
        for values in (mean, variance, probability, count):
            values.setflags(write=False)
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "final_nominal_probability", probability)
        object.__setattr__(self, "update_count", count)

    @property
    def updated_mask(self) -> np.ndarray:
        """Return read-only tracks that received at least one update."""

        updated = self.update_count > 0
        updated.setflags(write=False)
        return updated

    @property
    def mean(self) -> np.ndarray:
        """Compatibility alias for the historical endpoint posterior."""

        return self.mean_m

    @property
    def variance(self) -> np.ndarray:
        """Compatibility alias for the historical endpoint posterior."""

        return self.variance_m2

    @property
    def final_inlier_probability(self) -> np.ndarray:
        """Compatibility alias for the nominal-mixture responsibility."""

        return self.final_nominal_probability


__all__ = [
    "DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1",
    "FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION",
    "FixedBayesianAnchorConfigV1",
    "RobustEndpointPosteriorV1",
]
