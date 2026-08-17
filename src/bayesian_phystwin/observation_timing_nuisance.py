"""Explicit observation-timing nuisance for Bayesian physical updates.

A camera-to-actuator or provider-to-simulator clock offset creates a coherent
innovation proportional to the observation trajectory derivative. Treating
that effect as independent point noise can make a timing error look like a
physical discrepancy. This module builds the corresponding linear nuisance
design, checks confounding against competing state/bias designs, and provides a
small source-only Gaussian calibration update for hardware timing priors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_NUMERICAL_TOLERANCE = 1e-12


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 1, f"{name} must be a vector")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2, f"{name} must be a matrix")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _symmetric_positive_definite(value: np.ndarray, *, name: str) -> np.ndarray:
    result = _finite_matrix(value, name=name)
    _require(result.shape[0] == result.shape[1], f"{name} must be square")
    _require(np.allclose(result, result.T), f"{name} must be symmetric")
    result = 0.5 * (result + result.T)
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return result


def _solve_spd(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    cholesky = np.linalg.cholesky(matrix)
    lower_solution = np.linalg.solve(cholesky, right_hand_side)
    return np.linalg.solve(cholesky.T, lower_solution)


@dataclass(frozen=True)
class ObservationTimingPrior:
    """Source-derived prior for one named clock-domain offset."""

    clock_domain: str
    mean_offset_s: float
    standard_deviation_s: float
    source_artifact_id: str

    def __post_init__(self) -> None:
        _require(bool(self.clock_domain.strip()), "clock_domain must be nonempty")
        _require(
            np.isfinite(self.mean_offset_s),
            "mean_offset_s must be finite",
        )
        _require(
            np.isfinite(self.standard_deviation_s) and self.standard_deviation_s > 0.0,
            "standard_deviation_s must be finite and positive",
        )
        _require(
            bool(self.source_artifact_id.strip()),
            "source_artifact_id must be nonempty",
        )

    @property
    def precision_per_s2(self) -> float:
        """Return scalar Gaussian prior precision."""

        return 1.0 / (self.standard_deviation_s**2)


def timing_prior_mean(priors: Sequence[ObservationTimingPrior]) -> np.ndarray:
    """Return ordered timing-prior means for explicit nuisance state."""

    result = np.asarray([prior.mean_offset_s for prior in priors], dtype=np.float64)
    result.setflags(write=False)
    return result


def timing_prior_precision(priors: Sequence[ObservationTimingPrior]) -> np.ndarray:
    """Return diagonal timing-prior precision in declared clock-domain order."""

    domains = [prior.clock_domain for prior in priors]
    _require(len(set(domains)) == len(domains), "clock domains must be unique")
    precision = np.diag([prior.precision_per_s2 for prior in priors]).astype(
        np.float64,
        copy=False,
    )
    precision.setflags(write=False)
    return precision


def build_timing_jacobian(
    observation_derivative_per_s: np.ndarray,
    *,
    stream_indices: np.ndarray | None = None,
    stream_count: int | None = None,
) -> np.ndarray:
    """Build ``dh/dt`` columns for one or several clock-domain offsets.

    The derivative is one scalar value per observation row. For vector-valued
    observations, flatten coordinates using the same row order as the state and
    competing nuisance Jacobians. ``stream_indices`` assigns each row to one
    timing offset; omitted indices create one shared timing column.
    """

    derivative = _finite_vector(
        observation_derivative_per_s,
        name="observation_derivative_per_s",
    )
    _require(len(derivative) >= 1, "observation derivative must not be empty")

    if stream_indices is None:
        _require(
            stream_count is None or stream_count == 1,
            "stream_count must be one when stream_indices are omitted",
        )
        result = derivative[:, None].copy()
    else:
        indices = np.asarray(stream_indices)
        _require(indices.shape == derivative.shape, "stream index row count changed")
        _require(
            np.issubdtype(indices.dtype, np.integer),
            "stream_indices must contain integers",
        )
        indices = indices.astype(np.int64, copy=False)
        _require(np.all(indices >= 0), "stream_indices must be nonnegative")
        inferred_count = int(np.max(indices)) + 1
        count = inferred_count if stream_count is None else int(stream_count)
        _require(count >= inferred_count, "stream_count excludes assigned rows")
        _require(count >= 1, "stream_count must be positive")
        result = np.zeros((len(derivative), count), dtype=np.float64)
        result[np.arange(len(derivative)), indices] = derivative

    _require(
        np.all(np.linalg.norm(result, axis=0) > 0.0),
        "every timing stream must have at least one nonzero derivative row",
    )
    result.setflags(write=False)
    return result


def append_timing_nuisance(
    nuisance_jacobian: np.ndarray | None,
    timing_jacobian: np.ndarray,
) -> np.ndarray:
    """Append explicit timing columns to an existing nuisance design."""

    timing = _finite_matrix(timing_jacobian, name="timing_jacobian")
    _require(timing.shape[1] >= 1, "timing_jacobian must contain a timing column")
    if nuisance_jacobian is None:
        result = timing.copy()
    else:
        nuisance = _finite_matrix(nuisance_jacobian, name="nuisance_jacobian")
        _require(
            nuisance.shape[0] == timing.shape[0],
            "nuisance and timing row counts differ",
        )
        result = np.concatenate([nuisance, timing], axis=1)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TimingIdentifiabilityResult:
    """Confounding diagnostic for each explicit timing column."""

    identifiable: bool
    reason: str
    residual_fractions: np.ndarray
    subspace_cosines: np.ndarray

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residual_fractions, dtype=np.float64).copy()
        cosines = np.asarray(self.subspace_cosines, dtype=np.float64).copy()
        _require(residuals.ndim == 1, "residual_fractions must be a vector")
        _require(cosines.shape == residuals.shape, "confounding counts differ")
        _require(
            np.all(np.isfinite(residuals))
            and np.all((residuals >= 0.0) & (residuals <= 1.0 + 1e-12)),
            "residual fractions must lie in [0, 1]",
        )
        _require(
            np.all(np.isfinite(cosines))
            and np.all((cosines >= 0.0) & (cosines <= 1.0 + 1e-12)),
            "subspace cosines must lie in [0, 1]",
        )
        _require(bool(self.reason), "identifiability reason must be nonempty")
        residuals.setflags(write=False)
        cosines.setflags(write=False)
        object.__setattr__(self, "residual_fractions", residuals)
        object.__setattr__(self, "subspace_cosines", cosines)


def assess_timing_identifiability(
    timing_jacobian: np.ndarray,
    competing_design: np.ndarray | None,
    *,
    independent_timing_jacobian: np.ndarray | None = None,
    maximum_subspace_cosine: float = 0.999,
) -> TimingIdentifiabilityResult:
    """Check whether timing columns survive projection beyond competing modes.

    ``competing_design`` may contain physical-state, spatial-bias, gauge, or
    material-lag columns that must be distinguished from timing. Optional
    independent timing rows model a source-only synchronization pulse, contact
    sensor, or another anchor on which those competing modes are absent.
    """

    timing = _finite_matrix(timing_jacobian, name="timing_jacobian")
    _require(timing.shape[1] >= 1, "timing_jacobian must contain a timing column")
    _require(
        np.isfinite(maximum_subspace_cosine) and 0.0 <= maximum_subspace_cosine < 1.0,
        "maximum_subspace_cosine must lie in [0, 1)",
    )
    if competing_design is None:
        competing = np.empty((timing.shape[0], 0), dtype=np.float64)
    else:
        competing = _finite_matrix(competing_design, name="competing_design")
        _require(
            competing.shape[0] == timing.shape[0],
            "competing and timing row counts differ",
        )

    if independent_timing_jacobian is not None:
        independent = _finite_matrix(
            independent_timing_jacobian,
            name="independent_timing_jacobian",
        )
        _require(
            independent.shape[1] == timing.shape[1],
            "independent timing column count changed",
        )
        timing = np.vstack([timing, independent])
        competing = np.vstack(
            [
                competing,
                np.zeros(
                    (independent.shape[0], competing.shape[1]),
                    dtype=np.float64,
                ),
            ]
        )

    if competing.shape[1] == 0:
        residual = timing.copy()
    else:
        left, singular_values, _ = np.linalg.svd(competing, full_matrices=False)
        if len(singular_values):
            tolerance = (
                max(competing.shape) * np.finfo(np.float64).eps * singular_values[0]
            )
            rank = int(np.sum(singular_values > tolerance))
        else:
            rank = 0
        basis = left[:, :rank]
        residual = timing - basis @ (basis.T @ timing)

    timing_norms = np.linalg.norm(timing, axis=0)
    _require(
        np.all(timing_norms > 0.0),
        "timing_jacobian contains an empty timing column",
    )
    residual_fractions = np.linalg.norm(residual, axis=0) / timing_norms
    residual_fractions = np.clip(residual_fractions, 0.0, 1.0)
    subspace_cosines = np.sqrt(np.maximum(0.0, 1.0 - residual_fractions**2))
    identifiable = bool(np.all(subspace_cosines <= maximum_subspace_cosine))
    reason = (
        "timing design is distinguishable from declared competing modes"
        if identifiable
        else "timing design is confounded with declared competing modes"
    )
    return TimingIdentifiabilityResult(
        identifiable=identifiable,
        reason=reason,
        residual_fractions=residual_fractions,
        subspace_cosines=subspace_cosines,
    )


@dataclass(frozen=True)
class TimingCalibrationPosterior:
    """Gaussian timing posterior from source-only synchronization evidence."""

    mean_offset_s: np.ndarray
    covariance_s2: np.ndarray
    information_gain_nats: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_offset_s, dtype=np.float64).copy()
        covariance = _symmetric_positive_definite(
            self.covariance_s2,
            name="covariance_s2",
        ).copy()
        _require(mean.ndim == 1, "mean_offset_s must be a vector")
        _require(
            covariance.shape == (len(mean), len(mean)),
            "timing posterior covariance shape changed",
        )
        _require(np.all(np.isfinite(mean)), "timing posterior mean is not finite")
        _require(
            np.isfinite(self.information_gain_nats)
            and self.information_gain_nats >= 0.0,
            "information_gain_nats must be finite and nonnegative",
        )
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean_offset_s", mean)
        object.__setattr__(self, "covariance_s2", covariance)


def condition_timing_prior(
    innovation: np.ndarray,
    timing_jacobian: np.ndarray,
    observation_covariance: np.ndarray,
    prior_mean_s: np.ndarray,
    prior_covariance_s2: np.ndarray,
) -> TimingCalibrationPosterior:
    """Condition a source-only Gaussian timing prior on synchronization data."""

    residual = _finite_vector(innovation, name="innovation")
    timing = _finite_matrix(timing_jacobian, name="timing_jacobian")
    _require(timing.shape[0] == len(residual), "timing innovation row count changed")
    prior_mean = _finite_vector(prior_mean_s, name="prior_mean_s")
    _require(
        timing.shape[1] == len(prior_mean),
        "timing prior dimension changed",
    )
    observation_covariance_array = _symmetric_positive_definite(
        observation_covariance,
        name="observation_covariance",
    )
    _require(
        observation_covariance_array.shape == (len(residual), len(residual)),
        "observation covariance row count changed",
    )
    prior_covariance = _symmetric_positive_definite(
        prior_covariance_s2,
        name="prior_covariance_s2",
    )
    _require(
        prior_covariance.shape == (len(prior_mean), len(prior_mean)),
        "prior covariance dimension changed",
    )

    prior_precision = _solve_spd(
        prior_covariance,
        np.eye(len(prior_mean), dtype=np.float64),
    )
    observation_cholesky = np.linalg.cholesky(observation_covariance_array)
    whitened_timing = np.linalg.solve(observation_cholesky, timing)
    whitened_residual = np.linalg.solve(observation_cholesky, residual)
    posterior_precision = prior_precision + whitened_timing.T @ whitened_timing
    information_vector = (
        prior_precision @ prior_mean + whitened_timing.T @ whitened_residual
    )
    posterior_covariance = _solve_spd(
        posterior_precision,
        np.eye(len(prior_mean), dtype=np.float64),
    )
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    posterior_mean = posterior_covariance @ information_vector

    prior_logdet = float(np.linalg.slogdet(prior_covariance)[1])
    posterior_logdet = float(np.linalg.slogdet(posterior_covariance)[1])
    information_gain = max(0.0, 0.5 * (prior_logdet - posterior_logdet))
    return TimingCalibrationPosterior(
        mean_offset_s=posterior_mean,
        covariance_s2=posterior_covariance,
        information_gain_nats=information_gain,
    )
