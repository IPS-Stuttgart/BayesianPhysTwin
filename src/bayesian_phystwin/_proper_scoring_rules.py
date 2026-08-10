"""Numerically exact proper scoring rules for registered forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._proper_scoring_contracts import (
    _Forecast,
    _Pair,
    _ScalarIntervalForecast,
    _require,
)


def empirical_energy_score(
    observation: np.ndarray,
    samples: np.ndarray,
    *,
    maximum_pair_evaluations: int,
    block_size: int = 256,
) -> float:
    """Return the energy score of one finite empirical distribution."""

    y = np.asarray(observation, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    _require(y.ndim == 1, "observation must be a vector")
    _require(draws.ndim == 2, "samples must be a matrix")
    _require(draws.shape[1] == len(y), "samples have changed query dimension")
    pair_count = len(draws) ** 2
    _require(
        pair_count <= maximum_pair_evaluations,
        "energy-score pair-evaluation budget exceeded",
    )
    first = float(np.mean(np.linalg.norm(draws - y, axis=1)))
    squared_norms = np.einsum("ij,ij->i", draws, draws, optimize=True)
    total = 0.0
    for start in range(0, len(draws), block_size):
        left = draws[start : start + block_size]
        left_norms = squared_norms[start : start + block_size]
        squared = (
            left_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * (left @ draws.T)
        )
        np.maximum(squared, 0.0, out=squared)
        total += float(np.sum(np.sqrt(squared)))
    second = 0.5 * total / pair_count
    score = first - second
    tolerance = 1e-12 * (1.0 + abs(first) + abs(second))
    _require(score >= -tolerance, "energy score became materially negative")
    return max(0.0, score)


def _empirical_variogram_score(
    observation: np.ndarray,
    samples: np.ndarray,
    pairs: Sequence[_Pair],
    *,
    power: float,
    maximum_evaluations: int,
) -> float:
    y = np.asarray(observation, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    _require(
        len(draws) * len(pairs) <= maximum_evaluations,
        "variogram-score evaluation budget exceeded",
    )
    score = 0.0
    for pair in pairs:
        observed = abs(float(y[pair.left] - y[pair.right])) ** power
        predictive = float(
            np.mean(
                np.abs(draws[:, pair.left] - draws[:, pair.right]) ** power
            )
        )
        score += pair.weight * (observed - predictive) ** 2
    return float(score)


def gaussian_log_score(
    observation: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return exact Gaussian negative log predictive density."""

    centered = np.asarray(observation, dtype=np.float64) - np.asarray(
        mean, dtype=np.float64
    )
    matrix = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    factor = np.linalg.cholesky(matrix)
    whitened = np.linalg.solve(factor, centered)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
    return 0.5 * (
        len(centered) * np.log(2.0 * np.pi)
        + log_determinant
        + float(whitened @ whitened)
    )


def _weighted_interval_score(
    observation: float,
    forecast: _ScalarIntervalForecast,
) -> tuple[float, tuple[dict[str, object], ...]]:
    numerator = 0.5 * abs(observation - forecast.median)
    intervals: list[dict[str, object]] = []
    for interval in forecast.intervals:
        alpha = 1.0 - interval.nominal_coverage
        interval_score = interval.upper - interval.lower
        if observation < interval.lower:
            interval_score += 2.0 * (interval.lower - observation) / alpha
        elif observation > interval.upper:
            interval_score += 2.0 * (observation - interval.upper) / alpha
        numerator += 0.5 * alpha * interval_score
        intervals.append(
            {
                "nominal_coverage": interval.nominal_coverage,
                "covered": interval.lower <= observation <= interval.upper,
                "width": interval.upper - interval.lower,
            }
        )
    denominator = len(forecast.intervals) + 0.5
    return numerator / denominator, tuple(intervals)


def _score_forecast(
    observation: np.ndarray,
    forecast: _Forecast,
    pairs: tuple[_Pair, ...],
    *,
    variogram_power: float,
    gaussian_log_score_offset: float,
    maximum_energy_pair_evaluations: int,
    maximum_variogram_evaluations: int,
) -> dict[str, tuple[float, tuple[dict[str, object], ...], dict[str, object]]]:
    scores: dict[
        str,
        tuple[float, tuple[dict[str, object], ...], dict[str, object]],
    ] = {}
    if forecast.samples is not None:
        energy = empirical_energy_score(
            observation,
            forecast.samples,
            maximum_pair_evaluations=maximum_energy_pair_evaluations,
        )
        scores["energy_score"] = (
            energy,
            (),
            {"raw_score": energy, "additive_offset": 0.0},
        )
        if pairs:
            variogram = _empirical_variogram_score(
                observation,
                forecast.samples,
                pairs,
                power=variogram_power,
                maximum_evaluations=maximum_variogram_evaluations,
            )
            scores["variogram_score"] = (
                variogram,
                (),
                {"raw_score": variogram, "additive_offset": 0.0},
            )
    if (
        forecast.gaussian_mean is not None
        and forecast.gaussian_covariance is not None
    ):
        raw = gaussian_log_score(
            observation,
            forecast.gaussian_mean,
            forecast.gaussian_covariance,
        )
        shifted = raw + gaussian_log_score_offset
        tolerance = 1e-12 * (1.0 + abs(raw) + gaussian_log_score_offset)
        _require(
            shifted >= -tolerance,
            "shifted Gaussian log score is negative; freeze a larger common "
            "gaussian_log_score_offset",
        )
        scores["gaussian_log_score_shifted"] = (
            max(0.0, shifted),
            (),
            {
                "raw_score": raw,
                "additive_offset": gaussian_log_score_offset,
            },
        )
    if forecast.scalar_intervals is not None:
        score, intervals = _weighted_interval_score(
            float(observation[0]), forecast.scalar_intervals
        )
        scores["weighted_interval_score"] = (
            score,
            intervals,
            {"raw_score": score, "additive_offset": 0.0},
        )
    return scores


__all__ = [
    "_score_forecast",
    "empirical_energy_score",
    "gaussian_log_score",
]
