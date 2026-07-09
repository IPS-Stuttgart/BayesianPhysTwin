"""Reliability-aware pseudo-measurement models.

The functions here are deliberately independent of PhysTwin internals. A
PhysTwin adapter only has to provide observed values, simulated predictions,
and optional confidence/occlusion/flow cues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray | list[float] | list[list[float]]


@dataclass(frozen=True)
class PseudoMeasurementBatch:
    """Observed and predicted pseudo-measurements.

    Attributes:
        observed: Array with shape ``(n, d)``.
        predicted: Array with shape ``(n, d)``.
        variance: Per-coordinate observation variance. Accepts scalar,
            ``(n,)``, or ``(n, d)`` values.
        confidence: Optional learned confidence in ``[0, 1]``.
        occluded: Optional boolean mask where true means the measurement should
            be strongly downweighted.
        boundary_distance: Optional nonnegative distance to the nearest mask
            boundary. Low values imply higher segmentation ambiguity.
        flow_inconsistency: Optional nonnegative scene-flow consistency score.
            Higher values imply lower reliability.
    """

    observed: ArrayLike
    predicted: ArrayLike
    variance: ArrayLike | float = 1.0
    confidence: ArrayLike | None = None
    occluded: ArrayLike | None = None
    boundary_distance: ArrayLike | None = None
    flow_inconsistency: ArrayLike | None = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        observed = np.asarray(self.observed, dtype=float)
        predicted = np.asarray(self.predicted, dtype=float)
        if observed.shape != predicted.shape:
            raise ValueError(
                "observed and predicted must have the same shape, got "
                f"{observed.shape} and {predicted.shape}"
            )
        if observed.ndim != 2:
            raise ValueError(f"expected arrays with shape (n, d), got {observed.shape}")
        return observed, predicted


@dataclass(frozen=True)
class ReliabilityConfig:
    """Parameters for converting perception cues into prior reliability.

    ``residual_scale`` is optional because a residual is evidence about whether
    a measurement is an inlier, not an independent perception cue. Leave it at
    ``None`` for the robust-mixture model and set it only for the
    residual-gated baseline.
    """

    min_weight: float = 1e-3
    confidence_power: float = 1.0
    residual_scale: float | None = None
    boundary_scale: float = 0.03
    flow_scale: float = 0.10
    occlusion_weight: float = 0.05
    covariance_inflation_at_min_weight: float = 100.0


@dataclass(frozen=True)
class ReliabilityResult:
    """Reliability scores and covariance inflation for a batch."""

    weights: np.ndarray
    inflated_variance: np.ndarray
    residual_norm: np.ndarray

    @property
    def effective_sample_size(self) -> float:
        total = float(np.sum(self.weights))
        denom = float(np.sum(np.square(self.weights)))
        if denom == 0.0:
            return 0.0
        return total * total / denom


def _as_vector(
    values: ArrayLike | None,
    n: int,
    *,
    default: float,
    name: str,
    dtype: type = float,
) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=dtype)
    arr = np.asarray(values, dtype=dtype)
    if arr.shape == ():
        return np.full(n, arr.item(), dtype=dtype)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},), got {arr.shape}")
    return arr


def _variance_array(values: ArrayLike | float, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n, d = shape
    if arr.shape == ():
        return np.full(shape, arr.item(), dtype=float)
    if arr.shape == (n,):
        return np.repeat(arr[:, None], d, axis=1)
    if arr.shape == shape:
        return arr
    raise ValueError(f"variance must be scalar, shape ({n},), or {shape}, got {arr.shape}")


def measurement_variance(batch: PseudoMeasurementBatch) -> np.ndarray:
    """Return validated per-coordinate variance with shape ``(n, d)``."""

    observed, _ = batch.arrays()
    variance = _variance_array(batch.variance, observed.shape)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("variance values must be finite and positive")
    return variance


def _validate_config(config: ReliabilityConfig) -> None:
    if not 0.0 < config.min_weight <= 1.0:
        raise ValueError("min_weight must be in (0, 1]")
    if config.confidence_power < 0.0:
        raise ValueError("confidence_power must be nonnegative")
    if config.residual_scale is not None and config.residual_scale <= 0.0:
        raise ValueError("residual_scale must be positive when provided")
    if config.boundary_scale <= 0.0:
        raise ValueError("boundary_scale must be positive")
    if config.flow_scale <= 0.0:
        raise ValueError("flow_scale must be positive")
    if not 0.0 <= config.occlusion_weight <= 1.0:
        raise ValueError("occlusion_weight must be in [0, 1]")
    if config.covariance_inflation_at_min_weight < 1.0:
        raise ValueError("covariance inflation cap must be at least 1")


def score_reliability(
    batch: PseudoMeasurementBatch,
    config: ReliabilityConfig | None = None,
) -> ReliabilityResult:
    """Score prior reliability from learned cues and optional residual gating."""

    cfg = config or ReliabilityConfig()
    _validate_config(cfg)
    observed, predicted = batch.arrays()
    residual = observed - predicted
    residual_norm = np.linalg.norm(residual, axis=1)
    n = observed.shape[0]

    confidence = np.clip(
        _as_vector(batch.confidence, n, default=1.0, name="confidence"),
        0.0,
        1.0,
    )
    occluded = _as_vector(batch.occluded, n, default=False, name="occluded", dtype=bool)
    boundary_distance = np.maximum(
        _as_vector(batch.boundary_distance, n, default=np.inf, name="boundary_distance"),
        0.0,
    )
    flow_inconsistency = np.maximum(
        _as_vector(batch.flow_inconsistency, n, default=0.0, name="flow_inconsistency"),
        0.0,
    )

    if cfg.residual_scale is None:
        residual_weight = np.ones(n, dtype=float)
    else:
        residual_weight = np.exp(-0.5 * np.square(residual_norm / cfg.residual_scale))
    confidence_weight = np.power(confidence, cfg.confidence_power)
    boundary_weight = 1.0 - np.exp(-boundary_distance / cfg.boundary_scale)
    flow_weight = np.exp(-flow_inconsistency / cfg.flow_scale)
    occlusion_weight = np.where(occluded, cfg.occlusion_weight, 1.0)

    weights = residual_weight * confidence_weight * boundary_weight * flow_weight
    weights *= occlusion_weight
    weights = np.clip(weights, cfg.min_weight, 1.0)

    base_variance = measurement_variance(batch)
    inflation = 1.0 / weights
    inflation = np.minimum(inflation, cfg.covariance_inflation_at_min_weight)
    inflated_variance = base_variance * inflation[:, None]

    return ReliabilityResult(
        weights=weights,
        inflated_variance=inflated_variance,
        residual_norm=residual_norm,
    )


def reliability_weighted_loss(
    batch: PseudoMeasurementBatch,
    config: ReliabilityConfig | None = None,
) -> float:
    """Return a scalar weighted squared residual objective."""

    observed, predicted = batch.arrays()
    result = score_reliability(batch, config)
    residual = observed - predicted
    variance = np.maximum(result.inflated_variance, 1e-12)
    normalized_sq = np.sum(np.square(residual) / variance, axis=1)
    return float(np.mean(normalized_sq))
