"""Reliability-aware pseudo-measurement models.

The functions here are deliberately independent of PhysTwin internals. A
PhysTwin adapter only has to provide observed values, simulated predictions,
and optional confidence/occlusion/flow cues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray | list[float] | list[list[float]]


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
            boundary. Low values imply higher segmentation ambiguity. Positive
            infinity denotes no nearby boundary.
        flow_inconsistency: Optional finite nonnegative scene-flow consistency
            score. Higher values imply lower reliability.
    """

    observed: ArrayLike
    predicted: ArrayLike
    variance: ArrayLike | float = 1.0
    confidence: ArrayLike | None = None
    occluded: ArrayLike | None = None
    boundary_distance: ArrayLike | None = None
    flow_inconsistency: ArrayLike | None = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        raw_observed = np.asarray(self.observed)
        raw_predicted = np.asarray(self.predicted)
        if raw_observed.dtype.kind not in "iuf":
            raise ValueError("observed must contain real numeric values")
        if raw_predicted.dtype.kind not in "iuf":
            raise ValueError("predicted must contain real numeric values")
        observed = np.asarray(raw_observed, dtype=float)
        predicted = np.asarray(raw_predicted, dtype=float)
        if observed.shape != predicted.shape:
            raise ValueError(
                "observed and predicted must have the same shape, got "
                f"{observed.shape} and {predicted.shape}"
            )
        if observed.ndim != 2:
            raise ValueError(f"expected arrays with shape (n, d), got {observed.shape}")
        if observed.shape[0] == 0 or observed.shape[1] == 0:
            raise ValueError("observed and predicted must contain at least one value")
        if not np.all(np.isfinite(observed)):
            raise ValueError("observed must contain finite values")
        if not np.all(np.isfinite(predicted)):
            raise ValueError("predicted must contain finite values")
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

    def __post_init__(self) -> None:
        min_weight = _finite_real_scalar(self.min_weight, name="min_weight")
        confidence_power = _finite_real_scalar(
            self.confidence_power,
            name="confidence_power",
        )
        residual_scale = (
            None
            if self.residual_scale is None
            else _finite_real_scalar(self.residual_scale, name="residual_scale")
        )
        boundary_scale = _finite_real_scalar(
            self.boundary_scale,
            name="boundary_scale",
        )
        flow_scale = _finite_real_scalar(self.flow_scale, name="flow_scale")
        occlusion_weight = _finite_real_scalar(
            self.occlusion_weight,
            name="occlusion_weight",
        )
        covariance_cap = _finite_real_scalar(
            self.covariance_inflation_at_min_weight,
            name="covariance_inflation_at_min_weight",
        )

        if not 0.0 < min_weight <= 1.0:
            raise ValueError("min_weight must be in (0, 1]")
        if confidence_power < 0.0:
            raise ValueError("confidence_power must be nonnegative")
        if residual_scale is not None and residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive when provided")
        if boundary_scale <= 0.0:
            raise ValueError("boundary_scale must be positive")
        if flow_scale <= 0.0:
            raise ValueError("flow_scale must be positive")
        if not 0.0 <= occlusion_weight <= 1.0:
            raise ValueError("occlusion_weight must be in [0, 1]")
        if covariance_cap < 1.0:
            raise ValueError("covariance inflation cap must be at least 1")

        object.__setattr__(self, "min_weight", min_weight)
        object.__setattr__(self, "confidence_power", confidence_power)
        object.__setattr__(self, "residual_scale", residual_scale)
        object.__setattr__(self, "boundary_scale", boundary_scale)
        object.__setattr__(self, "flow_scale", flow_scale)
        object.__setattr__(self, "occlusion_weight", occlusion_weight)
        object.__setattr__(
            self,
            "covariance_inflation_at_min_weight",
            covariance_cap,
        )


@dataclass(frozen=True)
class ReliabilityResult:
    """Reliability scores and covariance inflation for a batch."""

    weights: np.ndarray
    inflated_variance: np.ndarray
    residual_norm: np.ndarray

    def __post_init__(self) -> None:
        weights = _owned_float_array(self.weights, name="weights")
        inflated = _owned_float_array(
            self.inflated_variance,
            name="inflated_variance",
        )
        residual = _owned_float_array(self.residual_norm, name="residual_norm")
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("weights must be a nonempty vector")
        if residual.shape != weights.shape:
            raise ValueError("residual_norm must have the same shape as weights")
        if inflated.ndim != 2 or inflated.shape[0] != weights.size:
            raise ValueError(
                "inflated_variance must be a matrix with one row per weight"
            )
        if inflated.shape[1] == 0:
            raise ValueError("inflated_variance must contain at least one coordinate")
        if not np.all(np.isfinite(weights)) or np.any(
            (weights < 0.0) | (weights > 1.0)
        ):
            raise ValueError("weights must be finite and lie in [0, 1]")
        if not np.all(np.isfinite(inflated)) or np.any(inflated <= 0.0):
            raise ValueError("inflated_variance must be finite and positive")
        if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
            raise ValueError("residual_norm must be finite and nonnegative")

        weights.setflags(write=False)
        inflated.setflags(write=False)
        residual.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "inflated_variance", inflated)
        object.__setattr__(self, "residual_norm", residual)

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
) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=np.float64)
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape == ():
        return np.full(n, float(arr.item()), dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},), got {arr.shape}")
    return np.array(arr, dtype=np.float64, copy=True, order="C")


def _as_boolean_vector(
    values: ArrayLike | None,
    n: int,
    *,
    default: bool,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=bool)
    raw = np.asarray(values)
    if raw.shape == ():
        raw = np.full(n, raw.item())
    elif raw.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},), got {raw.shape}")

    if raw.dtype.kind == "b":
        return np.array(raw, dtype=bool, copy=True, order="C")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain booleans or exact 0/1 values")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{name} must contain booleans or exact 0/1 values")
    return numeric.astype(bool)


def _variance_array(values: ArrayLike | float, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError("variance must contain real numeric values")
    arr = np.asarray(raw, dtype=float)
    n, d = shape
    if arr.shape == ():
        return np.full(shape, arr.item(), dtype=float)
    if arr.shape == (n,):
        return np.repeat(arr[:, None], d, axis=1)
    if arr.shape == shape:
        return arr
    raise ValueError(
        f"variance must be scalar, shape ({n},), or {shape}, got {arr.shape}"
    )


def measurement_variance(batch: PseudoMeasurementBatch) -> np.ndarray:
    """Return validated per-coordinate variance with shape ``(n, d)``."""

    observed, _ = batch.arrays()
    variance = _variance_array(batch.variance, observed.shape)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("variance values must be finite and positive")
    return variance


def score_reliability(
    batch: PseudoMeasurementBatch,
    config: ReliabilityConfig | None = None,
) -> ReliabilityResult:
    """Score prior reliability from learned cues and optional residual gating."""

    cfg = config if config is not None else ReliabilityConfig()
    if not isinstance(cfg, ReliabilityConfig):
        raise TypeError("config must be a ReliabilityConfig")
    observed, predicted = batch.arrays()
    with np.errstate(over="ignore", invalid="ignore"):
        residual = observed - predicted
        residual_norm = np.linalg.norm(residual, axis=1)
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual must be representable as finite values")
    if not np.all(np.isfinite(residual_norm)):
        raise ValueError("residual norm is not representable as a finite value")
    n = observed.shape[0]

    confidence = _as_vector(
        batch.confidence,
        n,
        default=1.0,
        name="confidence",
    )
    if not np.all(np.isfinite(confidence)) or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    ):
        raise ValueError("confidence must be finite and lie in [0, 1]")

    occluded = _as_boolean_vector(
        batch.occluded,
        n,
        default=False,
        name="occluded",
    )
    boundary_distance = _as_vector(
        batch.boundary_distance,
        n,
        default=np.inf,
        name="boundary_distance",
    )
    if np.any(np.isnan(boundary_distance)) or np.any(boundary_distance < 0.0):
        raise ValueError(
            "boundary_distance must be nonnegative and may contain positive infinity"
        )
    flow_inconsistency = _as_vector(
        batch.flow_inconsistency,
        n,
        default=0.0,
        name="flow_inconsistency",
    )
    if not np.all(np.isfinite(flow_inconsistency)) or np.any(flow_inconsistency < 0.0):
        raise ValueError("flow_inconsistency must be finite and nonnegative")

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
    if not np.all(np.isfinite(weights)):
        raise ValueError("reliability weights are not representable as finite values")
    weights = np.clip(weights, cfg.min_weight, 1.0)

    base_variance = measurement_variance(batch)
    inflation = 1.0 / weights
    inflation = np.minimum(inflation, cfg.covariance_inflation_at_min_weight)
    with np.errstate(over="ignore", invalid="ignore"):
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
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized_sq = np.sum(
            np.square(residual) / result.inflated_variance,
            axis=1,
        )
        loss = float(np.mean(normalized_sq))
    if not np.isfinite(loss):
        raise ValueError("weighted loss is not representable as a finite value")
    return loss
