"""Immutable result contracts for dynamic endpoint model averaging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._dynamic_endpoint_components import DynamicEndpointModelAverageConfigV2


def _readonly(
    value: np.ndarray,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_covariance(
    covariance: np.ndarray,
    *,
    name: str,
    final_shape: tuple[int, int],
) -> None:
    if covariance.ndim < 2 or covariance.shape[-2:] != final_shape:
        raise ValueError(f"{name} has an invalid covariance shape")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2)):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues, initial=0.0) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")


def _validated_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass(frozen=True, slots=True)
class DynamicEndpointPosteriorV2:
    """Model-averaged endpoint and complete component filter states."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    final_nominal_probability: np.ndarray
    update_count: np.ndarray
    component_weights: np.ndarray
    component_log_evidence: np.ndarray
    component_state_mean: np.ndarray
    component_state_covariance: np.ndarray
    config: DynamicEndpointModelAverageConfigV2
    end_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.config, DynamicEndpointModelAverageConfigV2):
            raise TypeError("config must be a DynamicEndpointModelAverageConfigV2")
        if isinstance(self.end_frame, (bool, np.bool_)) or not isinstance(
            self.end_frame,
            (int, np.integer),
        ):
            raise ValueError("end_frame must be a positive integer")
        end_frame = int(self.end_frame)
        if end_frame < 1:
            raise ValueError("end_frame must be a positive integer")
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        probability = np.asarray(self.final_nominal_probability, dtype=np.float64)
        raw_count = np.asarray(self.update_count)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        evidence = np.asarray(self.component_log_evidence, dtype=np.float64)
        state_mean = np.asarray(self.component_state_mean, dtype=np.float64)
        state_covariance = np.asarray(
            self.component_state_covariance,
            dtype=np.float64,
        )
        if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        track_count = len(mean)
        component_count = len(self.config.components)
        if covariance.shape != (track_count, 3, 3):
            raise ValueError("covariance_m2 must have shape (N, 3, 3)")
        _validate_covariance(
            covariance,
            name="covariance_m2",
            final_shape=(3, 3),
        )
        if probability.shape != (track_count,):
            raise ValueError("final_nominal_probability shape changed")
        if not np.issubdtype(raw_count.dtype, np.integer):
            raise ValueError("update_count must contain integers")
        count = np.asarray(raw_count, dtype=np.int64)
        if count.shape != (track_count,) or np.any(count < 0):
            raise ValueError("update_count must be a nonnegative track vector")
        expected = (track_count, component_count)
        if weights.shape != expected or evidence.shape != expected:
            raise ValueError("component weight/evidence shape changed")
        if state_mean.shape != (component_count, track_count, 2, 3):
            raise ValueError("component_state_mean shape changed")
        if state_covariance.shape != (component_count, track_count, 2, 2):
            raise ValueError("component_state_covariance shape changed")
        _validate_covariance(
            state_covariance,
            name="component_state_covariance",
            final_shape=(2, 2),
        )
        finite = (mean, probability, weights, evidence, state_mean)
        if not all(np.all(np.isfinite(value)) for value in finite):
            raise ValueError("dynamic endpoint posterior contains non-finite values")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("final_nominal_probability must lie in [0, 1]")
        if np.any(weights < 0.0) or not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("component_weights must be row-normalized")
        for name, value, dtype in (
            ("mean_m", mean, np.float64),
            ("covariance_m2", covariance, np.float64),
            ("final_nominal_probability", probability, np.float64),
            ("update_count", count, np.int64),
            ("component_weights", weights, np.float64),
            ("component_log_evidence", evidence, np.float64),
            ("component_state_mean", state_mean, np.float64),
            ("component_state_covariance", state_covariance, np.float64),
        ):
            object.__setattr__(self, name, _readonly(value, dtype=dtype))
        object.__setattr__(self, "end_frame", end_frame)

    @property
    def updated_mask(self) -> np.ndarray:
        result = self.update_count > 0
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class DynamicEndpointPredictionV2:
    """Horizon-dependent mixture moments and component diagnostics."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    component_weights: np.ndarray
    component_mean_m: np.ndarray
    component_variance_m2: np.ndarray
    component_velocity_mean_m_per_step: np.ndarray
    horizon_steps: int

    def __post_init__(self) -> None:
        horizon_steps = _validated_nonnegative_integer(
            self.horizon_steps,
            name="horizon_steps",
        )
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        component_mean = np.asarray(self.component_mean_m, dtype=np.float64)
        component_variance = np.asarray(self.component_variance_m2, dtype=np.float64)
        velocity = np.asarray(
            self.component_velocity_mean_m_per_step,
            dtype=np.float64,
        )
        if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        track_count = len(mean)
        if covariance.shape != (track_count, 3, 3):
            raise ValueError("covariance_m2 must have shape (N, 3, 3)")
        _validate_covariance(
            covariance,
            name="covariance_m2",
            final_shape=(3, 3),
        )
        if weights.ndim != 2 or weights.shape[0] != track_count:
            raise ValueError("component_weights shape changed")
        component_count = weights.shape[1]
        expected_mean = (component_count, track_count, 3)
        if component_mean.shape != expected_mean or velocity.shape != expected_mean:
            raise ValueError("component mean/velocity shape changed")
        if component_variance.shape != (component_count, track_count):
            raise ValueError("component_variance_m2 shape changed")
        finite = (mean, weights, component_mean, component_variance, velocity)
        if not all(np.all(np.isfinite(value)) for value in finite):
            raise ValueError("dynamic endpoint prediction contains non-finite values")
        if np.any(component_variance < 0.0):
            raise ValueError("component_variance_m2 must be nonnegative")
        if np.any(weights < 0.0) or not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("component_weights must be row-normalized")
        object.__setattr__(self, "mean_m", _readonly(mean))
        object.__setattr__(self, "covariance_m2", _readonly(covariance))
        object.__setattr__(self, "component_weights", _readonly(weights))
        object.__setattr__(self, "component_mean_m", _readonly(component_mean))
        object.__setattr__(
            self,
            "component_variance_m2",
            _readonly(component_variance),
        )
        object.__setattr__(
            self,
            "component_velocity_mean_m_per_step",
            _readonly(velocity),
        )
        object.__setattr__(self, "horizon_steps", horizon_steps)


__all__ = [
    "DynamicEndpointPosteriorV2",
    "DynamicEndpointPredictionV2",
]
