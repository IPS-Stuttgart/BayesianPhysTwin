"""Fail-closed input boundary for the frozen bias-aware v1 update."""

from __future__ import annotations

import numpy as np

from .bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    BiasAwareStateUpdateResult,
)
from .bias_aware_belief import (
    update_bias_aware_state as _update_bias_aware_state_v1,
)


def _strict_boolean_array(value: object, *, name: str) -> np.ndarray:
    """Return a copied boolean array without truthiness-based coercion."""

    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        return np.array(raw, dtype=bool, copy=True, order="C")
    if raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must contain booleans or exact 0/1 numeric values"
        )
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(
            f"{name} must contain booleans or exact 0/1 numeric values"
        )
    return np.array(numeric, dtype=bool, copy=True, order="C")


def update_bias_aware_state(
    camera_innovation_m: np.ndarray,
    camera_available: np.ndarray,
    state_basis: np.ndarray,
    shared_bias_basis: np.ndarray,
    *,
    prior_reliability: np.ndarray | None = None,
    observation_variance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_state_basis: np.ndarray | None = None,
    anchor_variance_m2: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    config: BiasAwareStateUpdateConfig | None = None,
) -> BiasAwareStateUpdateResult:
    """Run frozen v1 only after validating its public input contract."""

    if config is not None and not isinstance(config, BiasAwareStateUpdateConfig):
        raise TypeError("config must be a BiasAwareStateUpdateConfig")
    available = _strict_boolean_array(
        camera_available,
        name="camera_available",
    )
    return _update_bias_aware_state_v1(
        camera_innovation_m,
        available,
        state_basis,
        shared_bias_basis,
        prior_reliability=prior_reliability,
        observation_variance_m2=observation_variance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_state_basis=anchor_state_basis,
        anchor_variance_m2=anchor_variance_m2,
        state_prior_covariance_m2=state_prior_covariance_m2,
        config=config,
    )


__all__ = ["update_bias_aware_state"]
