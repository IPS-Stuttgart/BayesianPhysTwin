"""Prospective fail-closed boundary for the recursive RBF belief v1 implementation.

The registered v1 implementation underpins recorded evidence and must remain
byte-for-byte stable. This module rejects lossy availability-mask coercion and
passes a private writable copy to v1 so caller-owned or read-only masks cannot
be mutated by the legacy finite-row filtering step. Valid Boolean inputs retain
v1 numerical behavior exactly.
"""

from __future__ import annotations

import numpy as np

from .phystwin_online_belief import (
    BeliefFieldPrediction,
    RecursiveRbfBeliefConfig,
    RecursiveRbfBeliefSnapshot,
    decode_recursive_rbf_belief,
    deterministic_farthest_point_ids,
    finite_sample_absolute_residual_quantile_m as _finite_sample_v1,
    initialize_recursive_rbf_belief,
    robust_huber_continuation_gain,
    update_recursive_rbf_belief as _update_v1,
)


def _strict_boolean_array(value: object, *, name: str) -> np.ndarray:
    """Return an owned writable Boolean array without truthiness coercion."""

    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must contain only booleans")
    return np.array(raw, dtype=np.bool_, copy=True, order="C")


def finite_sample_absolute_residual_quantile_m(
    measured_residual_m: np.ndarray,
    available: np.ndarray,
    nominal_coverage: float,
) -> float:
    """Run the registered v1 quantile after strict mask admission.

    A private copy is required because v1 removes non-finite residual rows with
    an in-place mask operation. This keeps that registered behavior while making
    the prospective public boundary non-mutating and read-only safe.
    """

    mask = _strict_boolean_array(available, name="available")
    return _finite_sample_v1(
        measured_residual_m,
        mask,
        nominal_coverage,
    )


def update_recursive_rbf_belief(
    prior: RecursiveRbfBeliefSnapshot,
    frame_index: int,
    center_positions_m: np.ndarray,
    measured_residual_m: np.ndarray,
    available: np.ndarray,
    *,
    config: RecursiveRbfBeliefConfig,
) -> tuple[RecursiveRbfBeliefSnapshot, np.ndarray]:
    """Run the registered v1 recursive update after strict mask admission."""

    mask = _strict_boolean_array(available, name="available")
    return _update_v1(
        prior,
        frame_index,
        center_positions_m,
        measured_residual_m,
        mask,
        config=config,
    )


__all__ = [
    "BeliefFieldPrediction",
    "RecursiveRbfBeliefConfig",
    "RecursiveRbfBeliefSnapshot",
    "decode_recursive_rbf_belief",
    "deterministic_farthest_point_ids",
    "finite_sample_absolute_residual_quantile_m",
    "initialize_recursive_rbf_belief",
    "robust_huber_continuation_gain",
    "update_recursive_rbf_belief",
]
