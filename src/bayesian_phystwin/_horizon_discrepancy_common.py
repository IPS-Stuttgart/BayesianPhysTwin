"""Shared validation and identity helpers for horizon discrepancy contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np

HORIZON_DISCREPANCY_CALIBRATION_SCHEMA = (
    "bayesian-phystwin.horizon-discrepancy-calibration"
)
HORIZON_DISCREPANCY_CALIBRATION_VERSION = 1
HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS = (
    "source-group-selected-mean-reversion-and-process-growth-v1"
)

CALIBRATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "artifact_id",
        "source_group_ids",
        "source_summary_sha256",
        "horizon_steps",
        "mean_reversion_half_life_steps",
        "minimum_mean_retention",
        "stationary_std_m",
        "additional_process_std_m_per_sqrt_step",
        "component_process_variance_scale",
        "source_outcomes_used",
        "interval_calibration_outcomes_used",
        "confirmation_outcomes_used",
        "target_outcomes_used",
        "metadata",
    }
)


def readonly(value: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def finite_real(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number >= {minimum}")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite real number >= {minimum}")
    return result


def probability(value: object, *, name: str) -> float:
    result = finite_real(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def axis_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite nonnegative length-3 vector")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be a finite nonnegative length-3 vector")
    return readonly(result)


def horizon_vector(value: object, *, allow_zero: bool) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError("horizon_steps must contain genuine integers")
    result = np.asarray(raw, dtype=np.int64)
    minimum = 0 if allow_zero else 1
    if (
        result.ndim != 1
        or len(result) < 1
        or np.any(result < minimum)
        or np.any(np.diff(result) <= 0)
    ):
        boundary = "nonnegative" if allow_zero else "positive"
        raise ValueError(
            f"horizon_steps must be a strictly increasing {boundary} integer vector"
        )
    return readonly(result, dtype=np.int64)


def source_summary_id(
    group_ids: Sequence[str],
    horizons: np.ndarray,
    endpoint: np.ndarray,
    future: np.ndarray,
) -> str:
    def array_digest(value: np.ndarray) -> str:
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.view(np.uint8))
        return digest.hexdigest()

    payload = {
        "schema": "bayesian-phystwin.horizon-discrepancy-source-summary",
        "schema_version": 1,
        "source_group_ids": list(group_ids),
        "horizon_steps": [int(value) for value in horizons],
        "endpoint_mean_sha256": array_digest(endpoint),
        "future_mean_sha256": array_digest(future),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
