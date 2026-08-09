"""Numerical PIT construction for simulation-based calibration."""

from __future__ import annotations

import numpy as np

from ._canonical_contracts import immutable_array


def finite_float_array(
    values: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    """Validate a nonempty finite real array with an exact dimension count."""

    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def finite_scalar(value: object, *, name: str) -> float:
    """Validate one finite real scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def normalized_weights(
    values: object | None,
    *,
    draw_count: int,
    name: str,
) -> np.ndarray:
    """Return finite nonnegative posterior weights with unit total mass."""

    if values is None:
        return np.full(draw_count, 1.0 / draw_count, dtype=np.float64)
    weights = finite_float_array(values, name=name, ndim=1)
    if weights.shape != (draw_count,):
        raise ValueError(f"{name} must match the posterior draw count")
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite total mass")
    return weights / total


def weighted_randomized_pit(
    posterior_samples: object,
    truth: object,
    *,
    weights: object | None = None,
    tie_breaker: object = 0.5,
    absolute_tolerance: object = 0.0,
    relative_tolerance: object = 0.0,
) -> float:
    """Return posterior mass below truth plus randomized tied mass."""

    samples = finite_float_array(
        posterior_samples,
        name="posterior_samples",
        ndim=1,
    )
    truth_value = finite_scalar(truth, name="truth")
    tie = finite_scalar(tie_breaker, name="tie_breaker")
    if not 0.0 <= tie <= 1.0:
        raise ValueError("tie_breaker must lie in [0, 1]")
    atol = finite_scalar(absolute_tolerance, name="absolute_tolerance")
    rtol = finite_scalar(relative_tolerance, name="relative_tolerance")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("PIT tolerances must be nonnegative")

    normalized = normalized_weights(
        weights,
        draw_count=len(samples),
        name="weights",
    )
    tied = np.isclose(samples, truth_value, atol=atol, rtol=rtol)
    below = (samples < truth_value) & ~tied
    value = float(np.sum(normalized[below], dtype=np.float64))
    value += tie * float(np.sum(normalized[tied], dtype=np.float64))
    tolerance = 16.0 * np.finfo(np.float64).eps
    if value < -tolerance or value > 1.0 + tolerance:
        raise RuntimeError("randomized PIT escaped its probability range")
    return min(1.0, max(0.0, value))


def posterior_pit_matrix(
    posterior_samples: object,
    truths: object,
    *,
    weights: object | None = None,
    tie_breakers: object | None = None,
    absolute_tolerance: object = 0.0,
    relative_tolerance: object = 0.0,
) -> np.ndarray:
    """Compute immutable PIT values for replicate/draw/parameter samples."""

    samples = finite_float_array(
        posterior_samples,
        name="posterior_samples",
        ndim=3,
    )
    truth_values = finite_float_array(truths, name="truths", ndim=2)
    replicate_count, draw_count, parameter_count = samples.shape
    if truth_values.shape != (replicate_count, parameter_count):
        raise ValueError(
            "truths must match the posterior replicate and parameter axes"
        )

    weight_rows: np.ndarray | None
    if weights is None:
        weight_rows = None
    else:
        raw_weights = np.asarray(weights)
        if raw_weights.ndim == 1:
            shared = normalized_weights(
                raw_weights,
                draw_count=draw_count,
                name="weights",
            )
            weight_rows = np.repeat(shared[None, :], replicate_count, axis=0)
        elif raw_weights.shape == (replicate_count, draw_count):
            weight_rows = np.vstack(
                [
                    normalized_weights(
                        raw_weights[index],
                        draw_count=draw_count,
                        name=f"weights[{index}]",
                    )
                    for index in range(replicate_count)
                ]
            )
        else:
            raise ValueError(
                "weights must have shape (draw,) or (replicate, draw)"
            )

    if tie_breakers is None:
        tie_values = np.full(truth_values.shape, 0.5, dtype=np.float64)
    else:
        tie_values = finite_float_array(
            tie_breakers,
            name="tie_breakers",
            ndim=2,
        )
        if tie_values.shape != truth_values.shape:
            raise ValueError("tie_breakers must match truths")
        if np.any((tie_values < 0.0) | (tie_values > 1.0)):
            raise ValueError("tie_breakers must lie in [0, 1]")

    result = np.empty(truth_values.shape, dtype=np.float64)
    for replicate in range(replicate_count):
        row_weights = None if weight_rows is None else weight_rows[replicate]
        for parameter in range(parameter_count):
            result[replicate, parameter] = weighted_randomized_pit(
                samples[replicate, :, parameter],
                truth_values[replicate, parameter],
                weights=row_weights,
                tie_breaker=tie_values[replicate, parameter],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
    return immutable_array(result, dtype=np.dtype("<f8"))
