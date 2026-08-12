"""Exact randomization and deterministic bootstrap for group inference v1."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np

from ._independent_group_inference_common import (
    BOOTSTRAP_CHUNK_SIZE,
    SIGN_FLIP_COMPARISON_EPSILON_MULTIPLIER,
    FloatArray,
    sha256_array,
)


def _comparison_tolerance(observed: FloatArray) -> FloatArray:
    return (
        SIGN_FLIP_COMPARISON_EPSILON_MULTIPLIER
        * np.finfo(np.float64).eps
        * np.maximum(1.0, np.abs(observed))
    )


def exact_sign_flip_inference(
    effects: FloatArray,
    standardized_observed: FloatArray,
) -> tuple[FloatArray, FloatArray, float, int, str]:
    """Enumerate every joint group sign pattern for one estimand family."""

    group_count, estimand_count = effects.shape
    pattern_count = 1 << group_count
    unadjusted_count = np.zeros(estimand_count, dtype=np.int64)
    familywise_count = np.zeros(estimand_count, dtype=np.int64)
    global_count = 0
    tolerance = _comparison_tolerance(standardized_observed)
    global_threshold = float(np.min(standardized_observed))
    global_tolerance = float(np.max(tolerance))
    rms_scale = np.sqrt(np.mean(np.square(effects), axis=0))
    pattern_hash = hashlib.sha256()
    bit_positions = np.arange(group_count, dtype=np.uint64)[None, :]

    for start in range(0, pattern_count, BOOTSTRAP_CHUNK_SIZE):
        stop = min(pattern_count, start + BOOTSTRAP_CHUNK_SIZE)
        pattern_ids = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = (pattern_ids >> bit_positions) & np.uint64(1)
        signs_i8 = np.where(bits == 0, 1, -1).astype(np.int8, copy=False)
        pattern_hash.update(np.asarray(signs_i8, dtype=np.dtype("<i1")).tobytes())
        signs = signs_i8.astype(np.float64)
        signed_means = (signs @ effects) / float(group_count)
        statistics = np.zeros_like(signed_means)
        positive_scale = rms_scale > 0.0
        statistics[:, positive_scale] = (
            signed_means[:, positive_scale] / rms_scale[positive_scale]
        )
        unadjusted_count += np.sum(
            statistics <= standardized_observed[None, :] + tolerance[None, :],
            axis=0,
            dtype=np.int64,
        )
        minimum_statistics = np.min(statistics, axis=1)
        familywise_count += np.sum(
            minimum_statistics[:, None]
            <= standardized_observed[None, :] + tolerance[None, :],
            axis=0,
            dtype=np.int64,
        )
        global_count += int(
            np.sum(minimum_statistics <= global_threshold + global_tolerance)
        )

    denominator = float(pattern_count)
    return (
        np.asarray(unadjusted_count, dtype=np.float64) / denominator,
        np.asarray(familywise_count, dtype=np.float64) / denominator,
        global_count / denominator,
        pattern_count,
        pattern_hash.hexdigest(),
    )


def _quantile(
    values: FloatArray,
    probabilities: FloatArray,
    *,
    axis: int,
    method: Literal["linear", "higher"],
) -> FloatArray:
    return np.asarray(
        np.quantile(values, probabilities, axis=axis, method=method),
        dtype=np.float64,
    )


def bootstrap_inference(
    effects: FloatArray,
    observed_mean: FloatArray,
    standard_error: FloatArray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    str,
    str,
]:
    """Run one shared-index paired group bootstrap across all estimands."""

    group_count, estimand_count = effects.shape
    bootstrap_means = np.empty((replicates, estimand_count), dtype=np.float64)
    index_hash = hashlib.sha256()
    generator = np.random.Generator(np.random.PCG64(seed))

    for start in range(0, replicates, BOOTSTRAP_CHUNK_SIZE):
        stop = min(replicates, start + BOOTSTRAP_CHUNK_SIZE)
        indices = generator.integers(
            0,
            group_count,
            size=(stop - start, group_count),
            dtype=np.int64,
            endpoint=False,
        )
        index_hash.update(
            np.asarray(indices, dtype=np.dtype("<u4"), order="C").tobytes()
        )
        bootstrap_means[start:stop] = np.mean(effects[indices], axis=1)

    mean_digest = sha256_array(bootstrap_means, dtype=np.dtype("<f8"))
    alpha = (1.0 - confidence) / 2.0
    pointwise = _quantile(
        bootstrap_means,
        np.asarray([alpha, 1.0 - alpha], dtype=np.float64),
        axis=0,
        method="linear",
    )

    centered = bootstrap_means - observed_mean[None, :]
    standardized = np.zeros_like(centered)
    positive_error = standard_error > 0.0
    standardized[:, positive_error] = (
        centered[:, positive_error] / standard_error[positive_error]
    )
    zero_error = ~positive_error
    if np.any(zero_error):
        tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * np.maximum(1.0, np.abs(observed_mean[zero_error]))
        )
        if np.any(np.abs(centered[:, zero_error]) > tolerance[None, :]):
            raise RuntimeError(
                "zero standard-error estimand changed under group bootstrap"
            )

    maximum_absolute = np.max(np.abs(standardized), axis=1)
    maximum_upper = np.max(standardized, axis=1)
    two_sided_critical = float(
        np.quantile(maximum_absolute, confidence, method="higher")
    )
    one_sided_critical = max(
        0.0,
        float(np.quantile(maximum_upper, confidence, method="higher")),
    )
    simultaneous_lower = observed_mean - two_sided_critical * standard_error
    simultaneous_upper = observed_mean + two_sided_critical * standard_error
    superiority_upper = observed_mean + one_sided_critical * standard_error
    return (
        pointwise[0],
        pointwise[1],
        simultaneous_lower,
        simultaneous_upper,
        superiority_upper,
        two_sided_critical,
        one_sided_critical,
        index_hash.hexdigest(),
        mean_digest,
    )


__all__ = ["bootstrap_inference", "exact_sign_flip_inference"]
