"""Post-open diagnostics for action-conditioned cloth discrepancy.

This module does not define a confirmatory method. It tests whether a spatial
readout correction inferred from a causal prefix should remain fixed or follow
a repeatable phase-dependent scalar or global-translation profile.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def projected_residual_scale(
    correction_m: np.ndarray,
    physical_points_m: np.ndarray,
    associated_observation_m: np.ndarray,
) -> float:
    """Project one aligned residual field onto a prefix correction."""

    correction = np.asarray(correction_m, dtype=np.float64)
    physical = np.asarray(physical_points_m, dtype=np.float64)
    observed = np.asarray(associated_observation_m, dtype=np.float64)
    _require(
        correction.ndim == 2 and correction.shape[1] == 3,
        "correction_m must have shape (N, 3)",
    )
    _require(
        physical.shape == observed.shape == correction.shape,
        "physical, observation, and correction shapes differ",
    )
    _require(
        np.all(np.isfinite(correction))
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(observed)),
        "projected residual inputs must be finite",
    )
    energy = float(np.sum(np.square(correction)))
    _require(energy > 0.0, "correction_m must have positive energy")
    return float(np.sum(correction * (observed - physical)) / energy)


def _resample_profile(profile: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    _require(
        values.ndim == 1 and len(values) >= 1 and np.all(np.isfinite(values)),
        "each profile must be a finite nonempty vector",
    )
    _require(target_length >= 1, "target_length must be positive")
    if len(values) == target_length:
        return values.copy()
    return np.interp(
        np.linspace(0.0, 1.0, target_length),
        np.linspace(0.0, 1.0, len(values)),
        values,
    )


def fit_action_phase_profile(
    training_profiles: Sequence[np.ndarray],
    *,
    target_length: int,
    smoothing_window: int = 9,
    maximum_absolute_scale: float = 1.5,
) -> np.ndarray:
    """Average, smooth, and clip profiles from disjoint opened repeats."""

    _require(len(training_profiles) >= 1, "at least one training profile is required")
    _require(
        smoothing_window >= 1 and smoothing_window % 2 == 1,
        "smoothing_window must be a positive odd integer",
    )
    _require(
        np.isfinite(maximum_absolute_scale) and maximum_absolute_scale > 0.0,
        "maximum_absolute_scale must be positive and finite",
    )
    resampled = np.stack(
        [_resample_profile(profile, target_length) for profile in training_profiles]
    )
    mean_profile = np.mean(resampled, axis=0)
    radius = smoothing_window // 2
    padded = np.pad(mean_profile, (radius, radius), mode="edge")
    smoothed = np.convolve(
        padded,
        np.ones(smoothing_window, dtype=np.float64) / smoothing_window,
        mode="valid",
    )
    result = np.clip(
        smoothed,
        -maximum_absolute_scale,
        maximum_absolute_scale,
    )
    result.setflags(write=False)
    return result


def fit_action_phase_translation_delta(
    training_residual_fields_m: Sequence[np.ndarray],
    training_prefix_corrections_m: Sequence[np.ndarray],
    *,
    target_length: int,
    smoothing_window: int = 9,
    maximum_translation_m: float = 0.10,
) -> np.ndarray:
    """Fit a robust global-translation change around each prefix anchor."""

    _require(
        len(training_residual_fields_m) == len(training_prefix_corrections_m) >= 1,
        "residual fields and prefix corrections must be nonempty and paired",
    )
    _require(
        np.isfinite(maximum_translation_m) and maximum_translation_m > 0.0,
        "maximum_translation_m must be positive and finite",
    )
    translation_profiles = []
    for residual_field_m, prefix_correction_m in zip(
        training_residual_fields_m,
        training_prefix_corrections_m,
        strict=True,
    ):
        residual = np.asarray(residual_field_m, dtype=np.float64)
        correction = np.asarray(prefix_correction_m, dtype=np.float64)
        _require(
            residual.ndim == 3
            and residual.shape[2] == 3
            and correction.shape == residual.shape[1:],
            "training residual/correction shapes are invalid",
        )
        _require(
            np.all(np.isfinite(residual)) and np.all(np.isfinite(correction)),
            "training residuals and corrections must be finite",
        )
        translation_profiles.append(
            np.median(residual, axis=1) - np.median(correction, axis=0)[None]
        )
    result = np.column_stack(
        [
            fit_action_phase_profile(
                [profile[:, coordinate] for profile in translation_profiles],
                target_length=target_length,
                smoothing_window=smoothing_window,
                maximum_absolute_scale=maximum_translation_m,
            )
            for coordinate in range(3)
        ]
    )
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    result *= np.minimum(
        1.0,
        maximum_translation_m / np.maximum(norm, 1e-15),
    )
    result.setflags(write=False)
    return result


def apply_action_phase_correction(
    physical_future_m: np.ndarray,
    correction_m: np.ndarray,
    scale_by_frame: np.ndarray,
) -> np.ndarray:
    """Apply one spatial field with a scalar amplitude at each future frame."""

    physical = np.asarray(physical_future_m)
    correction = np.asarray(correction_m, dtype=physical.dtype)
    scale = np.asarray(scale_by_frame, dtype=physical.dtype)
    _require(
        physical.ndim == 3 and physical.shape[2] == 3,
        "physical_future_m must have shape (T, N, 3)",
    )
    _require(
        correction.shape == physical.shape[1:],
        "correction_m shape differs from the physical nodes",
    )
    _require(
        scale.shape == (len(physical),),
        "scale_by_frame must have one value per future frame",
    )
    _require(
        np.all(np.isfinite(physical))
        and np.all(np.isfinite(correction))
        and np.all(np.isfinite(scale)),
        "action-phase correction inputs must be finite",
    )
    if not np.any(correction):
        return physical.copy()
    return physical + scale[:, None, None] * correction[None, :, :]


def apply_action_phase_translation_delta(
    physical_future_m: np.ndarray,
    prefix_correction_m: np.ndarray,
    translation_delta_by_frame_m: np.ndarray,
    *,
    maximum_correction_m: float = 0.10,
) -> np.ndarray:
    """Add a learned global translation change around a held prefix field."""

    physical = np.asarray(physical_future_m)
    correction = np.asarray(prefix_correction_m, dtype=physical.dtype)
    translation = np.asarray(
        translation_delta_by_frame_m,
        dtype=physical.dtype,
    )
    _require(
        physical.ndim == 3 and physical.shape[2] == 3,
        "physical_future_m must have shape (T, N, 3)",
    )
    _require(
        correction.shape == physical.shape[1:],
        "prefix_correction_m shape differs from the physical nodes",
    )
    _require(
        translation.shape == (len(physical), 3),
        "translation delta must have shape (T, 3)",
    )
    _require(
        np.isfinite(maximum_correction_m) and maximum_correction_m > 0.0,
        "maximum_correction_m must be positive and finite",
    )
    _require(
        np.all(np.isfinite(physical))
        and np.all(np.isfinite(correction))
        and np.all(np.isfinite(translation)),
        "translation-delta correction inputs must be finite",
    )
    total_correction = correction[None] + translation[:, None, :]
    norm = np.linalg.norm(total_correction, axis=2, keepdims=True)
    total_correction *= np.minimum(
        1.0,
        maximum_correction_m / np.maximum(norm, 1e-15),
    )
    if not np.any(total_correction):
        return physical.copy()
    return physical + total_correction
