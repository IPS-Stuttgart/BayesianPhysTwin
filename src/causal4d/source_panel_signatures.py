"""Predeclared effect-size diagnostics for the 12-run source panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _residual_batch(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 4 or result.shape[3] != 3:
        raise ValueError(f"{name} must have shape (repeat, frame, node, 3)")
    if result.shape[0] < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain at least two finite repeats")
    return result


def _positive_scale(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).reshape(-1)
    right = np.asarray(second, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0.0 else 0.0


def _consistent_direction_count(effects: np.ndarray) -> int:
    flattened = np.asarray(effects, dtype=float).reshape(len(effects), -1)
    reference = np.mean(flattened, axis=0)
    if np.linalg.norm(reference) == 0.0:
        return 0
    return int(np.sum(flattened @ reference > 0.0))


def estimate_repeatability_floor(
    residuals_by_profile: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Estimate an equal-profile coordinate RMS floor from exact repeats."""

    if not residuals_by_profile:
        raise ValueError("at least one repeated profile is required")
    profile_variances = {}
    for profile, values in residuals_by_profile.items():
        residual = _residual_batch(values, f"residuals_by_profile[{profile!r}]")
        centered = residual - np.mean(residual, axis=0, keepdims=True)
        profile_variances[str(profile)] = float(np.mean(np.square(centered)))
    variance = float(np.mean(list(profile_variances.values())))
    return {
        "definition": "equal-profile between-repeat coordinate RMS",
        "sigma_repeat_m": float(np.sqrt(variance)),
        "profile_sigma_m": {
            profile: float(np.sqrt(value))
            for profile, value in profile_variances.items()
        },
        "profile_count": len(profile_variances),
        "replication_unit": "fresh_reset_session",
    }


def reversal_sign_flip_signature(
    lift_residual_m: np.ndarray,
    lower_residual_m: np.ndarray,
    *,
    common_action_axis: Sequence[float],
    sigma_repeat_m: float,
    minimum_sign_flip_cosine: float = 0.50,
    minimum_odd_rms_ratio: float = 1.50,
    minimum_consistent_pairs: int = 2,
) -> dict[str, Any]:
    """Score reset-separated direction reversal in a common world/controller axis."""

    lift = _residual_batch(lift_residual_m, "lift_residual_m")
    lower = _residual_batch(lower_residual_m, "lower_residual_m")
    if lift.shape != lower.shape:
        raise ValueError("lift and lower residual batches must match")
    sigma = _positive_scale(sigma_repeat_m, "sigma_repeat_m")
    axis = np.asarray(common_action_axis, dtype=float).reshape(-1)
    if (
        axis.shape != (3,)
        or not np.all(np.isfinite(axis))
        or np.linalg.norm(axis) == 0.0
    ):
        raise ValueError("common_action_axis must be a finite nonzero 3-vector")
    axis /= np.linalg.norm(axis)
    lift_axis = np.einsum("rtnd,d->rtn", lift, axis)
    lower_axis = np.einsum("rtnd,d->rtn", lower, axis)
    pair_scores = np.array(
        [
            -_cosine(first, second)
            for first, second in zip(lift_axis, lower_axis, strict=True)
        ]
    )
    odd_component = 0.5 * (lift_axis - lower_axis)
    odd_rms_m = float(np.sqrt(np.mean(np.square(odd_component))))
    consistent = int(np.sum(pair_scores > 0.0))
    score = float(np.mean(pair_scores))
    ratio = odd_rms_m / sigma
    return {
        "statistic": "reset-separated action-axis sign flip",
        "sign_flip_cosine": score,
        "pair_sign_flip_cosines": pair_scores.tolist(),
        "odd_component_rms_m": odd_rms_m,
        "odd_component_rms_over_sigma_repeat": ratio,
        "consistent_pair_count": consistent,
        "pair_count": len(pair_scores),
        "thresholds": {
            "minimum_sign_flip_cosine": minimum_sign_flip_cosine,
            "minimum_odd_rms_ratio": minimum_odd_rms_ratio,
            "minimum_consistent_pairs": minimum_consistent_pairs,
        },
        "eligible": bool(
            score >= minimum_sign_flip_cosine
            and ratio >= minimum_odd_rms_ratio
            and consistent >= minimum_consistent_pairs
        ),
        "confirmatory_claim": False,
    }


def continuous_nonclosure_signature(
    residual_m: np.ndarray,
    *,
    pre_action_frames: Sequence[int],
    post_return_frames: Sequence[int],
    sigma_repeat_m: float,
    minimum_rms_ratio: float = 1.50,
    minimum_consistent_repetitions: int = 2,
) -> dict[str, Any]:
    """Measure path-dependent residual non-closure within out-and-return runs."""

    residual = _residual_batch(residual_m, "residual_m")
    pre = np.asarray(pre_action_frames, dtype=int).reshape(-1)
    post = np.asarray(post_return_frames, dtype=int).reshape(-1)
    if len(pre) == 0 or len(post) < 3:
        raise ValueError(
            "non-closure needs pre-action and at least three post-return frames"
        )
    if (
        np.any(pre < 0)
        or np.any(post < 0)
        or np.any(pre >= residual.shape[1])
        or np.any(post >= residual.shape[1])
    ):
        raise ValueError("non-closure frame index is out of range")
    sigma = _positive_scale(sigma_repeat_m, "sigma_repeat_m")
    effect = np.mean(residual[:, post], axis=1) - np.mean(residual[:, pre], axis=1)
    effect_rms_m = float(np.sqrt(np.mean(np.square(effect))))
    ratio = effect_rms_m / sigma
    consistent = _consistent_direction_count(effect)
    return {
        "statistic": "continuous out-and-return residual non-closure",
        "nonclosure_rms_m": effect_rms_m,
        "nonclosure_rms_over_sigma_repeat": ratio,
        "consistent_repetition_count": consistent,
        "repetition_count": len(effect),
        "post_return_frame_count": len(post),
        "eligible": bool(
            ratio >= minimum_rms_ratio
            and consistent >= minimum_consistent_repetitions
            and len(post) >= 3
        ),
        "confirmatory_claim": False,
    }


def speed_signature(
    fast_residual_m: np.ndarray,
    slow_residual_m: np.ndarray,
    *,
    measured_fast_peak_speed_mps: Sequence[float],
    measured_slow_peak_speed_mps: Sequence[float],
    sigma_repeat_m: float,
    minimum_rms_ratio: float = 1.50,
    required_speed_ratio: tuple[float, float] = (0.35, 0.65),
    minimum_consistent_repetitions: int = 2,
) -> dict[str, Any]:
    """Score a phase-aligned speed effect at fixed amplitude and direction."""

    fast = _residual_batch(fast_residual_m, "fast_residual_m")
    slow = _residual_batch(slow_residual_m, "slow_residual_m")
    if fast.shape != slow.shape:
        raise ValueError("phase-aligned fast and slow residuals must match")
    sigma = _positive_scale(sigma_repeat_m, "sigma_repeat_m")
    fast_speed = np.asarray(measured_fast_peak_speed_mps, dtype=float).reshape(-1)
    slow_speed = np.asarray(measured_slow_peak_speed_mps, dtype=float).reshape(-1)
    if len(fast_speed) != len(fast) or len(slow_speed) != len(fast):
        raise ValueError("speed vectors must match repeats")
    if np.any(fast_speed <= 0.0) or np.any(slow_speed <= 0.0):
        raise ValueError("measured speeds must be positive")
    ratios = slow_speed / fast_speed
    ratio_valid = bool(
        np.all(
            (ratios >= required_speed_ratio[0]) & (ratios <= required_speed_ratio[1])
        )
    )
    effect = fast - slow
    effect_rms_m = float(np.sqrt(np.mean(np.square(effect))))
    standardized = effect_rms_m / sigma
    consistent = _consistent_direction_count(effect)
    return {
        "statistic": "phase-aligned residual change at fixed-amplitude speed contrast",
        "effect_rms_m": effect_rms_m,
        "effect_rms_over_sigma_repeat": standardized,
        "measured_slow_to_fast_speed_ratios": ratios.tolist(),
        "speed_ratio_valid": ratio_valid,
        "consistent_repetition_count": consistent,
        "repetition_count": len(effect),
        "eligible": bool(
            standardized >= minimum_rms_ratio
            and consistent >= minimum_consistent_repetitions
            and ratio_valid
        ),
        "confirmatory_claim": False,
    }


def hold_relaxation_signature(
    hold_residual_m: np.ndarray,
    *,
    frame_dt_s: float,
    sigma_repeat_m: float,
    minimum_amplitude_ratio: float = 1.50,
    minimum_log_r_squared: float = 0.80,
    observable_time_constant_s: tuple[float, float] = (0.0667, 0.50),
    minimum_consistent_repetitions: int = 2,
) -> dict[str, Any]:
    """Fit one source-only relaxation direction during the long hold."""

    residual = _residual_batch(hold_residual_m, "hold_residual_m")
    if residual.shape[1] < 7:
        raise ValueError("hold relaxation needs at least seven frames")
    if frame_dt_s <= 0.0 or not np.isfinite(frame_dt_s):
        raise ValueError("frame_dt_s must be positive and finite")
    sigma = _positive_scale(sigma_repeat_m, "sigma_repeat_m")
    early = np.mean(residual[:, :3], axis=1)
    late = np.mean(residual[:, -3:], axis=1)
    effect = early - late
    amplitude_m = float(np.sqrt(np.mean(np.square(effect))))
    amplitude_ratio = amplitude_m / sigma
    consistent = _consistent_direction_count(effect)
    direction = np.mean(effect, axis=0).reshape(-1)
    direction_norm = float(direction @ direction)
    mean_trace = np.mean(residual, axis=0)
    tail = np.mean(mean_trace[-3:], axis=0)
    scalar = (
        (mean_trace - tail).reshape(len(mean_trace), -1) @ direction / direction_norm
        if direction_norm > 0.0
        else np.zeros(len(mean_trace))
    )
    selected = np.flatnonzero(scalar > max(0.05 * scalar[0], 1.0e-12))
    time_constant_s = None
    r_squared = None
    if len(selected) >= 3:
        times = frame_dt_s * selected
        values = np.log(scalar[selected])
        slope, intercept = np.polyfit(times, values, 1)
        if slope < 0.0:
            fitted = intercept + slope * times
            residual_sum = float(np.sum(np.square(values - fitted)))
            total_sum = float(np.sum(np.square(values - np.mean(values))))
            r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
            time_constant_s = float(-1.0 / slope)
    time_valid = bool(
        time_constant_s is not None
        and observable_time_constant_s[0]
        <= time_constant_s
        <= observable_time_constant_s[1]
    )
    return {
        "statistic": "long-hold directional exponential relaxation",
        "relaxation_amplitude_m": amplitude_m,
        "relaxation_amplitude_over_sigma_repeat": amplitude_ratio,
        "consistent_repetition_count": consistent,
        "repetition_count": len(effect),
        "time_constant_s": time_constant_s,
        "log_r_squared": r_squared,
        "time_constant_observable": time_valid,
        "eligible": bool(
            amplitude_ratio >= minimum_amplitude_ratio
            and consistent >= minimum_consistent_repetitions
            and r_squared is not None
            and r_squared >= minimum_log_r_squared
            and time_valid
        ),
        "confirmatory_claim": False,
    }


def heldout_mechanism_eligibility(
    baseline_correction_rms_m: Sequence[float],
    mechanism_correction_rms_m: Sequence[float],
    *,
    track_gain_m: Sequence[float],
    late_track_gain_m: Sequence[float],
    cd_degradation_m: Sequence[float],
    track_repeatability_sd_m: float,
    late_track_repeatability_sd_m: float,
    cd_repeatability_sd_m: float,
    minimum_shrinkage_fraction: float = 0.10,
    minimum_positive_sessions: int = 8,
) -> dict[str, Any]:
    """Apply v3 eligibility gates to cross-fitted held-out source predictions."""

    baseline = np.asarray(baseline_correction_rms_m, dtype=float).reshape(-1)
    mechanism = np.asarray(mechanism_correction_rms_m, dtype=float).reshape(-1)
    track = np.asarray(track_gain_m, dtype=float).reshape(-1)
    late = np.asarray(late_track_gain_m, dtype=float).reshape(-1)
    cd = np.asarray(cd_degradation_m, dtype=float).reshape(-1)
    if not (
        len(baseline) == len(mechanism) == len(track) == len(late) == len(cd) == 12
    ):
        raise ValueError("eligibility requires all 12 held-out source sessions")
    if (
        np.any(baseline <= 0.0)
        or np.any(mechanism <= 0.0)
        or not all(
            np.all(np.isfinite(value))
            for value in (baseline, mechanism, track, late, cd)
        )
    ):
        raise ValueError(
            "eligibility inputs must be finite with positive correction RMS"
        )
    track_scale = _positive_scale(track_repeatability_sd_m, "track_repeatability_sd_m")
    late_scale = _positive_scale(
        late_track_repeatability_sd_m, "late_track_repeatability_sd_m"
    )
    cd_scale = _positive_scale(cd_repeatability_sd_m, "cd_repeatability_sd_m")
    geometric_ratio = float(np.exp(np.mean(np.log(mechanism / baseline))))
    shrinkage = 1.0 - geometric_ratio
    positive_sessions = int(np.sum(mechanism < baseline))
    standardized_track = float(np.mean(track) / track_scale)
    standardized_late = float(np.mean(late) / late_scale)
    standardized_cd = float(np.mean(cd) / cd_scale)
    gates = {
        "shrinkage": shrinkage >= minimum_shrinkage_fraction,
        "session_direction": positive_sessions >= minimum_positive_sessions,
        "track": standardized_track >= 1.0,
        "late_track": standardized_late >= 1.0,
        "cd_non_degradation": standardized_cd <= 0.5,
    }
    return {
        "geometric_mean_correction_ratio": geometric_ratio,
        "shrinkage_fraction": shrinkage,
        "positive_shrinkage_session_count": positive_sessions,
        "session_count": 12,
        "mean_track_gain_over_repeatability_sd": standardized_track,
        "mean_late_track_gain_over_repeatability_sd": standardized_late,
        "mean_cd_degradation_over_repeatability_sd": standardized_cd,
        "gates": gates,
        "eligible_for_confirmatory_evaluation": bool(all(gates.values())),
        "confirmatory_claim": False,
    }
