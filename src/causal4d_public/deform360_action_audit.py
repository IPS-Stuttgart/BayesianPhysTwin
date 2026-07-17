"""Source-only robot-action diagnostics for Deform360 episode protocols."""

from __future__ import annotations

from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def controller_centres(actions: np.ndarray) -> np.ndarray:
    """Return ``(frames, grippers, xyz)`` centres from Deform360 actions."""

    values = np.asarray(actions, dtype=np.float64)
    _require(np.all(np.isfinite(values)), "robot actions are non-finite")
    if values.ndim == 3:
        _require(values.shape[-1] == 3, "monomanual actions must end in xyz")
        values = values[:, None, :, :]
    _require(
        values.ndim == 4 and values.shape[-1] == 3,
        "robot actions must have shape (T,P,3) or (T,G,P,3)",
    )
    _require(
        len(values) >= 2 and values.shape[1] >= 1 and values.shape[2] >= 1,
        "robot actions contain no usable trajectory",
    )
    return np.mean(values, axis=2)


def closure_confidence(openings: np.ndarray) -> np.ndarray:
    """Return robust per-episode confidence that each gripper is closed.

    Deform360 records gripper aperture, so smaller values indicate closure.  The
    10th and 90th percentiles keep isolated encoder excursions from defining the
    scale.  A static aperture cannot identify closure; in that case we fall back
    to unweighted action path rather than pretending the action has no support.
    """

    aperture = np.asarray(openings, dtype=np.float64)
    _require(aperture.ndim == 2, "gripper openings must have shape (T,G)")
    _require(np.all(np.isfinite(aperture)), "gripper openings are non-finite")
    low = np.quantile(aperture, 0.1, axis=0)
    high = np.quantile(aperture, 0.9, axis=0)
    span = high - low
    confidence = np.ones_like(aperture)
    varying = span > 1e-9
    confidence[:, varying] = np.clip(
        (high[varying] - aperture[:, varying]) / span[varying],
        0.0,
        1.0,
    )
    return confidence


def _window_metrics(
    centres: np.ndarray,
    closed: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, Any]:
    _require(0 <= start < stop <= len(centres), "action window is out of range")
    selected = centres[start:stop]
    displacement = np.linalg.norm(selected - selected[:1], axis=-1)
    step = np.linalg.norm(np.diff(selected, axis=0), axis=-1)
    selected_closed = closed[start:stop]
    closed_step = step * np.minimum(selected_closed[:-1], selected_closed[1:])
    return {
        "frame_range_half_open": [int(start), int(stop)],
        "frame_count": int(stop - start),
        "mean_displacement_from_window_start_m": float(np.mean(displacement)),
        "maximum_displacement_from_window_start_m": float(np.max(displacement)),
        "mean_gripper_path_length_m": float(np.mean(np.sum(step, axis=0))),
        "maximum_gripper_path_length_m": float(np.max(np.sum(step, axis=0))),
        "mean_closed_weighted_path_length_m": float(
            np.mean(np.sum(closed_step, axis=0))
        ),
        "maximum_closed_weighted_path_length_m": float(
            np.max(np.sum(closed_step, axis=0))
        ),
        "mean_closure_confidence": float(np.mean(selected_closed)),
    }


def _best_window(
    centres: np.ndarray,
    closed: np.ndarray,
    window_length: int,
    metric: str,
    candidate_starts: np.ndarray,
) -> dict[str, Any]:
    _require(2 <= window_length <= len(centres), "invalid action audit window length")
    candidates = [
        _window_metrics(centres, closed, start, start + window_length)
        for start in candidate_starts
    ]
    _require(bool(candidates), "action audit has no complete candidate window")
    return max(candidates, key=lambda row: float(row[metric]))


def summarize_robot_action(
    actions: np.ndarray,
    openings: np.ndarray,
    *,
    locked_start: int,
    locked_stop: int,
    candidate_start_frame: int = 0,
    candidate_stride_frames: int = 1,
) -> dict[str, Any]:
    """Summarize full and fixed-window action support without object outcomes."""

    centres = controller_centres(actions)
    aperture = np.asarray(openings, dtype=np.float64)
    _require(np.all(np.isfinite(aperture)), "gripper openings are non-finite")
    if aperture.ndim == 1:
        aperture = aperture[:, None]
    _require(
        aperture.ndim == 2
        and aperture.shape[0] == len(centres)
        and aperture.shape[1] == centres.shape[1],
        "gripper openings do not match action groups",
    )
    _require(
        0 <= locked_start < locked_stop <= len(centres),
        "locked action window is unavailable",
    )
    window_length = locked_stop - locked_start
    _require(candidate_start_frame >= 0, "candidate start frame is negative")
    _require(candidate_stride_frames >= 1, "candidate stride must be positive")
    candidate_starts = np.arange(
        candidate_start_frame,
        len(centres) - window_length + 1,
        candidate_stride_frames,
        dtype=np.int64,
    )
    closed = closure_confidence(aperture)
    full = _window_metrics(centres, closed, 0, len(centres))
    locked = _window_metrics(centres, closed, locked_start, locked_stop)
    best_displacement = _best_window(
        centres,
        closed,
        window_length,
        "mean_displacement_from_window_start_m",
        candidate_starts,
    )
    best_path = _best_window(
        centres,
        closed,
        window_length,
        "mean_gripper_path_length_m",
        candidate_starts,
    )
    best_closed_path = _best_window(
        centres,
        closed,
        window_length,
        "mean_closed_weighted_path_length_m",
        candidate_starts,
    )
    opening_delta = np.abs(aperture - aperture[:1])
    return {
        "frame_count": int(len(centres)),
        "gripper_count": int(centres.shape[1]),
        "full_episode": full,
        "locked_window": locked,
        "candidate_start_frame": int(candidate_start_frame),
        "candidate_stride_frames": int(candidate_stride_frames),
        "candidate_count": int(len(candidate_starts)),
        "best_equal_length_displacement_window": best_displacement,
        "best_equal_length_path_window": best_path,
        "best_contact_conditioned_path_window": best_closed_path,
        "mean_opening_change_from_episode_start_m": float(np.mean(opening_delta)),
        "maximum_opening_change_from_episode_start_m": float(np.max(opening_delta)),
        "locked_to_best_displacement_ratio": float(
            locked["mean_displacement_from_window_start_m"]
            / max(
                1e-12,
                best_displacement["mean_displacement_from_window_start_m"],
            )
        ),
        "locked_to_best_path_ratio": float(
            locked["mean_gripper_path_length_m"]
            / max(
                1e-12,
                best_path["mean_gripper_path_length_m"],
            )
        ),
        "locked_to_best_contact_conditioned_path_ratio": float(
            locked["mean_closed_weighted_path_length_m"]
            / max(
                1e-12,
                best_closed_path["mean_closed_weighted_path_length_m"],
            )
        ),
    }


__all__ = ["closure_confidence", "controller_centres", "summarize_robot_action"]
