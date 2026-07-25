"""Action-only temporal staging for the fresh bias-aware Deform360 panel."""

from __future__ import annotations

from typing import Any

import numpy as np


PREDICTION_FRAME_COUNT = 76
STAGING_FRAME_COUNT = 81
PREFIX_FRAME_COUNT = 58
CANDIDATE_FIRST_FRAME = 8
CANDIDATE_STRIDE_FRAMES = 6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def controller_centres(actions: np.ndarray) -> np.ndarray:
    """Return one XYZ centre per frame and gripper."""

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
        len(values) >= STAGING_FRAME_COUNT
        and values.shape[1] >= 1
        and values.shape[2] >= 1,
        "robot action trajectory is too short",
    )
    return np.mean(values, axis=2)


def closure_confidence(openings: np.ndarray) -> np.ndarray:
    """Map smaller aperture to a robust within-episode closure score."""

    aperture = np.asarray(openings, dtype=np.float64)
    if aperture.ndim == 1:
        aperture = aperture[:, None]
    _require(aperture.ndim == 2, "gripper openings must have shape (T,G)")
    _require(np.all(np.isfinite(aperture)), "gripper openings are non-finite")
    low = np.quantile(aperture, 0.1, axis=0)
    high = np.quantile(aperture, 0.9, axis=0)
    span = high - low
    confidence = np.ones_like(aperture)
    varying = span > 1e-9
    confidence[:, varying] = np.clip(
        (high[varying] - aperture[:, varying]) / span[varying], 0.0, 1.0
    )
    return confidence


def select_action_only_window(
    actions: np.ndarray,
    openings: np.ndarray,
) -> dict[str, Any]:
    """Select one 81-frame window without object, tactile, or outcome input."""

    centres = controller_centres(actions)
    closed = closure_confidence(openings)
    _require(closed.shape == centres.shape[:2], "opening and action groups differ")
    starts = np.arange(
        CANDIDATE_FIRST_FRAME,
        len(centres) - STAGING_FRAME_COUNT + 1,
        CANDIDATE_STRIDE_FRAMES,
        dtype=np.int64,
    )
    _require(len(starts) > 0, "episode has no complete action window")
    rows: list[tuple[float, int]] = []
    for start_value in starts:
        start = int(start_value)
        selected = centres[start : start + STAGING_FRAME_COUNT]
        selected_closed = closed[start : start + STAGING_FRAME_COUNT]
        step = np.linalg.norm(np.diff(selected, axis=0), axis=-1)
        weighted = step * np.minimum(selected_closed[:-1], selected_closed[1:])
        rows.append((float(np.mean(np.sum(weighted, axis=0))), start))
    best_score = max(score for score, _ in rows)
    selected_start = next(start for score, start in rows if score == best_score)
    return {
        "selection_rule": "maximum_mean_closed_weighted_gripper_path",
        "selected_raw_frame_range_half_open": [
            selected_start,
            selected_start + STAGING_FRAME_COUNT,
        ],
        "prediction_raw_frame_range_half_open": [
            selected_start,
            selected_start + PREDICTION_FRAME_COUNT,
        ],
        "prefix_raw_frame_range_half_open": [
            selected_start,
            selected_start + PREFIX_FRAME_COUNT,
        ],
        "candidate_first_frame": CANDIDATE_FIRST_FRAME,
        "candidate_stride_frames": CANDIDATE_STRIDE_FRAMES,
        "candidate_count": len(rows),
        "tie_break": "earliest start",
        "mean_closed_weighted_path_length_m": best_score,
        "input_fields": ["robot.actions", "robot.openings"],
        "known_future_action_is_conditioning_input": True,
        "object_geometry_read": False,
        "object_tracks_read": False,
        "tactile_read": False,
    }


__all__ = [
    "CANDIDATE_FIRST_FRAME",
    "CANDIDATE_STRIDE_FRAMES",
    "PREDICTION_FRAME_COUNT",
    "PREFIX_FRAME_COUNT",
    "STAGING_FRAME_COUNT",
    "closure_confidence",
    "controller_centres",
    "select_action_only_window",
]
