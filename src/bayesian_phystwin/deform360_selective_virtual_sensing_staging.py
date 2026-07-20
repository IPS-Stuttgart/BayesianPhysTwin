"""Action-only temporal staging for the prospective Deform360 panel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .deform360_selective_virtual_sensing_protocol import (
    load_selective_virtual_sensing_protocol,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def controller_centres(actions: np.ndarray) -> np.ndarray:
    """Return one XYZ centre per frame and gripper from released actions."""

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
    """Map smaller aperture to robust per-episode closure confidence."""

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


def _window_metrics(
    centres: np.ndarray,
    closed: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, Any]:
    selected = centres[start:stop]
    step = np.linalg.norm(np.diff(selected, axis=0), axis=-1)
    selected_closed = closed[start:stop]
    closed_step = step * np.minimum(selected_closed[:-1], selected_closed[1:])
    return {
        "frame_range_half_open": [int(start), int(stop)],
        "frame_count": int(stop - start),
        "mean_closed_weighted_path_length_m": float(
            np.mean(np.sum(closed_step, axis=0))
        ),
    }


def select_action_only_window(
    actions: np.ndarray,
    openings: np.ndarray,
    *,
    protocol_path: str,
) -> dict[str, Any]:
    """Select the locked 81-frame window without object or tactile inputs."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    staging: Mapping[str, Any] = protocol["config"]["temporal_staging"]
    centres = controller_centres(actions)
    closed = closure_confidence(openings)
    _require(
        closed.shape == centres.shape[:2],
        "gripper openings do not match action groups",
    )
    length = int(staging["raw_window_length_frames"])
    candidates = np.arange(
        int(staging["candidate_starts"]["first"]),
        len(centres) - length + 1,
        int(staging["candidate_starts"]["stride"]),
        dtype=np.int64,
    )
    _require(len(candidates) > 0, "episode has no complete action-window candidate")
    rows = [
        _window_metrics(centres, closed, int(start), int(start) + length)
        for start in candidates
    ]
    selected = max(
        rows,
        key=lambda row: float(row["mean_closed_weighted_path_length_m"]),
    )
    return {
        "selected_raw_frame_range_half_open": selected["frame_range_half_open"],
        "selection_metric": "mean_closed_weighted_path_length_m",
        "selected_metric_value": selected[
            "mean_closed_weighted_path_length_m"
        ],
        "candidate_start_frame": int(candidates[0]),
        "candidate_stride_frames": int(staging["candidate_starts"]["stride"]),
        "candidate_count": len(rows),
        "tie_break": "earliest start",
        "input_fields": list(staging["input_fields"]),
        "object_geometry_read": False,
        "object_tracks_read": False,
        "tactile_read": False,
        "known_future_action_is_conditioning_input": True,
    }


__all__ = [
    "closure_confidence",
    "controller_centres",
    "select_action_only_window",
]
