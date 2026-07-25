"""Action-only temporal staging for the prospective Deform360 panel."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .deform360_selective_virtual_sensing_protocol import (
    load_selective_virtual_sensing_protocol,
)


DYNAMIC_WINDOW_SOURCE_PROTOCOL_ID = "deform360-dynamic-window-source-v1"
DYNAMIC_WINDOW_SOURCE_SELECTION_ARTIFACT_KIND = (
    "Deform360DynamicWindowSourceSelectionSeal"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_selection_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dynamic_window_source_case(
    selection_seal: Mapping[str, Any], case_name: str
) -> dict[str, Any]:
    """Validate a target-free source-window seal and return one case row."""

    _require(
        selection_seal.get("artifact_kind")
        == DYNAMIC_WINDOW_SOURCE_SELECTION_ARTIFACT_KIND,
        "wrong dynamic-window selection artifact kind",
    )
    _require(
        selection_seal.get("protocol_id") == DYNAMIC_WINDOW_SOURCE_PROTOCOL_ID,
        "dynamic-window source protocol changed",
    )
    _require(
        selection_seal.get("result_sha256")
        == _canonical_selection_sha256(selection_seal),
        "dynamic-window selection checksum changed",
    )
    boundary = selection_seal.get("information_boundary", {})
    _require(
        boundary.get("fresh_objects_or_reserved_targets_read") is False
        and boundary.get("object_geometry_used_for_window_selection") is False
        and boundary.get("object_tracks_used_for_window_selection") is False
        and boundary.get("target_metric_used_for_window_selection") is False
        and boundary.get("future_tactile_exposed_to_prediction_method") is False
        and boundary.get(
            "selection_sealed_before_open_outcomes_are_attached_for_diagnosis"
        )
        is True,
        "dynamic-window selection crossed its target-free boundary",
    )
    rows = selection_seal.get("cases")
    _require(isinstance(rows, list) and len(rows) == 24, "source case panel changed")
    _require(
        len({str(row.get("case")) for row in rows}) == len(rows),
        "source case repeated",
    )
    matches = [row for row in rows if row.get("case") == case_name]
    _require(len(matches) == 1, "case is absent from the source-window seal")
    row = dict(matches[0])
    selection = row.get("translation_contact_v2", {})
    _require(
        selection.get("object_geometry_read") is False
        and selection.get("object_tracks_read") is False
        and selection.get("target_metric_read") is False
        and selection.get("future_tactile_exposed_to_predictor") is False,
        "source case selection crossed its target-free boundary",
    )
    frame_range = selection.get("selected_raw_frame_range_half_open")
    _require(
        isinstance(frame_range, list)
        and len(frame_range) == 2
        and int(frame_range[1]) - int(frame_range[0]) == 81,
        "source case window changed",
    )
    return row


def controller_centres(actions: np.ndarray) -> np.ndarray:
    """Return the frozen-v1 mean over the five released action rows.

    Deform360 encodes an end-effector pose as translation, three rotation rows,
    and aperture metadata.  Treating those rows as points was a v1 staging
    error.  This function is retained unchanged solely to reproduce the frozen
    negative experiment; new protocols must use :func:`end_effector_origins`.
    """

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


def end_effector_origins(actions: np.ndarray) -> np.ndarray:
    """Return the actual end-effector translation for each frame and gripper."""

    values = np.asarray(actions, dtype=np.float64)
    _require(np.all(np.isfinite(values)), "robot actions are non-finite")
    if values.ndim == 3:
        _require(
            values.shape[1:] == (5, 3),
            "monomanual Deform360 actions must have shape (T,5,3)",
        )
        values = values[:, None, :, :]
    _require(
        values.ndim == 4 and values.shape[2:] == (5, 3),
        "Deform360 actions must have shape (T,5,3) or (T,G,5,3)",
    )
    _require(len(values) >= 2 and values.shape[1] >= 1, "robot actions are empty")
    return np.array(values[:, :, 0, :], copy=True)


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
        "selected_metric_value": selected["mean_closed_weighted_path_length_m"],
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


def select_translation_contact_window(
    actions: np.ndarray,
    openings: np.ndarray,
    contact_active: np.ndarray,
    *,
    staging_frame_count: int = 81,
    prediction_frame_count: int = 76,
    first_update_frame: int = 19,
    candidate_first_frame: int = 8,
    candidate_stride_frames: int = 6,
) -> dict[str, Any]:
    """Select a dynamic benchmark window from translation and tactile support.

    The known future action is a conditioning input.  Tactile is used only to
    choose a benchmark interval and is not exposed to the prediction method.
    Motion is scored after the first update so a selected future cannot be
    dominated by approach motion that has already ended at forecast time.
    """

    origins = end_effector_origins(actions)
    closed = closure_confidence(openings)
    contact = np.asarray(contact_active)
    _require(contact.dtype == np.dtype(np.bool_), "contact_active must be boolean")
    _require(contact.ndim == 1, "contact_active must have shape (T,)")
    _require(
        closed.shape == origins.shape[:2],
        "gripper openings do not match action groups",
    )
    _require(
        len(contact) == len(origins),
        "tactile contact does not match the robot timeline",
    )
    _require(staging_frame_count >= prediction_frame_count >= 2, "invalid frame counts")
    _require(
        0 <= first_update_frame < prediction_frame_count - 1,
        "first update must leave at least one scored action step",
    )
    _require(candidate_stride_frames > 0, "candidate stride must be positive")

    candidates = np.arange(
        candidate_first_frame,
        len(origins) - staging_frame_count + 1,
        candidate_stride_frames,
        dtype=np.int64,
    )
    _require(len(candidates) > 0, "episode has no complete translation window")
    rows: list[dict[str, Any]] = []
    for start_value in candidates:
        start = int(start_value)
        stop = start + staging_frame_count
        selected_origins = origins[start:stop]
        selected_closed = closed[start:stop]
        selected_contact = contact[start:stop]
        step = np.linalg.norm(np.diff(selected_origins, axis=0), axis=-1)
        closure_weight = np.minimum(selected_closed[:-1], selected_closed[1:])
        contact_weight = np.minimum(selected_contact[:-1], selected_contact[1:]).astype(
            np.float64
        )[:, None]
        weighted = step * closure_weight
        future_slice = slice(first_update_frame, prediction_frame_count - 1)
        future_path = float(np.mean(np.sum(weighted[future_slice], axis=0)))
        supported_future_path = float(
            np.mean(
                np.sum(weighted[future_slice] * contact_weight[future_slice], axis=0)
            )
        )
        rows.append(
            {
                "start": start,
                "future_translation_path_m": future_path,
                "contact_supported_future_translation_path_m": supported_future_path,
                "window_contact_fraction": float(np.mean(selected_contact)),
                "future_contact_fraction": float(
                    np.mean(selected_contact[first_update_frame:prediction_frame_count])
                ),
            }
        )

    best_score = max(
        float(row["contact_supported_future_translation_path_m"]) for row in rows
    )
    selected = next(
        row
        for row in rows
        if float(row["contact_supported_future_translation_path_m"]) == best_score
    )
    start = int(selected["start"])
    return {
        "selection_rule": (
            "maximum-mean-closure-and-tactile-supported-end-effector-translation-"
            "after-first-update"
        ),
        "selected_raw_frame_range_half_open": [start, start + staging_frame_count],
        "prediction_raw_frame_range_half_open": [start, start + prediction_frame_count],
        "candidate_first_frame": int(candidate_first_frame),
        "candidate_stride_frames": int(candidate_stride_frames),
        "candidate_count": len(rows),
        "tie_break": "earliest start",
        "first_update_frame": int(first_update_frame),
        "future_translation_path_m": selected["future_translation_path_m"],
        "contact_supported_future_translation_path_m": best_score,
        "window_contact_fraction": selected["window_contact_fraction"],
        "future_contact_fraction": selected["future_contact_fraction"],
        "has_contact_supported_future_motion": bool(best_score > 0.0),
        "input_fields": [
            "robot.actions[...,0,:]",
            "robot.openings",
            "aligned_tactile_contact_boolean",
        ],
        "known_future_action_is_conditioning_input": True,
        "future_tactile_used_for_dataset_window_selection": True,
        "future_tactile_exposed_to_predictor": False,
        "object_geometry_read": False,
        "object_tracks_read": False,
        "target_metric_read": False,
    }


__all__ = [
    "DYNAMIC_WINDOW_SOURCE_PROTOCOL_ID",
    "DYNAMIC_WINDOW_SOURCE_SELECTION_ARTIFACT_KIND",
    "closure_confidence",
    "controller_centres",
    "dynamic_window_source_case",
    "end_effector_origins",
    "select_action_only_window",
    "select_translation_contact_window",
]
