from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_fresh_source_window import (
    CANDIDATE_FIRST_FRAME,
    FROZEN_CAMERA_PANEL,
    PROTOCOL_ID,
    RAW_FRAME_COUNT,
    SCORE_STEP_RANGE,
    canonical_sha256,
    fresh_source_case,
    load_fresh_source_window_protocol,
    seal_fresh_source_window_selection,
    select_fresh_source_window,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "sota" / "deform360_fresh_source_window_v1.json"


def _actions(frame_count: int = 180, grippers: int = 1) -> np.ndarray:
    values = np.zeros((frame_count, grippers, 5, 3), dtype=np.float64)
    values[:, :, 1:4, :] = np.eye(3)[None, None]
    if grippers == 1:
        return values[:, 0]
    return values


def test_frozen_protocol_loads_and_is_tamper_evident(tmp_path: Path) -> None:
    protocol = load_fresh_source_window_protocol(PROTOCOL)

    assert protocol["protocol_id"] == PROTOCOL_ID
    assert tuple(protocol["camera_panel"]) == FROZEN_CAMERA_PANEL
    assert protocol["window_selection"]["action_position_field"].endswith("[...,0,:]")

    changed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed["window_selection"]["candidate_stride_frames"] = 7
    changed["config_sha256"] = canonical_sha256(
        changed, digest_key="config_sha256"
    )
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="selection rule changed"):
        load_fresh_source_window_protocol(path)


def test_selector_uses_translation_row_not_pose_row_mean() -> None:
    actions = _actions()
    openings = np.zeros(len(actions), dtype=np.float64)
    expected_start = CANDIDATE_FIRST_FRAME + 5 * 6
    action_start = expected_start + SCORE_STEP_RANGE[0]
    action_stop = expected_start + SCORE_STEP_RANGE[1] + 1
    ramp = np.arange(
        action_stop - action_start, dtype=np.float64
    )
    actions[action_start:action_stop, 0, 0] = ramp
    actions[action_stop:, 0, 0] = ramp[-1]
    actions[:, 1:, :] = 10_000.0

    selected = select_fresh_source_window(actions, openings)

    assert selected["selected_raw_frame_range_half_open"] == [
        expected_start,
        expected_start + RAW_FRAME_COUNT,
    ]
    assert selected["selected_score_m"] > 0.0
    assert selected["input_fields"] == [
        "robot.actions[...,0,:]",
        "robot.openings",
    ]


def test_selector_scores_only_steps_after_first_update() -> None:
    actions = _actions()
    openings = np.zeros(len(actions), dtype=np.float64)
    earliest = CANDIDATE_FIRST_FRAME
    early_stop = earliest + SCORE_STEP_RANGE[0]
    early_ramp = np.arange(SCORE_STEP_RANGE[0], dtype=np.float64)
    actions[earliest:early_stop, 0, 0] = early_ramp
    actions[early_stop:, 0, 0] = early_ramp[-1]
    later = earliest + 6
    start = earliest + SCORE_STEP_RANGE[1] + 1
    stop = later + SCORE_STEP_RANGE[1] + 1
    late_ramp = np.arange(stop - start, dtype=np.float64)
    actions[start:stop, 0, 1] = late_ramp
    actions[stop:, 0, 1] = late_ramp[-1]

    selected = select_fresh_source_window(actions, openings)

    assert selected["selected_raw_frame_range_half_open"][0] == later


def test_selector_tie_breaks_to_earliest_candidate() -> None:
    actions = _actions()
    openings = np.zeros(len(actions), dtype=np.float64)

    selected = select_fresh_source_window(actions, openings)

    assert selected["selected_raw_frame_range_half_open"][0] == CANDIDATE_FIRST_FRAME
    assert selected["selected_score_m"] == 0.0


def test_selector_handles_two_grippers_without_favoring_count() -> None:
    actions = _actions(grippers=2)
    openings = np.zeros((len(actions), 2), dtype=np.float64)
    selected_start = CANDIDATE_FIRST_FRAME + 12
    start = selected_start + SCORE_STEP_RANGE[0]
    stop = selected_start + SCORE_STEP_RANGE[1] + 1
    displacement = np.arange(stop - start, dtype=np.float64)
    actions[start:stop, 0, 0, 0] = displacement
    actions[start:stop, 1, 0, 0] = displacement
    actions[stop:, 0, 0, 0] = displacement[-1]
    actions[stop:, 1, 0, 0] = displacement[-1]

    selected = select_fresh_source_window(actions, openings)

    assert selected["selected_raw_frame_range_half_open"][0] == selected_start


def test_window_seal_binds_case_and_provenance() -> None:
    protocol = load_fresh_source_window_protocol(PROTOCOL)
    queue = {
        "candidates": [
            {
                "queue_rank": 1,
                "object_id": "006-fur",
                "episode_id": 0,
                "category": "filament",
            }
        ]
    }
    case = fresh_source_case(queue, "006-fur", 0)
    selection = select_fresh_source_window(_actions(), np.zeros(180))
    seal = seal_fresh_source_window_selection(
        protocol=protocol,
        case=case,
        selection=selection,
        source_robot_sha256="1" * 64,
        source_preparation_sha256="2" * 64,
        code_revision="3" * 40,
    )

    assert seal["result_sha256"] == canonical_sha256(
        seal, digest_key="result_sha256"
    )
    assert seal["information_boundary"]["object_tracks_read"] is False
    assert seal["information_boundary"]["known_future_action_read"] is True


def test_canonical_hash_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_sha256({"value": np.nan}, digest_key="digest")


def test_fixture_protocol_file_has_stable_sha256() -> None:
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
