from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_selective_virtual_sensing_staging import (
    closure_confidence,
    controller_centres,
    dynamic_window_source_case,
    end_effector_origins,
    select_action_only_window,
    select_translation_contact_window,
)


PROTOCOL = str(
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "deform360_selective_virtual_sensing_v1.json"
)


def _source_selection_seal() -> dict[str, object]:
    cases = []
    for index in range(24):
        cases.append(
            {
                "case": f"case-{index:02d}",
                "translation_contact_v2": {
                    "selected_raw_frame_range_half_open": [8, 89],
                    "object_geometry_read": False,
                    "object_tracks_read": False,
                    "target_metric_read": False,
                    "future_tactile_exposed_to_predictor": False,
                },
            }
        )
    payload: dict[str, object] = {
        "artifact_kind": "Deform360DynamicWindowSourceSelectionSeal",
        "protocol_id": "deform360-dynamic-window-source-v1",
        "cases": cases,
        "information_boundary": {
            "fresh_objects_or_reserved_targets_read": False,
            "object_geometry_used_for_window_selection": False,
            "object_tracks_used_for_window_selection": False,
            "target_metric_used_for_window_selection": False,
            "future_tactile_exposed_to_prediction_method": False,
            "selection_sealed_before_open_outcomes_are_attached_for_diagnosis": True,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_controller_centres_preserve_grippers_before_norms() -> None:
    actions = np.zeros((100, 2, 5, 3), dtype=float)
    actions[:, 0, :, 0] = 1.0
    actions[:, 1, :, 0] = -1.0

    centres = controller_centres(actions)

    assert centres.shape == (100, 2, 3)
    np.testing.assert_array_equal(centres[:, 0, 0], np.ones(100))
    np.testing.assert_array_equal(centres[:, 1, 0], -np.ones(100))


def test_static_aperture_falls_back_to_unit_confidence() -> None:
    confidence = closure_confidence(np.full((100, 2), 0.03))

    np.testing.assert_array_equal(confidence, np.ones((100, 2)))


def test_window_selection_uses_action_only_and_earliest_tie() -> None:
    actions = np.zeros((220, 1, 5, 3), dtype=float)
    actions[80:161, :, :, 0] = np.linspace(0.0, 1.0, 81)[:, None, None]
    actions[161:, :, :, 0] = 1.0
    openings = np.zeros((220, 1), dtype=float)

    result = select_action_only_window(actions, openings, protocol_path=PROTOCOL)

    assert result["selected_raw_frame_range_half_open"] == [80, 161]
    assert result["object_geometry_read"] is False
    assert result["tactile_read"] is False
    assert result["candidate_stride_frames"] == 6


def test_end_effector_origins_do_not_mix_pose_rows() -> None:
    actions = np.zeros((100, 2, 5, 3), dtype=float)
    actions[:, 0, 0, 0] = np.arange(100)
    actions[:, 1, 0, 1] = np.arange(100)
    actions[:, :, 1:4, :] = 1000.0
    actions[:, :, 4, :] = -1000.0

    origins = end_effector_origins(actions)

    assert origins.shape == (100, 2, 3)
    np.testing.assert_array_equal(origins[:, 0, 0], np.arange(100))
    np.testing.assert_array_equal(origins[:, 1, 1], np.arange(100))
    np.testing.assert_array_equal(origins[:, :, 2], 0.0)


def test_translation_contact_window_ignores_rotation_and_aperture_motion() -> None:
    actions = np.zeros((220, 5, 3), dtype=float)
    actions[80:161, 1:4, 0] = np.linspace(0.0, 100.0, 81)[:, None]
    actions[80:161, 4, 0] = np.linspace(0.0, 50.0, 81)
    openings = np.zeros(220, dtype=float)
    contact = np.ones(220, dtype=bool)

    result = select_translation_contact_window(actions, openings, contact)

    assert result["selected_raw_frame_range_half_open"] == [8, 89]
    assert result["contact_supported_future_translation_path_m"] == 0.0
    assert result["has_contact_supported_future_motion"] is False


def test_translation_contact_window_selects_supported_future_motion() -> None:
    actions = np.zeros((220, 5, 3), dtype=float)
    # Large unsupported motion should lose to smaller motion during contact.
    actions[20:77, 0, 0] = np.linspace(0.0, 2.0, 57)
    actions[77:, 0, 0] = 2.0
    actions[104:161, 0, 1] = np.linspace(0.0, 1.0, 57)
    actions[161:, 0, 1] = 1.0
    openings = np.zeros(220, dtype=float)
    contact = np.zeros(220, dtype=bool)
    contact[104:161] = True

    result = select_translation_contact_window(actions, openings, contact)

    assert result["selected_raw_frame_range_half_open"] == [86, 167]
    assert result["contact_supported_future_translation_path_m"] > 0.9
    assert result["has_contact_supported_future_motion"] is True
    assert result["future_tactile_exposed_to_predictor"] is False


def test_dynamic_window_source_case_validates_target_free_seal() -> None:
    seal = _source_selection_seal()

    row = dynamic_window_source_case(seal, "case-13")

    assert row["translation_contact_v2"]["selected_raw_frame_range_half_open"] == [
        8,
        89,
    ]


def test_dynamic_window_source_case_rejects_tampering() -> None:
    seal = _source_selection_seal()
    seal["cases"][13]["translation_contact_v2"]["target_metric_read"] = True

    with np.testing.assert_raises_regex(ValueError, "checksum changed"):
        dynamic_window_source_case(seal, "case-13")
