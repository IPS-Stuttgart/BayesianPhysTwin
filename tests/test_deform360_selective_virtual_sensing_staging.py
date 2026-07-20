from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_selective_virtual_sensing_staging import (
    closure_confidence,
    controller_centres,
    select_action_only_window,
)


PROTOCOL = str(
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "deform360_selective_virtual_sensing_v1.json"
)


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

    result = select_action_only_window(
        actions, openings, protocol_path=PROTOCOL
    )

    assert result["selected_raw_frame_range_half_open"] == [80, 161]
    assert result["object_geometry_read"] is False
    assert result["tactile_read"] is False
    assert result["candidate_stride_frames"] == 6
