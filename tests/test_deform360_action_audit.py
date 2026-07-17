from __future__ import annotations

import numpy as np
import pytest

from causal4d_public.deform360_action_audit import (
    closure_confidence,
    controller_centres,
    summarize_robot_action,
)


def test_controller_centres_supports_mono_and_bimanual_actions() -> None:
    mono = np.zeros((6, 5, 3), dtype=np.float64)
    mono[:, :, 0] = np.arange(6)[:, None]
    bi = np.stack((mono, mono + np.array([0.0, 2.0, 0.0])), axis=1)

    mono_centres = controller_centres(mono)
    bi_centres = controller_centres(bi)

    assert mono_centres.shape == (6, 1, 3)
    assert bi_centres.shape == (6, 2, 3)
    np.testing.assert_allclose(bi_centres[:, 0], mono_centres[:, 0])
    np.testing.assert_allclose(bi_centres[:, 1, 1], 2.0)


def test_action_summary_finds_more_dynamic_equal_length_window() -> None:
    actions = np.zeros((10, 5, 3), dtype=np.float64)
    actions[5:, :, 0] = np.arange(5, dtype=np.float64)[:, None]
    summary = summarize_robot_action(
        actions,
        np.zeros(10, dtype=np.float64),
        locked_start=0,
        locked_stop=4,
    )

    assert summary["locked_window"]["mean_displacement_from_window_start_m"] == 0.0
    assert (
        summary["best_equal_length_displacement_window"][
            "mean_displacement_from_window_start_m"
        ]
        > 0.0
    )
    assert summary["locked_to_best_displacement_ratio"] == 0.0


def test_action_summary_rejects_mismatched_opening_groups() -> None:
    actions = np.zeros((5, 2, 5, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="openings"):
        summarize_robot_action(
            actions,
            np.zeros(5, dtype=np.float64),
            locked_start=0,
            locked_stop=3,
        )


def test_action_summary_respects_candidate_start_grid() -> None:
    actions = np.zeros((14, 5, 3), dtype=np.float64)
    actions[3:8, :, 0] = np.arange(5, dtype=np.float64)[:, None]
    summary = summarize_robot_action(
        actions,
        np.zeros(14, dtype=np.float64),
        locked_start=0,
        locked_stop=4,
        candidate_start_frame=2,
        candidate_stride_frames=4,
    )

    start = summary["best_equal_length_displacement_window"]["frame_range_half_open"][0]
    assert start in (2, 6, 10)
    assert summary["candidate_count"] == 3


def test_closure_confidence_is_large_for_small_aperture() -> None:
    openings = np.linspace(0.1, 0.02, 20, dtype=np.float64)[:, None]

    confidence = closure_confidence(openings)

    assert confidence[0, 0] == 0.0
    assert confidence[-1, 0] == 1.0


def test_contact_conditioned_window_rejects_open_approach_motion() -> None:
    actions = np.zeros((24, 5, 3), dtype=np.float64)
    actions[1:9, :, 0] = np.arange(1, 9, dtype=np.float64)[:, None]
    actions[9:16, :, 0] = 8.0
    actions[16:, :, 0] = 8.0 + np.arange(8, dtype=np.float64)[:, None]
    openings = np.full(24, 0.1, dtype=np.float64)
    openings[12:] = 0.02

    summary = summarize_robot_action(
        actions,
        openings,
        locked_start=0,
        locked_stop=8,
    )

    displacement_start = summary["best_equal_length_displacement_window"][
        "frame_range_half_open"
    ][0]
    contact_start = summary["best_contact_conditioned_path_window"][
        "frame_range_half_open"
    ][0]
    assert displacement_start < 8
    assert contact_start >= 9
