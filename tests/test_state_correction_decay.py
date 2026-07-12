from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.state_correction_decay import (
    analyze_state_correction_decay,
    analyze_state_correction_modes,
)


def test_state_correction_decay_recovers_tail_and_time_constant() -> None:
    frame_dt_s = 0.05
    time_constant_s = 0.25
    frames = np.arange(100)
    magnitude = 0.002 + 0.010 * np.exp(-(frames * frame_dt_s) / time_constant_s)
    baseline = np.zeros((100, 2, 3), dtype=float)
    corrected = baseline.copy()
    corrected[:, :, 0] = magnitude[:, None]

    result = analyze_state_correction_decay(
        baseline,
        corrected,
        start_frame=0,
        stop_frame=100,
        frame_dt_s=frame_dt_s,
    )

    assert result["decay_fit"]["adequate_single_decay"] is True
    assert result["decay_fit"]["tail_floor_m"] == pytest.approx(0.002, abs=1e-8)
    assert result["decay_fit"]["time_constant_s"] == pytest.approx(
        time_constant_s, rel=1e-3
    )
    assert result["summary"]["final_aligned_retention"] < 0.17
    assert result["summary"]["final_orthogonal_rms_m"] == pytest.approx(0.0)


def test_state_correction_decay_tracks_direction_change() -> None:
    baseline = np.zeros((4, 1, 3), dtype=float)
    corrected = baseline.copy()
    corrected[0, 0, 0] = 1.0
    corrected[1, 0, 1] = 1.0
    corrected[2, 0, 1] = 0.5
    corrected[3, 0, 1] = 0.25

    result = analyze_state_correction_decay(
        baseline,
        corrected,
        start_frame=0,
        stop_frame=4,
        frame_dt_s=1.0,
    )

    assert result["per_frame"]["aligned_retention"][1:] == [0.0, 0.0, 0.0]
    assert result["per_frame"]["orthogonal_rms_m"][1] == pytest.approx(1.0)


def test_state_correction_decay_rejects_zero_injection() -> None:
    state = np.zeros((4, 1, 3), dtype=float)
    with pytest.raises(ValueError, match="zero position magnitude"):
        analyze_state_correction_decay(
            state,
            state,
            start_frame=0,
            stop_frame=4,
            frame_dt_s=1.0,
        )


def test_state_correction_modes_resolve_modes_and_constraint_distance() -> None:
    node_count = 9
    constant = np.ones(node_count) / np.sqrt(node_count)
    alternating = np.array([1.0, -1.0] * 4 + [0.0])
    alternating /= np.linalg.norm(alternating)
    basis = np.column_stack([constant, alternating])
    baseline = np.zeros((3, node_count, 3), dtype=float)
    corrected = baseline.copy()
    corrected[0, :, 0] = 2.0 * constant + alternating
    corrected[1, :, 0] = 1.5 * constant + 0.5 * alternating
    corrected[2, :, 0] = constant

    result = analyze_state_correction_modes(
        baseline,
        corrected,
        basis,
        np.array([0.0, 0.1]),
        np.arange(node_count, dtype=float),
        np.array([0]),
        start_frame=0,
        stop_frame=3,
        frame_dt_s=0.1,
    )

    modes = result["modal_retention"]["modes"]
    assert modes[0]["final_directional_retention"] == pytest.approx(0.5)
    assert modes[1]["final_directional_retention"] == pytest.approx(0.0, abs=1e-12)
    assert result["modal_retention"]["dominant_retained_direction_mode"] == 0
    assert result["modal_retention"]["final_in_basis_energy_fraction"] == pytest.approx(
        1.0
    )
    assert result["distance_resolved_retention"]["near"]["node_count"] > 0
    assert result["distance_resolved_retention"]["far"]["node_count"] > 0
