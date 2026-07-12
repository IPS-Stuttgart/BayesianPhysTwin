from __future__ import annotations

import numpy as np

from causal4d.source_panel_signatures import (
    continuous_nonclosure_signature,
    estimate_repeatability_floor,
    heldout_mechanism_eligibility,
    hold_relaxation_signature,
    reversal_sign_flip_signature,
    speed_signature,
)


def test_source_panel_signature_gates_accept_strong_synthetic_effects() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(scale=0.001, size=(3, 12, 4, 3))
    floor = estimate_repeatability_floor({"a": base, "b": base + 0.002})
    sigma = floor["sigma_repeat_m"]

    lift = base.copy()
    lift[..., 2] += 0.004
    lower = base.copy()
    lower[..., 2] -= 0.004
    reversal = reversal_sign_flip_signature(
        lift,
        lower,
        common_action_axis=[0.0, 0.0, 1.0],
        sigma_repeat_m=sigma,
    )
    assert reversal["eligible"] is True

    nonclosure_residual = base.copy()
    nonclosure_residual[:, -3:] += 0.004
    nonclosure = continuous_nonclosure_signature(
        nonclosure_residual,
        pre_action_frames=[0, 1, 2],
        post_return_frames=[9, 10, 11],
        sigma_repeat_m=sigma,
    )
    assert nonclosure["eligible"] is True

    slow = base.copy()
    fast = base + 0.004
    speed = speed_signature(
        fast,
        slow,
        measured_fast_peak_speed_mps=[0.10, 0.10, 0.10],
        measured_slow_peak_speed_mps=[0.05, 0.05, 0.05],
        sigma_repeat_m=sigma,
    )
    assert speed["eligible"] is True


def test_hold_relaxation_and_heldout_shrinkage_gates() -> None:
    frame_dt_s = 1.0 / 30.0
    times = frame_dt_s * np.arange(31)
    decay = 0.006 * np.exp(-times / 0.20)
    hold = np.zeros((3, 31, 2, 3), dtype=float)
    hold[..., 0] = decay[None, :, None]
    hold_result = hold_relaxation_signature(
        hold,
        frame_dt_s=frame_dt_s,
        sigma_repeat_m=0.0005,
    )
    assert hold_result["eligible"] is True

    eligibility = heldout_mechanism_eligibility(
        np.ones(12),
        np.full(12, 0.75),
        track_gain_m=np.full(12, 0.002),
        late_track_gain_m=np.full(12, 0.002),
        cd_degradation_m=np.zeros(12),
        track_repeatability_sd_m=0.001,
        late_track_repeatability_sd_m=0.001,
        cd_repeatability_sd_m=0.001,
    )
    assert eligibility["eligible_for_confirmatory_evaluation"] is True
