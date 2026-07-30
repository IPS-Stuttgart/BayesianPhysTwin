from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from bayesian_phystwin.cloth_sim2real_belief import (
    ClothReadoutBeliefConfig,
    GuardedReadoutCorrection,
)
from bayesian_phystwin.rgbench_dynamic_belief import (
    RGBenchDynamicSlope,
    build_rgbbench_dynamic_candidates,
    fit_rgbbench_dynamic_slope,
    select_leave_one_garment_out_shrinkages,
)


def _belief(*, accepted: bool) -> GuardedReadoutCorrection:
    correction = (
        np.full((3, 3), 0.01, dtype=np.float64)
        if accepted
        else np.zeros((3, 3), dtype=np.float64)
    )
    return GuardedReadoutCorrection(
        accepted=accepted,
        reason="accepted" if accepted else "rejected",
        selected_name="global" if accepted else "baseline",
        correction_m=correction,
        variance_m2=np.full((3, 3), 1e-4, dtype=np.float64),
        scores=(),
        diagnostics={},
    )


def test_rejected_dynamic_bank_is_exact_physical_fallback() -> None:
    physical = np.arange(45, dtype=np.float64).reshape(5, 3, 3) / 100.0
    candidates = build_rgbbench_dynamic_candidates(
        physical,
        np.arange(5, dtype=np.float64),
        2,
        _belief(accepted=False),
        None,
    )
    assert tuple(candidates) == (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    for candidate in candidates.values():
        assert candidate.exact_physical_fallback
        assert np.array_equal(candidate.trajectory_m, physical)


def test_dynamic_bank_leaves_prefix_untouched_and_propagates_variance() -> None:
    physical = np.zeros((5, 3, 3), dtype=np.float64)
    slope = RGBenchDynamicSlope(
        slope_m_per_s=np.full((3, 3), 0.02, dtype=np.float64),
        variance_m2_per_s2=np.full((3, 3), 4e-4, dtype=np.float64),
        spatial_model="global",
        diagnostics={},
    )
    candidates = build_rgbbench_dynamic_candidates(
        physical,
        np.arange(5, dtype=np.float64),
        2,
        _belief(accepted=True),
        slope,
        shrinkages=(0.0, 0.5),
    )
    static = candidates[0.0]
    dynamic = candidates[0.5]
    assert np.array_equal(static.trajectory_m[:3], physical[:3])
    assert np.array_equal(dynamic.trajectory_m[:3], physical[:3])
    assert np.allclose(static.trajectory_m[3:], 0.01)
    assert np.allclose(dynamic.trajectory_m[3], 0.02)
    assert np.allclose(dynamic.trajectory_m[4], 0.03)
    assert np.all(dynamic.variance_m2[4] > static.variance_m2[4])


def test_dynamic_bank_caps_total_correction() -> None:
    physical = np.zeros((4, 3, 3), dtype=np.float64)
    slope = RGBenchDynamicSlope(
        slope_m_per_s=np.full((3, 3), 10.0, dtype=np.float64),
        variance_m2_per_s2=np.ones((3, 3), dtype=np.float64),
        spatial_model="global",
        diagnostics={},
    )
    candidate = build_rgbbench_dynamic_candidates(
        physical,
        np.arange(4, dtype=np.float64),
        1,
        replace(_belief(accepted=True), correction_m=np.zeros((3, 3))),
        slope,
        shrinkages=(1.0,),
        maximum_correction_m=0.1,
    )[1.0]
    norms = np.linalg.norm(candidate.trajectory_m[2:], axis=2)
    assert np.allclose(norms, 0.1)


def test_dynamic_slope_recovers_linear_prefix_without_residual_reliability() -> None:
    base = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    slope = np.array([0.02, -0.01, 0.005], dtype=np.float64)
    physical = np.repeat(base[None], len(times), axis=0)
    clouds = [base + time * slope for time in times]
    estimate = fit_rgbbench_dynamic_slope(
        physical,
        clouds,
        times,
        np.array([[0, 1, 2]], dtype=np.int64),
        _belief(accepted=True),
        config=ClothReadoutBeliefConfig(
            candidate_count=1,
            covariance_probes=0,
        ),
    )
    assert np.allclose(estimate.slope_m_per_s, slope[None], atol=1e-8)
    assert estimate.diagnostics["prior_reliability_uses_state_innovation"] is False
    assert estimate.diagnostics["innovation_processed_once"] is True


def test_leave_one_garment_out_selection_never_uses_held_garment() -> None:
    garments = ("a", "b", "c")
    actions = ("fold",)
    shrinkages = (0.0, 0.5, 1.0)
    rows: list[dict[str, object]] = []
    preferred = {"a": 0.0, "b": 0.5, "c": 1.0}
    for garment in garments:
        for sample in range(3):
            for shrinkage in shrinkages:
                rows.append(
                    {
                        "garment": garment,
                        "action": "fold",
                        "sample": sample,
                        "shrinkage": shrinkage,
                        "candidate_score_m": abs(shrinkage - preferred[garment]),
                    }
                )
    selection = select_leave_one_garment_out_shrinkages(
        rows,
        garments=garments,
        actions=actions,
        shrinkages=shrinkages,
    )
    assert selection[("a", "fold")] == 0.5
    assert selection[("b", "fold")] == 0.0
    assert selection[("c", "fold")] == 0.0

    mutated = [
        {
            **row,
            "candidate_score_m": (
                1000.0
                if row["garment"] == "a"
                else row["candidate_score_m"]
            ),
        }
        for row in rows
    ]
    changed = select_leave_one_garment_out_shrinkages(
        mutated,
        garments=garments,
        actions=actions,
        shrinkages=shrinkages,
    )
    assert changed[("a", "fold")] == selection[("a", "fold")]


def test_dynamic_protocol_locks_the_implemented_bank() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root / "configs/sota/rgbbench_isotropic_dynamic_v2.json"
        ).read_text(encoding="utf-8")
    )
    bank = payload["development"]["temporal_bank"]
    assert tuple(bank["slope_shrinkages"]) == (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    assert bank["maximum_correction_m"] == 0.1
    assert bank["exact_physical_fallback"] is True
    assert "other two garments" in payload["development"]["selection"]
