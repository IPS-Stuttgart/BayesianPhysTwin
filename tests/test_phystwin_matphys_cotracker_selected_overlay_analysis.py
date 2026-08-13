import copy

import pytest

from bayesian_phystwin.phystwin_matphys_cotracker_selected_overlay_analysis import (
    FIXED_COMPARATOR,
    PRIMARY_CANDIDATE,
    analyze_matphys_cotracker_selected_overlay_report,
)


def _report(*, primary_scale: float = 0.8) -> dict:
    cases = {}
    for index in range(22):
        cases[f"case-{index:02d}"] = {
            "baseline": {
                "chamfer_distance_m": 0.009,
                "track_error_m": 0.016,
            },
            "candidates": {
                PRIMARY_CANDIDATE: {
                    "chamfer_distance_m": 0.009 * primary_scale,
                    "track_error_m": 0.016 * primary_scale,
                },
                FIXED_COMPARATOR: {
                    "chamfer_distance_m": 0.0085,
                    "track_error_m": 0.0155,
                },
            },
        }
    return {
        "config": {
            "baseline_kind": "selected_overlay_sequential",
            "observation_source": "cotracker3_source_depth",
            "manual_prefix_override": False,
            "manual_observed_track_count": None,
            "cotracker_minimum_quality": 0.5,
            "cotracker_maximum_cycle_error_px": 5.0,
            "dense_correction_scales": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            "maximum_residuals_m": [0.04, 0.06, 0.1, 0.15, 0.2, 0.4],
            "prior_strengths": [0.0001],
            "relative_cap_multipliers": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            "temporal_gamma_candidates": [0.0, 0.25, 0.5, 1.0],
        },
        "information_boundary": {
            "future_inputs_used_for_prediction": False,
            "manual_prefix_role": "disabled; manual tracks are evaluation-only",
        },
        "case_results": cases,
    }


def test_locked_primary_arm_passes_all_advancement_gates() -> None:
    result = analyze_matphys_cotracker_selected_overlay_report(_report())

    assert result["gate_passed"] is True
    assert result["decision"] == "advance-to-fresh-public-object-source-panel"
    assert result["arms"]["primary_causal_temporal"]["joint_case_wins"] == 22
    assert result["arms"]["primary_causal_temporal"][
        "below_published_8mm_15mm_operating_point"
    ]


def test_manual_prefix_input_is_rejected() -> None:
    report = _report()
    report["config"]["manual_prefix_override"] = True

    with pytest.raises(ValueError, match="manual prefix is enabled"):
        analyze_matphys_cotracker_selected_overlay_report(report)


def test_future_input_is_rejected() -> None:
    report = _report()
    report["information_boundary"]["future_inputs_used_for_prediction"] = True

    with pytest.raises(ValueError, match="future observations entered prediction"):
        analyze_matphys_cotracker_selected_overlay_report(report)


def test_nonregistered_candidate_cannot_change_the_decision() -> None:
    report = _report()
    altered = copy.deepcopy(report)
    for case in altered["case_results"].values():
        case["candidates"]["future_oracle"] = {
            "chamfer_distance_m": 0.0,
            "track_error_m": 0.0,
        }

    first = analyze_matphys_cotracker_selected_overlay_report(report)
    second = analyze_matphys_cotracker_selected_overlay_report(altered)

    assert first == second


def test_worst_case_regression_fails_the_gate() -> None:
    report = _report()
    report["case_results"]["case-00"]["candidates"][PRIMARY_CANDIDATE][
        "track_error_m"
    ] = 0.020

    result = analyze_matphys_cotracker_selected_overlay_report(report)

    assert result["gate_passed"] is False
    assert result["gates"]["maximum_case_metric_regression_within_limit"] is False
