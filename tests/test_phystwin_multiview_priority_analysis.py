import copy

import pytest

from bayesian_phystwin.phystwin_multiview_priority_analysis import (
    FIXED_GRAPH_ARM,
    PRIMARY_ARM,
    analyze_multiview_priority_results,
)


def _protocol() -> dict:
    return {
        "cohort": {
            "cases": [
                "single_lift_cloth",
                "single_lift_cloth_1",
                "single_lift_rope",
            ]
        },
        "method": {
            "minimum_identity_prefix_availability_fraction": 0.4,
            "minimum_view_quality": 0.5,
            "maximum_forward_backward_error_px": 5.0,
            "maximum_reprojection_error_px": 3.0,
            "minimum_camera_count": 3,
        },
        "transfer_gate_for_fresh_evaluation": {
            "two_metric_win_or_tie_count_at_least": 2
        },
    }


def _result(observation_source: str, offset: float) -> dict:
    config = {
        "observation_source": observation_source,
        "multiview_priority_minimum_availability_fraction": 0.4,
        "cotracker_minimum_quality": 0.5,
        "cotracker_maximum_cycle_error_px": 5.0,
        "cotracker_maximum_reprojection_error_px": 3.0,
        "cotracker_minimum_camera_count": 3,
        "baseline_kind": "raw_matphys_replay",
        "manual_prefix_override": False,
        "prior_strengths": [0.0001],
        "maximum_residuals_m": [0.04],
        "dense_correction_scales": [1.0],
        "relative_cap_quantile": 0.95,
        "relative_cap_multipliers": [1.0],
        "process_std_m": 0.005,
        "observation_std_m": 0.001,
        "initial_std_m": 0.01,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }
    cases = {}
    for index, case in enumerate(_protocol()["cohort"]["cases"]):
        value = 0.01 + 0.001 * index + offset
        cases[case] = {
            "candidates": {
                PRIMARY_ARM: {
                    "chamfer_distance_m": value,
                    "track_error_m": 2.0 * value,
                },
                FIXED_GRAPH_ARM: {
                    "chamfer_distance_m": 1.1 * value,
                    "track_error_m": 2.2 * value,
                }
            },
            "cotracker_depth_lift": {
                "priority_identity_count": 10 + index,
                "priority_identity_fraction": 0.1 + index * 0.01,
            },
        }
    return {"config": config, "case_results": cases}


def test_analysis_applies_locked_arm_and_gates() -> None:
    source = _result("cotracker3_source_depth", 0.0)
    candidate = _result("cotracker3_multiview_priority", -0.001)

    result = analyze_multiview_priority_results(
        source,
        candidate,
        _protocol(),
        bootstrap_draws=1000,
        bootstrap_seed=4,
    )

    assert result["case_count"] == 3
    assert result["aggregate"]["both_metric_win_or_tie_count"] == 3
    assert result["fresh_evaluation_justified"]
    assert result["aggregate"]["metrics"]["track_error_m"]["difference_m"] < 0.0
    assert (
        result["aggregate"]["fixed_graph_60mm_diagnostic"][
            "both_metric_win_or_tie_count"
        ]
        == 3
    )


def test_analysis_rejects_changed_priority_threshold() -> None:
    source = _result("cotracker3_source_depth", 0.0)
    candidate = _result("cotracker3_multiview_priority", -0.001)
    changed = copy.deepcopy(candidate)
    changed["config"]["multiview_priority_minimum_availability_fraction"] = 0.3

    with pytest.raises(ValueError, match="differs from the locked protocol"):
        analyze_multiview_priority_results(
            source,
            changed,
            _protocol(),
            bootstrap_draws=10,
        )
