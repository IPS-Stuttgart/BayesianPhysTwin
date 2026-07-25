from copy import deepcopy

import pytest

from bayesian_phystwin.phystwin_directional_priority_analysis import (
    FIXED_GRAPH_ARM,
    PRIMARY_ARM,
    analyze_directional_priority_results,
)


def _result(observation_source: str, offset: float) -> dict:
    config = {
        "observation_source": observation_source,
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
        "multiview_priority_minimum_availability_fraction": 0.4,
        "cotracker_minimum_quality": 0.5,
        "cotracker_maximum_cycle_error_px": 5.0,
        "cotracker_maximum_reprojection_error_px": 3.0,
        "cotracker_minimum_camera_count": 3,
        "multiview_tangent_neighbor_count": 16,
    }
    case = {
        "candidates": {
            PRIMARY_ARM: {
                "chamfer_distance_m": 0.010 + offset,
                "track_error_m": 0.020 + offset,
            },
            FIXED_GRAPH_ARM: {
                "chamfer_distance_m": 0.011 + offset,
                "track_error_m": 0.021 + offset,
            },
        },
        "cotracker_depth_lift": {
            "priority_identity_count": 4,
            "priority_identity_fraction": 0.5,
            "source_update_count": 20,
            "multiview_tangent_update_count": 10,
            "multiview_tangent_updates_without_source_count": 3,
        },
    }
    return {
        "config": config,
        "case_results": {
            "single_lift_cloth": deepcopy(case),
            "single_lift_rope": deepcopy(case),
        },
    }


def _protocol() -> dict:
    return {
        "cohort": {"cases": ["single_lift_cloth", "single_lift_rope"]},
        "method": {
            "priority_threshold": 0.4,
            "minimum_view_quality": 0.5,
            "maximum_forward_backward_error_px": 5.0,
            "maximum_reprojection_error_px": 3.0,
            "minimum_camera_count": 3,
        },
        "transfer_gate_for_fresh_evaluation": {
            "two_metric_win_or_tie_count_at_least": 2
        },
    }


def test_directional_analysis_passes_consistent_two_metric_gain() -> None:
    source = _result("cotracker3_source_depth", 0.0)
    hard = _result("cotracker3_multiview_priority", -0.0005)
    candidate = _result(
        "cotracker3_multiview_directional_priority",
        -0.001,
    )

    result = analyze_directional_priority_results(
        source,
        hard,
        candidate,
        _protocol(),
        bootstrap_draws=1000,
        bootstrap_seed=7,
    )

    assert result["fresh_evaluation_justified"]
    assert result["aggregate"]["versus_source"][
        "both_metric_win_or_tie_count"
    ] == 2
    assert result["aggregate"]["versus_hard_priority"]["metrics"][
        "track_error_m"
    ]["difference_m"] < 0.0


def test_directional_analysis_rejects_changed_neighbor_count() -> None:
    source = _result("cotracker3_source_depth", 0.0)
    hard = _result("cotracker3_multiview_priority", -0.0005)
    candidate = _result(
        "cotracker3_multiview_directional_priority",
        -0.001,
    )
    candidate["config"]["multiview_tangent_neighbor_count"] = 15

    with pytest.raises(ValueError, match="neighbor count"):
        analyze_directional_priority_results(
            source,
            hard,
            candidate,
            _protocol(),
            bootstrap_draws=10,
        )
