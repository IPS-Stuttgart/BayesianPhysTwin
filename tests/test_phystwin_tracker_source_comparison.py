from copy import deepcopy

import pytest

from bayesian_phystwin.phystwin_tracker_source_comparison import (
    PRIMARY_ARM,
    analyze_tracker_source_comparison,
)


CASES = (
    "single_lift_cloth",
    "single_lift_cloth_3",
    "single_lift_cloth_4",
)


def _config(observation_source: str) -> dict:
    return {
        "observation_source": observation_source,
        "baseline_kind": "raw_matphys_replay",
        "manual_prefix_override": False,
        "cotracker_minimum_quality": 0.5,
        "cotracker_maximum_cycle_error_px": 5.0,
        "prior_strengths": [0.0001],
        "maximum_residuals_m": [0.04],
        "dense_correction_scales": [1.0],
        "nearest_cloud_windows": [],
        "relative_cap_quantile": 0.95,
        "relative_cap_multipliers": [1.0],
        "temporal_gamma_candidates": [0.0],
        "rbf_center_counts": [16],
        "rbf_minimum_availability_fraction": 0.5,
        "planar_degrees": [0, 1, 2],
        "planar_ridge_strength": 0.001,
        "process_std_m": 0.005,
        "observation_std_m": 0.001,
        "initial_std_m": 0.01,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }


def _result(observation_source: str, values: tuple[tuple[float, float], ...]) -> dict:
    return {
        "config": _config(observation_source),
        "case_results": {
            case: {
                "candidates": {
                    PRIMARY_ARM: {
                        "chamfer_distance_m": value[0],
                        "track_error_m": value[1],
                    }
                }
            }
            for case, value in zip(CASES, values, strict=True)
        },
    }


def _protocol() -> dict:
    return {
        "protocol_id": "phystwin-alltracker-source-depth-smoke-v1",
        "cases": list(CASES),
        "method": {
            "baseline_kind": "raw_matphys_replay",
            "manual_prefix_override": False,
            "minimum_quality": 0.5,
            "maximum_cycle_error_px": 5.0,
            "prior_strengths": [0.0001],
            "maximum_residuals_m": [0.04],
            "dense_correction_scales": [1.0],
            "nearest_cloud_windows": [],
            "relative_cap_quantile": 0.95,
            "relative_cap_multipliers": [1.0],
            "temporal_gamma_candidates": [0.0],
            "rbf_center_counts": [16],
            "rbf_minimum_availability_fraction": 0.5,
            "planar_degrees": [0, 1, 2],
            "planar_ridge_strength": 0.001,
            "endpoint_filter": {
                "process_std_m": 0.005,
                "observation_std_m": 0.001,
                "initial_std_m": 0.01,
                "inlier_prior": 0.95,
                "outlier_variance_multiplier": 100.0,
            },
        },
        "smoke_gate": {
            "both_metric_win_count_at_least": 2,
            "maximum_allowed_case_regression_fraction": 0.1,
            "pass_action": "continue",
            "fail_action": "stop",
        },
    }


def test_tracker_source_comparison_passes_consistent_gain() -> None:
    comparator = _result(
        "cotracker3_source_depth",
        ((0.010, 0.020), (0.012, 0.022), (0.014, 0.024)),
    )
    candidate = _result(
        "alltracker_source_depth",
        ((0.009, 0.019), (0.011, 0.021), (0.013, 0.023)),
    )

    result = analyze_tracker_source_comparison(
        comparator,
        candidate,
        _protocol(),
    )

    assert result["smoke_gate_passed"]
    assert result["aggregate"]["both_metric_win_or_tie_count"] == 3
    assert result["recommendation"] == "continue"


def test_tracker_source_comparison_rejects_split_metric_result() -> None:
    comparator = _result(
        "cotracker3_source_depth",
        ((0.010, 0.020), (0.010, 0.020), (0.010, 0.020)),
    )
    candidate = _result(
        "alltracker_source_depth",
        ((0.0105, 0.019), (0.0105, 0.019), (0.0105, 0.019)),
    )

    result = analyze_tracker_source_comparison(
        comparator,
        candidate,
        _protocol(),
    )

    assert not result["smoke_gate_passed"]
    assert not result["gates"][
        "both_equal_case_future_means_improve_over_cotracker3_source"
    ]
    assert result["recommendation"] == "stop"


def test_tracker_source_comparison_rejects_changed_setting() -> None:
    comparator = _result(
        "cotracker3_source_depth",
        ((0.010, 0.020),) * 3,
    )
    candidate = _result(
        "alltracker_source_depth",
        ((0.009, 0.019),) * 3,
    )
    candidate = deepcopy(candidate)
    candidate["config"]["cotracker_minimum_quality"] = 0.6

    with pytest.raises(ValueError, match="minimum_quality"):
        analyze_tracker_source_comparison(
            comparator,
            candidate,
            _protocol(),
        )


def test_tracker_source_comparison_requires_exact_candidate_cases() -> None:
    comparator = _result(
        "cotracker3_source_depth",
        ((0.010, 0.020),) * 3,
    )
    candidate = _result(
        "alltracker_source_depth",
        ((0.009, 0.019),) * 3,
    )
    candidate["case_results"].pop("single_lift_cloth_4")

    with pytest.raises(ValueError, match="case order"):
        analyze_tracker_source_comparison(
            comparator,
            candidate,
            _protocol(),
        )
