from bayesian_phystwin.phystwin_bias_aware_ray_analysis import (
    PRIMARY_ARM,
    analyze_bias_aware_ray_smoke,
)


def _case(
    baseline: tuple[float, float],
    candidate: tuple[float, float],
    *,
    accepted: bool,
) -> dict:
    return {
        "baseline": {
            "chamfer_distance_m": baseline[0],
            "track_error_m": baseline[1],
        },
        "causal_selection": {
            "selectors": {
                PRIMARY_ARM: {
                    "accepted": accepted,
                    "fallback_applied": not accepted,
                    "fallback_is_exact": True if not accepted else None,
                    "admission": {"accepted": accepted},
                    "future_metrics": {
                        "chamfer_distance_m": candidate[0],
                        "track_error_m": candidate[1],
                    },
                }
            }
        },
        "candidates": {
            "forbidden_side_arm": {
                "chamfer_distance_m": 1e-9,
                "track_error_m": 1e-9,
            }
        },
    }


def _comparator_case(metrics: tuple[float, float]) -> dict:
    return {
        "causal_selection": {
            "selectors": {
                "causal_selected_dense_relative_cap": {
                    "future_metrics": {
                        "chamfer_distance_m": metrics[0],
                        "track_error_m": metrics[1],
                    }
                }
            }
        }
    }


def test_locked_analyzer_reads_only_primary_arm() -> None:
    cases = ["a", "b", "c"]
    candidate = {
        "config": {"observation_source": "alltracker_multiview_ray_bias_aware"},
        "case_results": {
            "a": _case((0.010, 0.020), (0.009, 0.018), accepted=True),
            "b": _case((0.012, 0.021), (0.012, 0.021), accepted=False),
            "c": _case((0.011, 0.019), (0.010, 0.018), accepted=True),
        },
    }
    comparator = {
        "config": {"observation_source": "cotracker3_source_depth"},
        "case_results": {
            case: _comparator_case((0.012, 0.022))
            for case in [*cases, "unused_full_cohort_case"]
        },
    }
    protocol = {
        "protocol_id": "unit",
        "cases": cases,
        "future_read": {"primary_arm": PRIMARY_ARM},
        "smoke_gate": {
            "both_metric_win_or_tie_count_at_least": 2,
            "maximum_allowed_case_metric_regression_fraction": 0.05,
            "pass_action": "continue",
            "fail_action": "stop",
        },
    }

    result = analyze_bias_aware_ray_smoke(
        candidate,
        comparator,
        protocol,
    )

    assert result["smoke_gate_passed"]
    assert result["recommendation"] == "continue"
    assert "forbidden_side_arm" not in str(result)
    assert result["per_case"][1]["fallback_applied"]
