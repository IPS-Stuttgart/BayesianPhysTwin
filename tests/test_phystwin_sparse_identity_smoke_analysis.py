from bayesian_phystwin.phystwin_sparse_identity_smoke_analysis import (
    PRIMARY_ARM,
    analyze_sparse_identity_smoke,
)


def _case(
    *,
    baseline: tuple[float, float],
    candidate: tuple[float, float],
) -> dict:
    return {
        "baseline": {
            "chamfer_distance_m": baseline[0],
            "track_error_m": baseline[1],
        },
        "baseline_trajectory": {"sha256": "physical"},
        "cotracker_depth_lift": {
            "identity_count": 8,
            "valid_fraction": 0.25,
            "two_view_fraction_of_valid": 0.1,
            "mean_prior_reliability": 0.6,
            "median_observation_std_m": 0.01,
            "reliability_uses_phystwin_innovation": False,
            "innovation_likelihood_count": 1,
        },
        "causal_selection": {
            "selectors": {
                PRIMARY_ARM: {
                    "accepted": True,
                    "base_relative_candidate": {"relative_cap_multiplier": 1.0},
                    "selected_candidate": {"gamma": 0.25},
                    "future_metrics": {
                        "chamfer_distance_m": candidate[0],
                        "track_error_m": candidate[1],
                    },
                }
            }
        },
        "candidates": {
            "forbidden_metric_oracle": {
                "chamfer_distance_m": 1e-9,
                "track_error_m": 1e-9,
            }
        },
    }


def _protocol() -> dict:
    return {
        "protocol_id": "unit",
        "cases": ["case"],
        "method": {
            "runner_config": {
                "observation_source": (
                    "final_data_plus_cotracker3_sparse_identity"
                ),
                "manual_prefix_override": False,
            }
        },
        "inputs": {"physical_baseline": {"sha256": "physical"}},
        "future_read": {"primary_arm": PRIMARY_ARM},
        "smoke_gate": {
            "minimum_identity_count": 1,
            "minimum_valid_fraction": 0.01,
            "minimum_track_improvement_fraction": 0.05,
            "maximum_cd_regression_fraction": 0.01,
            "pass_action": "continue",
            "fail_action": "stop",
        },
    }


def test_locked_sparse_identity_analyzer_ignores_side_arms() -> None:
    candidate = {
        "config": {
            "observation_source": (
                "final_data_plus_cotracker3_sparse_identity"
            ),
            "manual_prefix_override": False,
        },
        "case_results": {
            "case": _case(
                baseline=(0.020, 0.060),
                candidate=(0.011, 0.047),
            )
        },
    }
    comparator = {
        "config": {
            "observation_source": "final_data",
            "manual_prefix_override": False,
        },
        "case_results": {
            "case": {
                "baseline": {
                    "chamfer_distance_m": 0.020,
                    "track_error_m": 0.060,
                },
                "causal_selection": {
                    "selectors": {
                        PRIMARY_ARM: {
                            "future_metrics": {
                                "chamfer_distance_m": 0.011,
                                "track_error_m": 0.050,
                            }
                        }
                    }
                },
            }
        },
    }

    result = analyze_sparse_identity_smoke(
        candidate,
        comparator,
        _protocol(),
    )

    assert result["smoke_gate_passed"]
    assert result["recommendation"] == "continue"
    assert "forbidden_metric_oracle" not in str(result)


def test_sparse_identity_smoke_fails_without_support() -> None:
    candidate_case = _case(
        baseline=(0.020, 0.060),
        candidate=(0.011, 0.047),
    )
    candidate_case["cotracker_depth_lift"]["identity_count"] = 0
    candidate_case["cotracker_depth_lift"]["valid_fraction"] = 0.0
    candidate = {
        "config": {
            "observation_source": (
                "final_data_plus_cotracker3_sparse_identity"
            ),
            "manual_prefix_override": False,
        },
        "case_results": {"case": candidate_case},
    }
    comparator = {
        "config": {
            "observation_source": "final_data",
            "manual_prefix_override": False,
        },
        "case_results": {
            "case": {
                "baseline": {
                    "chamfer_distance_m": 0.020,
                    "track_error_m": 0.060,
                },
                "causal_selection": {
                    "selectors": {
                        PRIMARY_ARM: {
                            "future_metrics": {
                                "chamfer_distance_m": 0.011,
                                "track_error_m": 0.050,
                            }
                        }
                    }
                },
            }
        },
    }

    result = analyze_sparse_identity_smoke(
        candidate,
        comparator,
        _protocol(),
    )

    assert not result["smoke_gate_passed"]
    assert not result["gates"]["automatic_identity_support"]
