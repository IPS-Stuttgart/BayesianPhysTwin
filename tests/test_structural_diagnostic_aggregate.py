from bayesian_phystwin.structural_diagnostic_aggregate import (
    aggregate_structural_diagnostics,
)


def _summary(case):
    methods = {}
    for method in (
        "released_phystwin",
        "graph_persistence_readout",
        "baseline",
        "frame_only",
        "initial_state_only",
        "rest_geometry_only",
        "rest_state",
        "hierarchical",
    ):
        ratio = 0.8 if method == "graph_persistence_readout" else 1.0
        metrics = {
            metric: {
                "baseline_mean_m": 0.02,
                "candidate_mean_m": 0.02 * ratio,
                "percent_change": 100.0 * (ratio - 1.0),
            }
            for metric in ("chamfer_distance_m", "track_error_m")
        }
        methods[method] = {
            "future": metrics,
            "horizon": {"late": metrics},
            "far_graph": {"future_observation_error_mean_m": 0.02 * ratio},
        }
    return {
        "schema_version": 1,
        "experiment": "hierarchical_graph_structural_calibration",
        "case": case,
        "selected_physical_variant": "baseline",
        "selected_physical_rank": 4,
        "methods": methods,
    }


def test_structural_aggregate_compares_selected_physics_to_graph_persistence():
    result = aggregate_structural_diagnostics((_summary("a"), _summary("b")))
    assert result["case_count"] == 2
    assert result["selected_physical_variant_counts"] == {"baseline": 2}
    gate = result["acceptance_gates"][
        "cross_action_track_vs_graph_persistence"
    ]
    assert gate["observed_mean_error_ratio"] == 1.25
    assert gate["passed"] is False
    assert result["structural_candidate_accepted"] is False
