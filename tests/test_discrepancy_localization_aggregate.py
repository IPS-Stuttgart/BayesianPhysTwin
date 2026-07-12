import json

from bayesian_phystwin.discrepancy_localization_aggregate import (
    aggregate_discrepancy_localization,
)
from bayesian_phystwin.phystwin_discrepancy_localization import (
    BASELINE,
    GENERALIZED_FORCE,
    LOCALIZATION_METHODS,
    PREFIX_STATE,
    READOUT,
    STRUCTURAL_CONTROL,
)


def _summary(case, ratios, *, cross_view=False):
    methods = {}
    for method in LOCALIZATION_METHODS:
        ratio = ratios[method]
        methods[method] = {
            "future": {
                "chamfer_distance_m": {"candidate_mean_m": 0.02 * ratio},
                "track_error_m": {"candidate_mean_m": 0.03 * ratio},
            },
            "horizon": {
                "late": {
                    "chamfer_distance_m": {"candidate_mean_m": 0.025 * ratio},
                    "track_error_m": {"candidate_mean_m": 0.035 * ratio},
                }
            },
            "far_graph": {"future_observation_error_mean_m": 0.03 * ratio},
            "coverage": {"coordinate_coverage_90": 0.7},
        }
    return {
        "experiment": "phystwin_discrepancy_localization_v1",
        "case": case,
        "method_order": list(LOCALIZATION_METHODS),
        "methods": methods,
        "comparison_contract": {
            "graph_rank": 4,
            "common_physical_particle_count": 4,
            "official_nonlinear_warp_rerun": True,
        },
        "zero_force_parity": {"bitwise_identical": True},
        "fit_diagnostics": {"force_limit": {"limit_applied": False}},
        "observation_model_audit": {
            "cross_view": (
                {
                    "status": "available",
                    "mean_cross_view_error_ratio": 0.7,
                }
                if cross_view
                else {"available": False}
            )
        },
    }


def test_aggregate_supports_force_only_when_it_beats_readout_everywhere(tmp_path):
    ratios = {
        BASELINE: 1.0,
        READOUT: 0.8,
        PREFIX_STATE: 0.9,
        GENERALIZED_FORCE: 0.7,
        STRUCTURAL_CONTROL: 1.1,
    }
    paths = []
    for index in range(3):
        path = tmp_path / f"case_{index}.json"
        path.write_text(json.dumps(_summary(f"case_{index}", ratios)))
        paths.append(path)

    result = aggregate_discrepancy_localization(paths, tmp_path / "aggregate.json")

    assert result["acceptance_gates"]["constant_force_supported"] is True
    assert (
        result["localization_conclusion"]
        == "generalized_force_location_supported_diagnostically"
    )
    assert result["claim_boundary"]["may_select_confirmatory_physical_mechanism"] is False


def test_aggregate_keeps_readout_only_result_unresolved_without_cross_view(tmp_path):
    ratios = {
        BASELINE: 1.0,
        READOUT: 0.7,
        PREFIX_STATE: 0.9,
        GENERALIZED_FORCE: 0.95,
        STRUCTURAL_CONTROL: 1.05,
    }
    paths = []
    for index in range(2):
        path = tmp_path / f"case_{index}.json"
        path.write_text(json.dumps(_summary(f"case_{index}", ratios)))
        paths.append(path)

    result = aggregate_discrepancy_localization(paths, tmp_path / "aggregate.json")

    assert result["best_equal_case_track_method"] == READOUT
    assert (
        result["localization_conclusion"]
        == "readout_is_best_but_physical_vs_observation_location_unresolved"
    )
    assert result["acceptance_gates"]["cross_view_available_in_every_case"] is False
