from bayesian_phystwin.structural_benchmark import (
    run_structural_recovery_benchmark,
)


def test_structural_recovery_benchmark_passes_all_gates():
    result = run_structural_recovery_benchmark()
    assert result["passed"] is True
    assert all(result["acceptance_gates"].values())
    selected = {
        value["family"]: (value["selected_variant"], value["selected_rank"])
        for value in result["family_results"]
    }
    assert selected == {
        "frame": ("frame_only", 4),
        "gravity": ("hierarchical", 4),
        "rest_geometry": ("rest_geometry_only", 4),
        "initial_state": ("initial_state_only", 4),
        "combined": ("hierarchical", 4),
        "omitted_physics": ("baseline", 4),
    }
    assert result["withheld_future_mutation"]["future_arrays_admitted_to_fit_api"] is False
    assert result["zero_correction_parity"]["passed"] is True
    assert result["contact_inference"]["preserved"] is True
