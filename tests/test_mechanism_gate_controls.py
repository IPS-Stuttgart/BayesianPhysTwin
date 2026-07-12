from __future__ import annotations

from causal4d.mechanism_gate_controls import (
    MechanismGateControlConfig,
    run_mechanism_gate_controls,
)


def test_gate_controls_are_deterministic_and_use_all_crossfit_sessions() -> None:
    config = MechanismGateControlConfig(simulation_count=3, random_seed=17)
    first = run_mechanism_gate_controls(config)
    second = run_mechanism_gate_controls(config)

    assert first == second
    assert first["artifact_kind"] == "MechanismGateControlEvidence"
    assert len(first["result_sha256"]) == 64
    assert set(first["arms"]) == {
        "placebo_null",
        "positive_control",
        "placebo_on_positive",
    }
    assert all(arm["simulation_count"] == 3 for arm in first["arms"].values())
    assert len(first["threshold_sensitivity"]) == 9


def test_positive_control_recovers_the_injected_grid_value_without_noise() -> None:
    config = MechanismGateControlConfig(
        simulation_count=2,
        random_seed=4,
        persistent_offset_std_m=1.0e-9,
        temporal_noise_std_m=1.0e-9,
    )
    result = run_mechanism_gate_controls(config)
    quantiles = result["arms"]["positive_control"]["fitted_parameter_quantiles"]

    assert quantiles["0.05"] == config.positive_actuation_gain
    assert quantiles["0.5"] == config.positive_actuation_gain
    assert quantiles["0.95"] == config.positive_actuation_gain
