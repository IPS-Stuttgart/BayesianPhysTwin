import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.anytime_factor_envelope_v4 import (
    run_factor_envelope_study,
    simulate_factor_envelope_scenario,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols" / "anytime_factor_envelope_v4.json"


def test_protocol_discloses_development_and_freezes_confirmation() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    disclosure = protocol["development_disclosure"]

    assert protocol["status"] == "frozen-before-v4-confirmation-execution"
    assert disclosure["version_3_results_observed_before_design"] is True
    assert disclosure["pilot_seed_base"] != disclosure["confirmation_seed_base"]
    assert disclosure["confirmation_seed_roster_opened_before_protocol_commit"] is False
    assert protocol["information_boundary"]["real_outcomes_used"] is False


def test_deterministic_safe_phase_crosses_at_least_as_early() -> None:
    result = simulate_factor_envelope_scenario(
        phases=[
            {
                "name": "deterministic-safe",
                "duration": 200,
                "probabilities": [1.0],
                "gain_scores": [0.2],
                "harmful": [False],
            }
        ],
        replication_count=10,
        minimum_resolved_trials=1,
        shared_epoch_alpha=0.025,
        gain_bet_fractions=np.asarray([0.05, 0.1, 0.2, 0.4, 0.6, 0.8]),
        maximum_harm_rate=0.1,
        harm_alternative_fractions=np.asarray([0.1, 0.25, 0.5, 0.75, 0.9]),
        robust_bet_fractions=np.asarray([0.05, 0.1, 0.2, 0.4, 0.6, 0.8]),
        seed=1,
    )

    scalar = result["switching_union_min_score_v3"]
    envelope = result["switching_union_factor_envelope_v4"]
    assert envelope["component_count"] == 30
    assert envelope["median_first_crossing"] <= scalar["median_first_crossing"]


def test_null_phase_diagnostics_verify_the_lower_envelope_inequality() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = simulate_factor_envelope_scenario(
        phases=protocol["scenarios"]["switching_invalidity"]["phases"],
        replication_count=50,
        minimum_resolved_trials=20,
        shared_epoch_alpha=0.025,
        gain_bet_fractions=np.asarray(
            protocol["design"]["gain_bet_fractions"],
            dtype=np.float64,
        ),
        maximum_harm_rate=protocol["design"]["maximum_harm_rate"],
        harm_alternative_fractions=np.asarray(
            protocol["design"]["harm_alternative_fractions"],
            dtype=np.float64,
        ),
        robust_bet_fractions=np.asarray(
            protocol["design"]["robust_bet_fractions"],
            dtype=np.float64,
        ),
        seed=2,
    )

    for phase in result["phase_expectations"]:
        assert phase["maximum_expected_envelope_factor"] <= 1.0 + 1e-12


def test_small_study_is_deterministic_and_records_power_tradeoff() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(protocol)
    protocol["design"]["replication_count"] = {
        "null": 200,
        "alternative": 200,
    }
    protocol["mechanism_gate"] = {
        "maximum_envelope_null_wilson_upper": 1.0,
        "maximum_switching_null_envelope_crossing": 1.0,
        "minimum_moderate_envelope_power": 0.0,
        "minimum_moderate_power_gain": -1.0,
        "maximum_moderate_median_crossing_ratio": 10.0,
        "minimum_strong_envelope_power": 0.0,
        "decision": "test-only",
    }

    first = run_factor_envelope_study(protocol)
    second = run_factor_envelope_study(protocol)

    assert first == second
    assert first["design"]["factor_envelope_component_count"] == 30
    assert first["mechanism_gate"]["passed"] is True
    assert (
        first["derived_comparison"]["moderate_envelope_power"]
        >= first["derived_comparison"]["moderate_min_score_power"]
    )


def test_simulator_rejects_malformed_phase_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        simulate_factor_envelope_scenario(
            phases=[
                {
                    "duration": 10,
                    "probabilities": [0.4, 0.4],
                    "gain_scores": [0.1, -0.1],
                    "harmful": [False, True],
                }
            ],
            replication_count=10,
            minimum_resolved_trials=1,
            shared_epoch_alpha=0.025,
            gain_bet_fractions=np.asarray([0.2]),
            maximum_harm_rate=0.1,
            harm_alternative_fractions=np.asarray([0.5]),
            robust_bet_fractions=np.asarray([0.2]),
            seed=1,
        )
