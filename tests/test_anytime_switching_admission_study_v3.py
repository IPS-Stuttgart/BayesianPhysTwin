import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.anytime_switching_admission_v3 import (
    run_switching_admission_study,
    simulate_switching_admission_scenario,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols" / "anytime_switching_admission_v3.json"


def _shared_arguments() -> dict[str, object]:
    return {
        "minimum_resolved_trials": 1,
        "shared_epoch_alpha": 0.025,
        "gain_bet_fractions": np.asarray([0.1, 0.4, 0.8]),
        "maximum_harm_rate": 0.10,
        "harm_alternative_fractions": np.asarray([0.1, 0.5, 0.9]),
        "robust_bet_fractions": np.asarray([0.1, 0.4, 0.8]),
    }


def test_deterministic_safe_stream_crosses_both_certificates() -> None:
    result = simulate_switching_admission_scenario(
        phases=[
            {
                "name": "safe",
                "duration": 200,
                "active_null_component": None,
                "probabilities": [1.0],
                "gain_scores": [0.2],
                "harmful": [False],
            }
        ],
        replication_count=1,
        seed=1,
        **_shared_arguments(),
    )

    assert result["latched_shared_alpha_iut"]["crossing_count"] == 1
    assert result["switching_union_min_score"]["crossing_count"] == 1
    phase = result["phase_expectations"][0]
    assert phase["expected_gain_score"] == pytest.approx(0.2)
    assert phase["expected_harm_rate"] == pytest.approx(0.0)
    assert phase["expected_robust_score"] > 0.0


def test_deterministic_switching_invalidity_never_crosses_robust_process() -> None:
    result = simulate_switching_admission_scenario(
        phases=[
            {
                "name": "harm-invalid",
                "duration": 20,
                "active_null_component": "harm-rate",
                "probabilities": [1.0],
                "gain_scores": [0.5],
                "harmful": [True],
            },
            {
                "name": "gain-invalid",
                "duration": 200,
                "active_null_component": "mean-gain",
                "probabilities": [1.0],
                "gain_scores": [-0.1],
                "harmful": [False],
            },
        ],
        replication_count=1,
        seed=1,
        **_shared_arguments(),
    )

    assert result["switching_union_min_score"]["crossing_count"] == 0
    assert all(
        phase["expected_robust_score"] <= 0.0
        for phase in result["phase_expectations"]
    )


def test_frozen_protocol_encodes_switching_union_counterexample() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    phases = protocol["scenarios"]["switching_invalidity"]["phases"]

    assert protocol["status"] == "frozen-before-controlled-execution"
    assert [phase["active_null_component"] for phase in phases] == [
        "harm-rate",
        "mean-gain",
    ]
    assert sum(phase["duration"] for phase in phases) == 500
    assert protocol["information_boundary"]["real_outcomes_used"] is False


def test_small_study_is_deterministic() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["design"]["replication_count"] = {
        "null": 300,
        "alternative": 300,
    }
    protocol["mechanism_gate"] = {
        "maximum_robust_null_wilson_upper": 1.0,
        "minimum_switching_null_iut_crossing": 0.0,
        "maximum_switching_null_robust_crossing": 1.0,
        "minimum_moderate_robust_power": 0.0,
        "minimum_strong_robust_power": 0.0,
        "decision": "test-only",
    }

    first = run_switching_admission_study(protocol)
    second = run_switching_admission_study(protocol)

    assert first == second
    assert first["mechanism_gate"]["passed"] is True
    assert first["design"]["e_value_threshold"] == pytest.approx(40.0)
    switching = first["scenarios"]["switching_invalidity"]
    assert (
        switching["latched_shared_alpha_iut"]["crossing_probability"]
        > switching["switching_union_min_score"]["crossing_probability"]
    )


def test_malformed_phase_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        simulate_switching_admission_scenario(
            phases=[
                {
                    "duration": 10,
                    "probabilities": [0.4, 0.4],
                    "gain_scores": [0.1, -0.1],
                    "harmful": [False, True],
                }
            ],
            replication_count=10,
            seed=1,
            **_shared_arguments(),
        )
