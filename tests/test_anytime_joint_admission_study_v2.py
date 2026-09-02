import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.anytime_joint_admission_v2 import (
    AdmissionContractV2,
    JointAdmissionConfigV2,
    JointAnytimeAdmissionControllerV2,
)
from bayesian_phystwin_experiments.anytime_joint_admission_v2 import (
    run_joint_admission_study,
    simulate_discrete_admission_scenario,
    wilson_interval,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols" / "anytime_joint_admission_v2.json"


def _contract() -> AdmissionContractV2:
    return AdmissionContractV2(
        candidate_id="candidate",
        fallback_id="fallback",
        score_id="score",
        harm_definition_id="harm",
        information_set_id="information",
        reveal_policy_id="reveal",
    )


def test_wilson_interval_contains_observed_fraction() -> None:
    lower, upper = wilson_interval(15, 100)

    assert lower < 0.15 < upper
    with pytest.raises(ValueError, match="invalid binomial"):
        wilson_interval(2, 1)


def test_vectorized_deterministic_crossing_matches_controller() -> None:
    simulated = simulate_discrete_admission_scenario(
        probabilities=np.asarray([1.0]),
        gain_scores=np.asarray([1.0]),
        harmful=np.asarray([False]),
        replication_count=1,
        horizon=120,
        minimum_resolved_trials=1,
        shared_epoch_alpha=0.025,
        gain_bet_fractions=np.asarray([0.05, 0.10, 0.20, 0.40, 0.60, 0.80]),
        maximum_harm_rate=0.10,
        harm_alternative_fractions=np.asarray([0.10, 0.25, 0.50, 0.75, 0.90]),
        seed=1,
    )
    expected = int(simulated["shared_alpha_iut"]["median_first_crossing"])

    controller = JointAnytimeAdmissionControllerV2(
        JointAdmissionConfigV2(
            loss_cap=1.0,
            maximum_harm_rate=0.10,
            total_alpha=0.05,
            epoch_alpha_continuation=0.5,
            minimum_resolved_trials=1,
        ),
        _contract(),
    )
    observed = None
    for index in range(120):
        trial_id = f"trial-{index}"
        controller.issue_trial(
            trial_id=trial_id,
            issued_step=2 * index,
            maturity_step=2 * index + 1,
        )
        controller.resolve_trial(
            trial_id=trial_id,
            resolved_step=2 * index + 1,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        if controller.authorized:
            observed = index + 1
            break

    assert observed == expected


def test_frozen_protocol_has_two_distinct_union_nulls() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    scenarios = protocol["scenarios"]

    assert protocol["status"] == "frozen-before-controlled-execution"
    assert scenarios["gain_boundary_low_harm"]["null_component"] == "mean-gain"
    assert scenarios["harm_boundary_positive_gain"]["null_component"] == "harm-rate"
    assert protocol["design"]["total_alpha"] == 0.05
    assert protocol["design"]["epoch_alpha_continuation"] == 0.5
    assert protocol["information_boundary"]["real_outcomes_used"] is False


def test_small_study_is_deterministic_and_records_threshold_advantage() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["design"]["replication_count"] = {
        "null": 100,
        "alternative": 100,
    }
    protocol["design"]["horizon"] = 80
    protocol["mechanism_gate"] = {
        "maximum_null_wilson_upper": 1.0,
        "minimum_moderate_power_gain": -1.0,
        "maximum_moderate_median_crossing_ratio": 100.0,
        "minimum_strong_power": 0.0,
        "decision": "test-only",
    }

    first = run_joint_admission_study(protocol)
    second = run_joint_admission_study(protocol)

    assert first == second
    assert first["design"]["shared_component_threshold"] == pytest.approx(40.0)
    assert first["design"]["split_component_threshold"] == pytest.approx(80.0)
    assert first["mechanism_gate"]["passed"] is True


def test_scenario_validation_rejects_malformed_atoms() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        simulate_discrete_admission_scenario(
            probabilities=np.asarray([0.4, 0.4]),
            gain_scores=np.asarray([0.1, -0.1]),
            harmful=np.asarray([False, True]),
            replication_count=10,
            horizon=10,
            minimum_resolved_trials=1,
            shared_epoch_alpha=0.025,
            gain_bet_fractions=np.asarray([0.2]),
            maximum_harm_rate=0.1,
            harm_alternative_fractions=np.asarray([0.5]),
            seed=1,
        )
