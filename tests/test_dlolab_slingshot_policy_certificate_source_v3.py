"""Pure contracts for the fresh independent-action Slingshot study."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.policy_gain_certificate import (
    calibrate_policy_gain_lower_bound,
    fit_local_policy_gain_predictor,
)
from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
from bayesian_phystwin_experiments.dlolab_slingshot_belief import BASELINE
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v2 import (
    score as score_v2,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v3 import (
    ACTION_COUNT,
    BOOTSTRAP_REPLICATES,
    COUNTS,
    FUTURE_PROCESS_COUNT,
    candidate_predictions,
    continuous_worlds,
    future_action_task,
    guarded_decisions,
    pre_future_checks,
    prefix_batch_count,
    prefix_task,
    protocol,
    score,
    sensor_errors,
    validate_rosters,
)


def _predictor():
    rng = np.random.default_rng(262075)
    return fit_local_policy_gain_predictor(
        reference_ids=tuple(f"ref-{index:02d}" for index in range(10)),
        reference_features=rng.normal(size=(10, 161)),
        reference_action_gains=rng.normal(0.01, 0.01, size=(10, 7)),
        neighbor_count=7,
    )


def test_v3_rosters_are_fresh_disjoint_and_process_complete() -> None:
    validate_rosters()
    calibration = continuous_worlds("calibration")
    evaluation = continuous_worlds("evaluation")
    assert len(calibration) == COUNTS["calibration"] == 128
    assert len(evaluation) == COUNTS["evaluation"] == 288
    assert prefix_batch_count("calibration") == 16
    assert prefix_batch_count("evaluation") == 36
    assert prefix_task("evaluation", 35)["world_indices"] == list(range(280, 288))
    assert future_action_task("evaluation", 287, 7)["action_index"] == 7
    assert FUTURE_PROCESS_COUNT == ACTION_COUNT * (128 + 288) == 3328
    with pytest.raises(ValueError, match="independent future"):
        future_action_task("evaluation", 288, 0)


def test_protocol_changes_only_rosters_and_native_execution_surface() -> None:
    frozen = protocol()
    assert frozen["qualification"]["passed"] is True
    assert frozen["reference"]["world_count"] == 147
    assert frozen["reference"]["neighbor_count"] == 7
    assert frozen["calibration"]["rank"] == 117
    assert frozen["execution"]["future_process_count"] == 3328
    assert frozen["execution"]["world_qa_before_calibration_or_scoring"] is True
    assert frozen["evaluation_future_before_decision_barrier"] is False
    assert frozen["retry_authorized"] is False
    assert frozen["replacement_authorized"] is False
    assert frozen["v2_world_retry_authorized"] is False
    assert frozen["new_recordings"] is False
    assert frozen["bootstrap_replicates"] == BOOTSTRAP_REPLICATES


def test_v3_sensor_draws_are_deterministic_and_new() -> None:
    calibration = sensor_errors("calibration")
    assert calibration.tobytes() == sensor_errors("calibration").tobytes()
    assert calibration.shape == (128, 3, 4, 3)
    assert sensor_errors("evaluation").shape == (288, 3, 4, 3)
    assert calibration[:10].tobytes() != sensor_errors("evaluation")[:10].tobytes()


def test_v3_candidate_and_guard_preserve_the_frozen_policy_contract() -> None:
    rng = np.random.default_rng(262076)
    candidate = candidate_predictions(
        "calibration",
        rng.normal(size=(128, 3, 4, 3)),
        rng.normal(size=(27, 3, 4, 3)),
        rng.normal(size=(27, 7)),
        _predictor(),
    )
    assert candidate["candidate_actions"].shape == (128,)
    assert candidate["features"].shape == (128, 161)
    assert candidate["neighbor_indices"].shape == (128, 7)

    evaluation = {
        "candidate_actions": np.full(288, 4, dtype=np.int64),
        "predicted_gain": np.r_[np.full(30, 0.2), np.zeros(258)],
        "expected_losses": np.tile(np.arange(7, dtype=np.float64), (288, 1)),
        "mean_raw_upper": np.zeros((288, 7), dtype=np.float64),
    }
    calibration = calibrate_policy_gain_lower_bound(
        predicted_gain=np.linspace(0.0, 0.127, 128),
        realized_gain=np.zeros(128),
        miscoverage=0.10,
    )
    simultaneous = RegretCalibration(coverage=0.9, count=128, rank=117, offset=1.0)
    guarded = guarded_decisions(evaluation, calibration, simultaneous)
    accepted = guarded["accepted_mask"]
    assert np.all(guarded["decisions"][~accepted, 3] == BASELINE)
    assert pre_future_checks(guarded, all_prefix_qa=True)["pre_future_gate_passed"]


def test_v3_score_is_the_frozen_scorer_with_a_new_schema() -> None:
    rng = np.random.default_rng(262077)
    rewards = rng.normal(7.0, 0.02, size=(288, 7))
    candidate = {
        "candidate_actions": np.full(288, 4, dtype=np.int64),
        "mean_raw_upper": np.zeros((288, 7), dtype=np.float64),
    }
    guarded = {
        "decisions": np.column_stack(
            (
                np.full(288, BASELINE),
                np.full(288, 4),
                np.full(288, BASELINE),
                np.full(288, BASELINE),
            )
        ),
        "lower_gain_bound": np.full(288, -1.0),
    }
    calibration = calibrate_policy_gain_lower_bound(
        predicted_gain=np.linspace(0.0, 0.127, 128),
        realized_gain=np.zeros(128),
        miscoverage=0.10,
    )
    result = score(
        candidate,
        guarded,
        rewards,
        calibration,
        RegretCalibration(coverage=0.9, count=128, rank=117, offset=1.0),
        all_native_qa=True,
        pre_future_gate_passed=False,
    )
    comparator = score_v2(
        candidate,
        guarded,
        rewards,
        calibration,
        RegretCalibration(coverage=0.9, count=128, rank=117, offset=1.0),
        all_native_qa=True,
        pre_future_gate_passed=False,
    )
    assert {key: value for key, value in result.items() if key != "schema"} == {
        key: value for key, value in comparator.items() if key != "schema"
    }
    assert result["schema"] == "dlolab-slingshot-policy-certificate-result-v3"
    assert result["arms"]["policy_gain_guard"]["mean_gain_over_incumbent"] == 0.0
    assert result["source_gate_passed"] is False
