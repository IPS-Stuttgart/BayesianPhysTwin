"""Pure contracts for the reward-aligned stochastic-execution study."""

from __future__ import annotations

from typing import Any

import numpy as np

from bayesian_phystwin.policy_gain_certificate import (
    calibrate_policy_gain_lower_bound,
)
from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
from bayesian_phystwin_experiments.dlolab_slingshot_belief import BASELINE
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v3 import (
    COUNTS,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v3 import (
    score as score_v3,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v4 import (
    continuous_worlds,
    protocol,
    reward_aligned_world_qa,
    reward_aligned_world_rewards,
    score,
    validate_rosters,
)


def test_v4_rosters_are_fresh_and_disjoint() -> None:
    validate_rosters()
    assert len(continuous_worlds("calibration")) == COUNTS["calibration"]
    assert len(continuous_worlds("evaluation")) == COUNTS["evaluation"]


def test_protocol_freezes_reward_aligned_execution_without_weakening_v3() -> None:
    frozen = protocol()
    assert frozen["qualification"]["passed"] is True
    assert frozen["parent_v3"]["rescored"] is False
    assert frozen["parent_v3"]["roster_reused"] is False
    assert frozen["execution"]["duplicate_reward_error_at_most"] == 0.001
    assert frozen["execution"]["duplicate_position"] == "reported_not_admission"
    assert frozen["execution"]["incumbent_reward_estimator"] == (
        "mean_of_action_slots_5_and_7"
    )
    assert frozen["retry_authorized"] is False
    assert frozen["replacement_authorized"] is False
    assert frozen["protected_data_read"] is False
    assert frozen["new_recordings"] is False


def test_incumbent_reward_averages_independent_slots(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v4.task_metrics",
        lambda row: {"native_reward": float(row["reward"])},
    )
    rows = [{"reward": np.array(float(index))} for index in range(8)]
    rewards = reward_aligned_world_rewards(rows)
    assert rewards.shape == (7,)
    assert rewards[BASELINE] == 6.0
    assert np.array_equal(rewards[:5], np.arange(5, dtype=np.float64))


def test_reward_aligned_qa_reports_position_divergence_but_gates_reward(
    monkeypatch: Any,
) -> None:
    base: dict[str, Any] = {
        "checks": {
            "ordinary": True,
            "duplicate_positions": False,
            "duplicate_rewards": True,
        },
        "qa_passed": False,
    }
    monkeypatch.setattr(
        "bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v4.independent_world_qa",
        lambda *args, **kwargs: base,
    )
    accepted = reward_aligned_world_qa([], [], np.empty(0), {}, world_count=1)
    assert accepted["qa_passed"] is True
    assert accepted["duplicate_position_deterministic"] is False
    assert accepted["checks"]["duplicate_position_reported"] is True

    base["checks"]["duplicate_rewards"] = False
    rejected = reward_aligned_world_qa([], [], np.empty(0), {}, world_count=1)
    assert rejected["qa_passed"] is False


def test_v4_score_preserves_v3_statistics_and_adds_estimand() -> None:
    rng = np.random.default_rng(262084)
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
    simultaneous = RegretCalibration(
        coverage=0.9, count=128, rank=117, offset=1.0
    )
    result = score(
        candidate,
        guarded,
        rewards,
        calibration,
        simultaneous,
        all_native_qa=True,
        pre_future_gate_passed=False,
    )
    comparator = score_v3(
        candidate,
        guarded,
        rewards,
        calibration,
        simultaneous,
        all_native_qa=True,
        pre_future_gate_passed=False,
    )
    extras = {"execution_estimand", "incumbent_reward_estimator"}
    assert {
        key: value
        for key, value in result.items()
        if key not in extras | {"schema"}
    } == {key: value for key, value in comparator.items() if key != "schema"}
    assert result["schema"] == "dlolab-slingshot-policy-certificate-result-v4"
