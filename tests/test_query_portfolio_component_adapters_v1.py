from __future__ import annotations

import numpy as np

from bayesian_phystwin.query_portfolio_component_adapters_v1 import (
    slingshot_outcome,
    wrapping_outcome,
)
from bayesian_phystwin.query_portfolio_replication_v1 import WORLD_COUNT


def test_wrapping_adapter_uses_equal_world_sensor_mean_and_exact_fallback() -> None:
    rewards = np.zeros((WORLD_COUNT, 8), dtype=np.float64)
    rewards[:, 1] = 0.01
    decisions = np.zeros((WORLD_COUNT, 4, 2), dtype=np.int64)
    decisions[0, :2, 1] = 1
    outcome = wrapping_outcome(decisions, rewards, primary_arm_index=1)
    assert outcome.gain[0] == 0.005
    assert outcome.candidate_deployed[0]
    assert np.array_equal(outcome.gain[1:], np.zeros(WORLD_COUNT - 1))
    assert not np.any(outcome.candidate_deployed[1:])


def test_slingshot_adapter_uses_guard_and_incumbent_columns() -> None:
    rewards = np.zeros((WORLD_COUNT, 7), dtype=np.float64)
    rewards[:, 2] = 0.01
    decisions = np.zeros((WORLD_COUNT, 4), dtype=np.int64)
    decisions[0, 3] = 2
    outcome = slingshot_outcome(decisions, rewards)
    assert outcome.gain[0] == 0.01
    assert outcome.candidate_deployed.sum() == 1
