"""Exact world-level adapters for the two registered portfolio components."""

from __future__ import annotations

import numpy as np

from .query_portfolio_replication_v1 import WORLD_COUNT, QueryOutcomeV1


def wrapping_outcome(
    decisions: object,
    rewards: object,
    *,
    primary_arm_index: int,
) -> QueryOutcomeV1:
    """Recover equal-world Wrapping gains from sealed decisions and rewards."""

    decision = np.asarray(decisions)
    reward = np.asarray(rewards, dtype=np.float64)
    if (
        decision.ndim != 3
        or decision.shape[0] != WORLD_COUNT
        or reward.shape != (WORLD_COUNT, 8)
        or decision.shape[2] <= primary_arm_index
        or decision.dtype.kind not in "iu"
        or np.any((decision < 0) | (decision >= reward.shape[1]))
        or not np.isfinite(reward).all()
    ):
        raise ValueError("complete aligned Wrapping decisions and rewards required")
    selected = np.take_along_axis(reward[:, None, :], decision, axis=2).mean(axis=1)
    gain = selected[:, primary_arm_index] - selected[:, 0]
    deployed = np.any(
        decision[:, :, primary_arm_index] != decision[:, :, 0], axis=1
    )
    return QueryOutcomeV1(
        query_id="dlolab_wrapping_v9",
        gain=gain,
        candidate_deployed=deployed,
        ordinary_success=np.ones(WORLD_COUNT, dtype=np.bool_),
    )


def slingshot_outcome(decisions: object, rewards: object) -> QueryOutcomeV1:
    """Recover equal-world Slingshot gains from sealed decisions and rewards."""

    decision = np.asarray(decisions)
    reward = np.asarray(rewards, dtype=np.float64)
    if (
        decision.shape != (WORLD_COUNT, 4)
        or reward.shape != (WORLD_COUNT, 7)
        or decision.dtype.kind not in "iu"
        or np.any((decision < 0) | (decision >= reward.shape[1]))
        or not np.isfinite(reward).all()
    ):
        raise ValueError("complete aligned Slingshot decisions and rewards required")
    selected = np.take_along_axis(reward, decision, axis=1)
    gain = selected[:, 3] - selected[:, 0]
    deployed = decision[:, 3] != decision[:, 0]
    return QueryOutcomeV1(
        query_id="dlolab_slingshot_v4",
        gain=gain,
        candidate_deployed=deployed,
        ordinary_success=np.ones(WORLD_COUNT, dtype=np.bool_),
    )
