from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.query_portfolio_replication_v1 import (
    QUERY_IDS,
    WORLD_COUNT,
    QueryOutcomeV1,
    maximum_harms_allowed,
    protocol,
    score,
)


def _outcome(query_id: str, *, gain: float = 0.004) -> QueryOutcomeV1:
    return QueryOutcomeV1(
        query_id=query_id,
        gain=np.full(WORLD_COUNT, gain),
        candidate_deployed=np.ones(WORLD_COUNT, dtype=np.bool_),
        ordinary_success=np.ones(WORLD_COUNT, dtype=np.bool_),
    )


def test_protocol_has_single_joint_95_percent_budget() -> None:
    value = protocol()
    assert value["joint_claim_confidence"] == 0.95
    assert value["allocation"]["gain_family_alpha"] == 0.01
    assert value["allocation"]["harm_family_alpha"] == 0.04
    assert value["cross_task_reward_pooling"] is False
    assert value["outcomes_opened"] is False
    assert maximum_harms_allowed() == 8


def test_complete_positive_portfolio_passes() -> None:
    result = score({query_id: _outcome(query_id) for query_id in QUERY_IDS})
    assert result["joint_portfolio_claim_passed"] is True
    assert result["complete_denominator"] is True


def test_fallback_must_be_byte_exact_in_reward() -> None:
    deployed = np.ones(WORLD_COUNT, dtype=np.bool_)
    deployed[0] = False
    with pytest.raises(ValueError, match="exact zero gain"):
        QueryOutcomeV1(
            query_id=QUERY_IDS[0],
            gain=np.full(WORLD_COUNT, 0.004),
            candidate_deployed=deployed,
            ordinary_success=np.ones(WORLD_COUNT, dtype=np.bool_),
        )


def test_incomplete_denominator_fails_closed() -> None:
    success = np.ones(WORLD_COUNT, dtype=np.bool_)
    success[-1] = False
    with pytest.raises(ValueError, match="ordinary-success"):
        QueryOutcomeV1(
            query_id=QUERY_IDS[0],
            gain=np.zeros(WORLD_COUNT),
            candidate_deployed=np.zeros(WORLD_COUNT, dtype=np.bool_),
            ordinary_success=success,
        )
