from __future__ import annotations

import copy

import numpy as np
import pytest

from bayesian_phystwin.query_portfolio_evidence_v1 import (
    assemble,
    component_evidence,
    load_component_evidence,
)
from bayesian_phystwin.query_portfolio_replication_v1 import (
    QUERY_IDS,
    WORLD_COUNT,
    QueryOutcomeV1,
)


def _record(query_id: str, gain: float = 0.004) -> dict:
    outcome = QueryOutcomeV1(
        query_id=query_id,
        gain=np.full(WORLD_COUNT, gain),
        candidate_deployed=np.ones(WORLD_COUNT, dtype=np.bool_),
        ordinary_success=np.ones(WORLD_COUNT, dtype=np.bool_),
    )
    return component_evidence(
        outcome,
        component_result_id="1" * 64,
        component_result_sha256="2" * 64,
    )


def test_joint_evidence_preserves_component_lineage() -> None:
    records = {query_id: _record(query_id) for query_id in QUERY_IDS}
    result = assemble(records)
    assert result["joint_portfolio_claim_passed"] is True
    assert result["partial_results_used"] is False
    assert set(result["component_evidence"]) == set(QUERY_IDS)


def test_component_identity_and_fallback_are_fail_closed() -> None:
    record = _record(QUERY_IDS[0])
    changed = copy.deepcopy(record)
    changed["gain"][0] = 1.0
    with pytest.raises(ValueError, match="identity changed"):
        load_component_evidence(changed)
    changed = copy.deepcopy(record)
    changed["candidate_deployed"][0] = False
    changed.pop("artifact_id")
    from bayesian_phystwin._portable_contracts import content_id

    changed["artifact_id"] = content_id(changed)
    with pytest.raises(ValueError, match="exact zero gain"):
        load_component_evidence(changed)
