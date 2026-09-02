from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.query_portfolio_evidence_v2 import (
    assemble,
    component_evidence,
    load_component_evidence,
)
from bayesian_phystwin.query_portfolio_replication_v1 import QueryOutcomeV1
from bayesian_phystwin.query_portfolio_replication_v5 import protocol

DIGEST = "a" * 64


def _outcome(query_id: str, gain: float) -> QueryOutcomeV1:
    return QueryOutcomeV1(
        query_id=query_id,
        gain=np.full(320, gain, dtype=np.float64),
        candidate_deployed=np.ones(320, dtype=np.bool_),
        ordinary_success=np.ones(320, dtype=np.bool_),
    )


def test_v2_component_binds_v5_protocol_and_round_trips() -> None:
    value = component_evidence(
        _outcome("dlolab_wrapping_v9", 0.01),
        component_result_id=DIGEST,
        component_result_sha256=DIGEST,
    )
    assert value["version"] == 2
    assert value["portfolio_protocol_id"] == protocol()["protocol_id"]
    recovered = load_component_evidence(value)
    np.testing.assert_array_equal(recovered.gain, np.full(320, 0.01))


def test_v2_rejects_v1_or_tampered_component() -> None:
    value = component_evidence(
        _outcome("dlolab_wrapping_v9", 0.01),
        component_result_id=DIGEST,
        component_result_sha256=DIGEST,
    )
    value["version"] = 1
    with pytest.raises(ValueError, match="invalid v5"):
        load_component_evidence(value)


def test_v2_assembles_only_complete_two_query_evidence() -> None:
    records = {
        query: component_evidence(
            _outcome(query, 0.01),
            component_result_id=DIGEST,
            component_result_sha256=DIGEST,
        )
        for query in ("dlolab_wrapping_v9", "dlolab_slingshot_v4")
    }
    result = assemble(records)
    assert result["joint_portfolio_claim_passed"] is True
    assert result["partial_results_used"] is False
    assert result["protocol_id"] == protocol()["protocol_id"]
