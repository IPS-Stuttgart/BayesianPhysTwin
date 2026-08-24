"""Rank two target-closed probe actions by registered-query contraction."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from bayesian_phystwin.query_identifying_action_design_v1 import (
    QueryIdentifyingActionCandidateV1,
    QueryIdentifyingActionDesignV1,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def main() -> None:
    candidates = (
        QueryIdentifyingActionCandidateV1(
            action_id="small-lift",
            state_jacobian=np.array([[1.2, 0.0]]),
            nuisance_jacobian=np.array([[0.1]]),
            observation_covariance=np.array([[0.25]]),
            reliability=0.9,
            dimensionless_cost=0.1,
            dimensionless_risk=0.05,
        ),
        QueryIdentifyingActionCandidateV1(
            action_id="lateral-pull",
            state_jacobian=np.array([[0.4, 0.0]]),
            nuisance_jacobian=np.array([[1.0]]),
            observation_covariance=np.array([[0.25]]),
            reliability=0.9,
            dimensionless_cost=0.1,
            dimensionless_risk=0.05,
        ),
    )
    decision = QueryIdentifyingActionDesignV1(
        prior_belief_id=_digest("prior-belief"),
        query_id=_digest("endpoint-query"),
        query_scale_id=_digest("endpoint-query-scale"),
        protocol_id=_digest("probe-design-protocol"),
        prior_state_precision=np.eye(2),
        prior_nuisance_precision=np.array([[0.05]]),
        prior_state_nuisance_precision=np.zeros((2, 1)),
        query_jacobian=np.array([[1.0, 0.0]]),
        query_scale=np.array([[1.0]]),
        candidates=candidates,
        cost_weight=0.2,
        risk_weight=0.5,
        maximum_risk=0.2,
        minimum_objective_improvement=0.01,
    )
    print(json.dumps(decision.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
