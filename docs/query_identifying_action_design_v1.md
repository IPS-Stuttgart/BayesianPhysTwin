# Query-Identifying Action Design v1

`bayesian_phystwin.query_identifying_action_design_v1` ranks a finite,
externally supplied roster of physical probe actions by the expected reduction
of one registered physical-query covariance.

The module fills a different role from the existing active-observation tools:

- `phystwin_active_queries` chooses graph identities and image locations to
  track after an action and physical rollout are already fixed;
- `query_aware_anchor_planning` chooses observation anchors within a fixed
  experiment; and
- `query_identifying_action_design_v1` compares alternative action-conditioned
  prospective observation blocks before any action is executed.

The caller, typically an experiment protocol or Causal4D, remains responsible
for constructing the action roster and for physical safety. BayesianPhysTwin
only evaluates the declared local information model.

## Local model

For candidate action `a`, the prospective observation is

```text
y_a = H_a z + G_a n + epsilon_a
```

where `z` contains physical coefficients, `n` contains declared nuisance
coefficients, and `epsilon_a` has the candidate's conditional observation
covariance. The current joint Gaussian information state may already contain
state--nuisance cross information.

Each action is evaluated by:

1. adding its reliability-weighted observation block through
   `NuisanceAwareInformationState`;
2. marginalizing nuisance coefficients with the existing Schur-complement
   information operator;
3. projecting the resulting physical covariance into the registered query;
4. normalizing query covariance by a positive-definite query scale; and
5. adding caller-frozen dimensionless action cost and risk penalties.

No explicit covariance inverse is formed. Cholesky solves are used for
covariance recovery and observation whitening.

## Finite-roster decision

```python
import hashlib

import numpy as np

from bayesian_phystwin.query_identifying_action_design_v1 import (
    QueryIdentifyingActionCandidateV1,
    QueryIdentifyingActionDesignV1,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


candidates = [
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
]

decision = QueryIdentifyingActionDesignV1(
    prior_belief_id=digest("prior-belief"),
    query_id=digest("endpoint-query"),
    query_scale_id=digest("endpoint-query-scale"),
    protocol_id=digest("probe-design-protocol"),
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

print(decision.summary())
```

Candidate order cannot affect the content identity or decision. Exact objective
ties are resolved by lexicographic `action_id`. An action is never selected when
all candidates are unsafe, exceed the frozen risk limit, or fail to provide the
minimum objective improvement. A query with zero prior uncertainty produces an
explicit `trivial_query` no-action decision.

## Diagnostics

Each action evaluation reports:

- posterior query covariance;
- normalized query trace and maximum eigenvalue;
- query-trace reduction;
- an ideal no-nuisance reduction;
- the signed nuisance effect on query contraction (positive means a loss);
- nuisance-marginalized physical-state information gain in nats;
- the cost/risk-augmented objective; and
- the exact admission or rejection status.

The candidate, evaluation, and complete decision are content-addressed. Arrays
are copied into immutable float64 storage, metadata is frozen finite JSON, and
candidate ordering is canonicalized before the decision identity is computed.

## Scientific boundary

This module provides target-closed local design evidence only. It does not:

- generate a command or certify a physical safety envelope;
- execute an action or read an action outcome;
- establish global or nonlinear identifiability;
- validate a perception provider, simulator, or covariance calibration;
- authorize a physical-state interpretation;
- establish unseen-object transfer or deployment safety; or
- demonstrate downstream Causal4D benefit.

A claim-bearing experiment must separately freeze the action generator, safety
policy, physical and nuisance linearizations, query, query scale, reliability,
cost/risk semantics, nonlinear-closure evidence, cohort, and outcome-access
order before using the selected action.
