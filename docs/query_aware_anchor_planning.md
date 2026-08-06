# Query-aware anchor planning

`bayesian_phystwin.query_aware_anchor_planning` selects tactile, depth, contact,
or additional-view measurements for a declared physical query. It differs from
global information gain: a candidate is valuable only when it reduces the
query covariance after gauge, camera-bias, timing, and other nuisance variables
have been marginalized.

```python
import numpy as np

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
)
from bayesian_phystwin.query_aware_anchor_planning import (
    greedy_query_aware_selection,
)

prior = NuisanceAwareInformationState.from_independent_priors(
    state_precision=np.eye(2),
    nuisance_precision=np.eye(1),
)

plan = greedy_query_aware_selection(
    prior,
    query_jacobian=np.array([[1.0, 0.0]]),  # declared endpoint query
    state_jacobians=[np.array([[2.0, 0.0]]), np.array([[0.0, 4.0]])],
    nuisance_jacobians=[np.zeros((1, 1)), np.zeros((1, 1))],
    observation_covariances=[np.eye(1), np.eye(1)],
    costs=[1.0, 1.0],
    dependence_groups=["contact-a", "depth-capture-b"],
    count=1,
)
```

The planner ranks expected query-trace reduction per unit acquisition cost.
Candidates sharing a non-null dependence-group identifier are mutually
exclusive, which conservatively prevents duplicate points from one capture from
being counted as independent anchors. Ties are deterministic and resolve to the
lowest original candidate index.

## Contract checks

The returned `QueryAwareAnchorSelection` is a checked diagnostic contract rather
than an unconstrained report. It requires:

- one unique nonnegative integer index per selected candidate;
- one finite nonnegative query-trace reduction, score, and positive cost per
  selected candidate;
- `score_per_cost * selected_cost == query_trace_reduction` for every step;
- the sum of stepwise reductions to equal the initial-to-final query-trace
  change within the declared numerical tolerance;
- a valid `NuisanceAwareInformationState` as the final joint information state;
- no increase in the declared query variance; and
- bytes-backed retained arrays whose NumPy writeability cannot be restored.

`query_covariance(...)` applies the same immutable-storage rule. Invalid query
geometry, non-finite inputs, boolean counts or costs, incoherent direct contract
construction, and unhashable dependence groups fail closed.

## Evidence boundary

The plan is a source/calibration diagnostic, not outcome evidence. Candidate
Jacobians, costs, reliabilities, dependence groups, and the query definition
must be frozen before a confirmation cohort is opened. Selecting an anchor
does not establish provider competence, realized physical-query improvement,
calibrated coverage, deployment safety, Causal4D benefit, or state of the art.
