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

## Support--precision sufficiency curves

`bayesian_phystwin.query_anchor_sufficiency` evaluates how much independent
anchor support and precision are needed to reduce one declared query variance.
The candidate identities, Jacobians, nuisance designs, reliabilities, costs,
dependence groups, and order remain fixed. A precision multiplier scales
information by dividing every declared observation covariance by that value.

```python
from bayesian_phystwin.query_anchor_sufficiency import (
    evaluate_query_anchor_sufficiency,
)

curve = evaluate_query_anchor_sufficiency(
    prior,
    query_jacobian=np.array([[1.0, 0.0]]),
    state_jacobians=[np.array([[2.0, 0.0]]), np.array([[0.0, 4.0]])],
    nuisance_jacobians=[np.zeros((1, 1)), np.zeros((1, 1))],
    observation_covariances=[np.eye(1), np.eye(1)],
    precision_multipliers=[0.5, 1.0, 2.0, 4.0],
    costs=[1.0, 1.0],
    maximum_count=2,
    target_remaining_variance_fraction=0.25,
)

print(curve.first_sufficient_support)
print(curve.remaining_variance_fractions)
```

For each precision multiplier, the greedy planner runs once to the maximum
support. Every smaller support value is an exact prefix of that plan. The
immutable result retains:

- the selected candidate order and actual selected count;
- query-variance traces from support zero through the maximum support;
- cumulative declared acquisition cost;
- the fraction of support-zero variance remaining; and
- the first support count that reaches the frozen target, or `-1` if the
  candidate set cannot reach it.

Selection can stop before the requested maximum when dependence groups exhaust
independent candidates or no candidate exceeds the frozen minimum reduction.
The curve then remains exactly constant rather than pretending that unavailable
anchors add support.

A deterministic controlled study demonstrates the difference between global
state information and query-specific information:

```bash
python scripts/science/run_query_anchor_sufficiency_study.py \
  --output-dir outputs/query-anchor-sufficiency
```

The study writes `summary.json`, `curve.csv`, and `report.md` without replacing
existing outputs unless `--force` is supplied. Its protocol includes a highly
informative but query-irrelevant measurement, coherent visual and timing
nuisances, and independent metric anchors. The result is exercised by the
read-only query-aware-anchor workflow and is retained only as controlled
planner-mechanism evidence.

## Evidence boundary

The plan and support--precision curve are source/calibration diagnostics, not
outcome evidence. Candidate Jacobians, costs, reliabilities, dependence groups,
precision sweep, support limit, query definition, and target fraction must be
frozen before a confirmation cohort is opened. Model-based variance reduction
does not establish provider competence, realized physical-query improvement,
calibrated coverage, deployment safety, Causal4D benefit, or state of the art.
