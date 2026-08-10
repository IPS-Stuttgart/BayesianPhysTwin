# Tree-block sparse-gauge IRLS parity trace

`bayesian_phystwin.tree_block_sparse_gauge_parity_trace` adds an opt-in numerical
shadow trace around the production tree-block robust update.

The ordinary entry point remains unchanged:

```python
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    update_tree_block_sparse_prior_aware_gauge_belief,
)
```

It does not import the parity trace, execute the independent solver, alter
production diagnostics, or change the result content address.

## Traced entry point

Use the traced entry point only when an experiment or compatibility gate needs
independent numerical evidence for the production IRLS systems:

```python
from bayesian_phystwin.tree_block_sparse_gauge_parity_trace import (
    update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace,
)

trace = update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
    batch,
    tree_gauge,
    config=config,
)
result = trace.result
```

The production solver still performs every authoritative operation. For each
IRLS iteration it:

1. builds the production `TreeBlockNormalSystemV1`;
2. applies the historical node-elimination and condition-number gate;
3. reports the admitted solve system to the parity observer;
4. solves and refreshes the robust mixture statistics;
5. builds and condition-checks the final system for that iteration; and
6. reports the admitted final system to the parity observer.

The observer calls `require_tree_separator_gaussian_parity`. A numerical mismatch
raises before a traced result can be returned. The independent solver never
selects the production coefficients, covariance, robust weights, convergence
decision, admission decision, fallback reason, or result identity.

## Trace contents

`TreeBlockSparseGaugeParityTraceV1` binds:

- the exact production `result_id`, admissibility flag, and reason;
- the production maximum-condition-number limit;
- the relative and absolute shadow tolerances;
- an ordered `irls-solve` and `irls-final` step sequence;
- each complete content-addressed parity report;
- the number of observed iterations and steps; and
- the sum of dense precision bytes avoided by the shadow evaluations.

Each parity report already binds the exact normal-system array bytes, production
and independent solver identities, selected node covariance roster, means,
selected covariance blocks, log determinant, structured residual, and scaled
errors.

For an admissible result, the trace must finish with a complete `irls-final`
step. Early physical fallback before an admitted IRLS system produces an empty
trace bound to the exact fallback result. If a later production factorization is
rejected, the trace may contain only the earlier admitted systems; it cannot turn
a rejected production update into an accepted one.

## Identity preservation

The traced function uses the same private production engine as the historical
entry point. The observer receives read-only normal-system contracts after the
production factorization gate. Tests require the traced result descriptor,
result ID, covariance descriptor, diagnostics, coefficients, and robust weights
to equal the ordinary result exactly.

The trace is a separate artifact. It is deliberately not inserted into the
production result diagnostics or input lineage, because doing so would change the
historical result content address. The caller also owns any persistence or
publication of the trace; merely invoking the traced entry point does not write
an artifact or mutate the input contracts.

## Cost boundary

Shadow parity runs an additional exact block-tree solve for every admitted solve
and final system. It therefore belongs in verification, compatibility, and
scientific-evidence jobs rather than the default runtime path. Selected node
covariances remain bounded by the parity module's deterministic maximum-eight
node policy unless an explicit frozen node roster is supplied.

## Scientific boundary

This trace establishes only that an independent bounded-memory solver agrees
with production-admitted normal systems during one robust update. It does not
establish observation-provider competence, empirical covariance calibration,
physical-prediction benefit, Causal4D intervention benefit, deployment safety,
generalization, or state of the art.
