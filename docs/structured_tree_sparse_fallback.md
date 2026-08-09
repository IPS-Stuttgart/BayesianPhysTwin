# Structured tree-sparse fallback results

The historical `GaugeAwareBeliefResult` returns one complete dense posterior
covariance for both accepted and rejected updates. That behavior remains frozen
for compatibility. In a tree-sparse Prob4D update, however, a rejected candidate
should not need to invert and materialize the complete gauge prior merely to
report exact physical fallback.

BayesianPhysTwin therefore provides the additive version-1 structured result in
`bayesian_phystwin.structured_gauge_aware_result` and the claim-bearing tree
adapter in `bayesian_phystwin.tree_sparse_structured_gauge_prob4d`.

## Covariance representations

`DenseCovarianceV1` contains an already materialized complete covariance. It is
used by accepted version-1 structured results because the current accepted
solver already computes its complete local posterior covariance.

`PrecisionBackedCovarianceV1` stores:

- the normally small physical-state prior covariance;
- the joint gauge and other nuisance prior precision;
- an optional nuisance covariance only for a legacy dense-prior input.

For the claim-bearing causal gauge tree, the optional nuisance covariance is
absent. Constructing or hashing a rejected result therefore performs no dense
prior inversion and allocates no complete block covariance.

The representation reports its dimension, stored bytes and estimated dense
materialization bytes. A caller must materialize deliberately:

```python
covariance = result.materialize_posterior_covariance(
    maximum_bytes=256 * 1024 * 1024,
)
```

The operation raises `MemoryError` before allocation when the declared limit is
insufficient. `result.to_legacy(maximum_covariance_bytes=...)` performs the same
explicit conversion to `GaugeAwareBeliefResult`.

## Solver surfaces

The existing function remains unchanged:

```python
update_sparse_prior_aware_gauge_belief(...)
```

It returns the historical dense result and may materialize the prior covariance
when an update is rejected.

The additive function is:

```python
update_sparse_prior_aware_gauge_belief_structured(...)
```

It runs the same numerical implementation and decision logic. Accepted outputs
are numerically identical and carry a `DenseCovarianceV1`. Rejected outputs
carry a `PrecisionBackedCovarianceV1` and retain exact zero candidate
coefficients without materializing the dense prior.

The mode selection is held in a context-local token, so concurrent calls cannot
change one another's result contract.

## Claim-bearing Prob4D surface

Use:

```python
from bayesian_phystwin.tree_sparse_structured_gauge_prob4d import (
    update_claim_bearing_tree_sparse_prob4d_structured_from_path,
)

update = update_claim_bearing_tree_sparse_prob4d_structured_from_path(
    "claim-bearing-tree-sparse.json",
    linearization,
    physical_prediction_xyz_m=physical_prediction,
)
```

`ClaimBearingTreeSparseProb4DUpdateV2` binds:

- the admitted observation and physical-linearization artifact IDs;
- provider manifest, calibration artifacts and independently verified runtime
  revision;
- the structured numerical-result identity;
- covariance representation and materialization state; and
- the exact accept/reject reason.

Its content identity hashes covariance factors directly. It does not materialize
the covariance. `to_legacy(...)` is the only compatibility conversion.

## Claim boundary

This contract improves memory behavior, explicitness and auditability. It does
not establish Prob4D provider competence, physical-query benefit, calibrated
uncertainty, deployment safety, intervention benefit or state of the art. Exact
fallback means that a rejected candidate preserves the declared baseline; it is
not a universal safety theorem.
