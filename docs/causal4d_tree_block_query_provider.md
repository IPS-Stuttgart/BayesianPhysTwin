# Causal4D tree-block posterior query provider

`bayesian_phystwin.causal4d_tree_block_provider_v1` is the versioned public
boundary through which Causal4D may evaluate uncertainty queries on a
claim-bearing tree-block Prob4D update.

The provider is additive. It does not replace the fixed-anchor endpoint,
model-averaged horizon-discrepancy, replay, graph, artifact, or public-study
providers.

## Admitted input

The provider accepts only a `ClaimBearingTreeBlockProb4DUpdateV1`. Before using
its covariance it independently:

- validates every compact tree factor and condition diagnostic;
- requires strict tree-block admission version 2;
- reconstructs the claim-bearing wrapper from its public fields;
- rechecks the admission, update, and tree-block result identities; and
- checks that the registered query is finite, real-valued, nonempty, and has the
  exact public coefficient dimension.

The public coefficient order remains:

```text
physical state, flattened gauge nodes, shared/view/anchor biases
```

## Query result

Use:

```python
from bayesian_phystwin.causal4d_tree_block_provider_v1 import (
    evaluate_claim_bearing_tree_block_query,
)

query_result = evaluate_claim_bearing_tree_block_query(
    update,
    query_matrix,
    query_id=registered_query_sha256,
)
```

The immutable `Causal4DTreeBlockQueryCovarianceV1` binds:

- the complete claim-bearing update identity;
- the strict tree-block result identity;
- the caller-owned registered query identity;
- the actual query-matrix content digest;
- the complete coefficient dimension;
- the accepted or rejected inference status and reason; and
- the exact factorized linear-query covariance.

A rejected strict update remains queryable because its result carries the exact
physical and nuisance prior fallback. The result does not relabel that fallback
as accepted evidence.

## Allocation boundary

The provider evaluates `Q P Q.T` through
`TreeBlockPosteriorOperatorV1`. It allocates storage proportional to the full
coefficient dimension times the number of query rows. It does not construct the
complete joint covariance and does not call the explicit dense compatibility
path.

## Manifest

`causal4d_tree_block_provider_manifest()` reports API version 1, the exact
artifact schema versions, and capabilities for strict update validation,
identity-bound factorized queries, immutable results, and absence of dense
covariance materialization.

## Scientific boundary

The provider establishes factor integrity, strict-admission lineage, query
identity, and exact numerical covariance application. The covariance remains the
admitted working Gauss-Newton/IRLS covariance. Observation competence, empirical
uncertainty calibration, target-side coverage, physical-query benefit,
intervention benefit, deployment safety, and state of the art require separate
registered evidence.
