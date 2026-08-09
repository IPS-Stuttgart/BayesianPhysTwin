# Claim-bearing tree-sparse Prob4D observations

BayesianPhysTwin can admit the portable tree-sparse observation artifact added by
Prob4D pull request 184 and merged at
`b2953319e9b7afea04013c214c502b38c5a83489`.

The integration is implemented in
`bayesian_phystwin.tree_sparse_explicit_gauge_prob4d`. It is deliberately
separate from the older schema-v4 dense compatibility bridge and from the
native-sparse bridge that still consumes a materialized joint gauge covariance.

## Admission boundary

The producer-owned Prob4D loader first validates the portable artifact. The
BayesianPhysTwin adapter then independently revalidates the evidence needed for
a physical-state update:

- the claim-bearing envelope and inner observation artifact identities;
- the frozen Prob4D source repository and exact source revision;
- provider API version 2, the required tree-sparse capabilities, and both
  schema versions;
- the complete provider attestation, calibration artifact IDs, and independently
  verified runtime revision;
- sequence, case, stream, observation count, gauge order, causal cutoff, sparse
  prior artifact ID, and semantic prior ID;
- the selected source-window lineage and the requirement that every observation
  frame lies inside the interval assigned to its gauge;
- finite row means, positive-definite conditional point covariances, local
  `3 x 7` gauge Jacobians, row identities, probabilities, and group settings;
- causal-tree parent ordering, transition matrices, lower-triangular innovation
  factors, and the declared tree-prior semantics; and
- exact alignment of frame, entity, view, and gauge indices with the registered
  `PhysicalLinearizationV1`.

The adapter fails closed before forming an innovation when any of these checks
changes.

## Precision-form gauge prior

For gauge state `g_k`, parent `p(k)`, transition `F_k`, and innovation covariance
`Q_k = L_k L_k^T`, the prior is represented by

```text
g_0 ~ N(0, Q_0)
g_k | g_p(k) ~ N(F_k g_p(k), Q_k).
```

`TreeSparseGaugeDesignV1` assembles the exact block information matrix from the
transition and innovation factors. The accepted update path does not construct
or invert the dense joint `7K x 7K` gauge covariance. It also avoids the dense
row design of shape `M x 3 x 7K`; each row keeps only its local `3 x 7` Jacobian
and one gauge index.

The diagnostics report:

- `gauge_prior_representation = tree-transition-innovation-information-v1`;
- `dense_gauge_prior_covariance_materialized = false`;
- the avoided dense-prior and dense-design byte counts; and
- the retained tree-factor storage size.

A rejected update may still materialize a dense prior lazily because the legacy
`GaugeAwareBeliefResult` fallback contract returns a complete posterior/prior
covariance. This does not occur on the accepted precision-form path.

## Association and group likelihood power

The tree-sparse Prob4D contract keeps two distinct quantities:

- `association_probability` is a row-level generalized-Bayes power and may vary
  between rows in the same correlation group;
- `composite_weight` is the group-level likelihood power and must be constant
  within a correlation group, as must the nominal component probability.

BayesianPhysTwin therefore applies

```text
row precision weight = prior_reliability * association_probability
```

inside the grouped residual norm, while applying `composite_weight` once to the
group likelihood. Source reliability is not replaced by association
probability.

The two older Prob4D adapters retain their pre-existing final-power convention,
where the producer-facing compatibility layer supplies `association_probability
* composite_weight` as one already-finalized group power. Their numerical
behavior and frozen contracts are unchanged.

## Public functions

The module-level API provides:

- `load_claim_bearing_tree_sparse_prob4d(path)`;
- `build_claim_bearing_tree_sparse_prob4d_batch(...)`;
- `update_claim_bearing_tree_sparse_prob4d_from_artifacts(...)`; and
- `update_claim_bearing_tree_sparse_prob4d_from_path(...)`.

Prob4D is imported lazily only when the path loader is called. Importing
BayesianPhysTwin or the adapter module therefore does not load Prob4D or widen
the stable root-package API.

## Validation

The dedicated read-only workflow checks:

- Ruff formatting and lint;
- strict MyPy on the affected numerical and adapter surfaces;
- dense, sparse, observed-information, legacy Prob4D, and tree-sparse contract
  tests;
- exact precision parity against a test-only materialized tree covariance; and
- a real serialized producer-to-consumer round trip using the exact merged
  Prob4D revision.

The real integration test creates a content-addressed Prob4D artifact, reloads
it through the strict producer surface, admits it independently in
BayesianPhysTwin, and executes the precision-form update. The unit and serialized
round-trip tests are both registered in the centralized `stable-core-coverage`
and `core-contracts` suites.

## Scientific boundary

A successful load and update establishes artifact identity, causal information
order, provider and calibration provenance, exact numerical consumption of the
registered tree prior, and the absence of a dense prior covariance on the
accepted path. It does **not** establish Prob4D competence on fresh objects,
empirical uncertainty calibration, physical-query benefit, intervention
benefit, deployment safety, or state of the art. Those claims require separate
registered evidence.
