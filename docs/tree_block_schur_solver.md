# Tree-block Schur solver for claim-bearing Prob4D updates

The earlier tree-sparse adapter avoided the dense Prob4D prior covariance and the
dense row design, but its numerical update still assembled the complete gauge
precision and joint nuisance normal matrix. Both grow quadratically with the
number of causal windows.

The additive solver in
`bayesian_phystwin.tree_block_sparse_gauge_belief` preserves the gauge tree as
block factors through identifiability analysis, every robust IRLS iteration, the
final solve, and posterior uncertainty representation.

## Normal-system structure

Let `x` contain the retained physical-state modes and the small shared, view and
anchor-bias variables. Let `g_i` be the local seven-parameter gauge of tree node
`i`. The weighted normal equations have:

- one small dense global block for `x`;
- one `7 x 7` diagonal block per gauge;
- one `7 x 7` child-parent coupling per non-root gauge;
- one `7 x dim(x)` gauge-global coupling per gauge; and
- no sibling or non-tree gauge coupling.

The solver eliminates gauges in reverse causal order. Each child sends an exact
Schur message to its parent and the global block. After the root is eliminated,
only the small global Schur complement is factorized densely. A forward pass
then recovers every gauge.

No `7K x 7K` tree precision, full nuisance matrix or complete joint normal matrix
is assembled.

## Identifiability

The same elimination is used before selecting state modes. Gauge variables are
marginalized by leaf-to-root elimination. Remaining shared/view/anchor biases
are eliminated from the small global block, and the resulting conditional state
information is standardized by the physical-state prior exactly as in the
existing prior-aware solver.

## Robust fixed point

The grouped nominal/outlier Student-t responsibilities and generalized-Bayes
powers are unchanged. Every IRLS iteration:

1. constructs block-local weighted factors;
2. eliminates the gauge tree;
3. solves the global Schur complement;
4. back-substitutes gauges;
5. refreshes group responsibilities; and
6. checks both solution movement and the structured normal-equation residual.

## Posterior uncertainty

`TreeBlockPosteriorCovarianceV1` retains:

- the physical-state prior covariance and retained state mapping;
- the posterior node Cholesky factors;
- child-parent and node-global couplings; and
- the final global Schur Cholesky factor.

The state marginal can be obtained without constructing gauge covariance:

```python
state_covariance = result.covariance.state_marginal_covariance()
```

A complete covariance is an explicit, peak-memory-budgeted compatibility
operation:

```python
legacy = result.to_legacy(maximum_covariance_bytes=512 * 1024 * 1024)
```

The result identity hashes the block factors directly and does not materialize
the covariance.

## Claim-bearing surface

Use:

```python
from bayesian_phystwin.tree_block_sparse_prob4d import (
    update_claim_bearing_tree_block_prob4d_from_path,
)

update = update_claim_bearing_tree_block_prob4d_from_path(
    "claim-bearing-tree-sparse.json",
    linearization,
    physical_prediction_xyz_m=physical_prediction,
)
```

The wrapper binds the observation and linearization artifacts, provider
manifest, calibration artifacts, independently verified runtime revision,
accept/reject reason, and tree-factorized numerical result.

## Compatibility and claim boundary

The existing dense and tree-sparse solver functions are unchanged. This is an
opt-in module-scoped surface for the BayesianPhysTwin 0.4 line.

The implementation improves numerical scaling and provenance. It does not
establish observation-provider competence, calibrated uncertainty,
physical-query benefit, intervention benefit, deployment safety or state of the
art.
