# Native block-tree information and structured Schur covariance

## Scope

BayesianPhysTwin includes additive NumPy primitives for exact inference on a
causally ordered block tree without constructing the complete dense gauge
information matrix. The primitives are intentionally module-scoped. Existing
dense, native-sparse, structured-fallback, and claim-bearing update entry points
are unchanged by this implementation slice.

The two public module-level classes are:

- `TreeBlockInformationFactorV1`, an exact factor and solver for a symmetric
  positive-definite block-tree information matrix; and
- `TreeSchurCovarianceV1`, an accepted-posterior covariance operator backed by a
  small Schur complement and the tree factor.

They establish the numerical base for a later tree-native prior-aware update.
They do not themselves change a registered estimator or scientific result.

## Block-tree factorization

For node blocks `g_i` with a topological parent vector, the information matrix
contains one diagonal block per node and one child-parent block per non-root
edge. Reverse topological elimination factors each child block and applies its
exact Schur update to the parent. No array with both dimensions proportional to
the total gauge dimension is required.

For a fixed block size `D` and `N` tree nodes, storage is `O(N D^2)` and the
factorization cost is `O(N D^3)`. Applying the inverse to `K` right-hand sides
costs `O(N D^2 K)`. The factor supports:

```python
from bayesian_phystwin.tree_block_information import (
    TreeBlockInformationFactorV1,
)

factor = TreeBlockInformationFactorV1.from_transition_innovation(
    parent_indices=parents,
    transition_matrices=transitions,
    innovation_scale_tril=innovation_scales,
    local_information_blocks=local_information,
)

solution = factor.solve(right_hand_side)
selected_covariance = factor.marginal_covariance([0, 4, 9])
```

The transition builder represents

```text
g_i - T_i g_parent ~ Normal(0, L_i L_i.T)
```

for non-root nodes and an analogous root prior. Local observation information
may be added to each node diagonal before factorization.

## Structured accepted covariance

Consider a reduced information system with a small retained core `c` and tree
variables `g`:

```text
[ C   B.T ]
[ B    G  ]
```

The tree factor supplies `G^-1 B`, while the caller supplies the small core Schur
covariance

```text
S^-1 = (C - B.T G^-1 B)^-1.
```

`TreeSchurCovarianceV1` retains those quantities together with the physical-state
prior and the retained-state mapping. It can apply the complete posterior
covariance, evaluate registered query covariance, and return selected
coefficient marginals without constructing the full joint covariance:

```python
from bayesian_phystwin.tree_schur_covariance import TreeSchurCovarianceV1

posterior_covariance = TreeSchurCovarianceV1(
    state_prior_covariance=state_prior,
    state_mapping=retained_state_mapping,
    core_covariance=schur_covariance,
    tree_factor=factor,
    tree_core_solve=tree_core_solve,
)

query_covariance = posterior_covariance.query_covariance(query_jacobian)
selected = posterior_covariance.marginal_covariance([0, 5, 11])
```

Dense conversion remains available only as an explicit compatibility operation.
Both classes estimate the required dense byte count before allocation and reject
a materialization that exceeds the caller-supplied budget.

## Validation boundary

The regression suite compares solves, matrix products, log determinants,
quadratic forms, selected inverse blocks, covariance applications, registered
queries, and selected posterior marginals against dense NumPy references. It
also exercises a 2,048-node chain while requiring stored factor bytes to remain
below one percent of the estimated dense information bytes.

These checks establish numerical parity, validation behavior, and the intended
allocation boundary. They do not establish provider competence, covariance
calibration, physical-query benefit, intervention benefit, deployment safety,
or state of the art.
