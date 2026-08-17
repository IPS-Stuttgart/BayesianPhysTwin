# Block-tree Gaussian solver

`bayesian_phystwin.tree_separator_gaussian` provides an exact Gaussian solver
for a fixed-size block tree or forest coupled to a small dense separator.

It is the numerical foundation for a future accepted tree-sparse Prob4D update.
The existing claim-bearing solver is intentionally unchanged until an adapter
has demonstrated complete numerical parity and preserved all admission and
fallback contracts.

## System form

For tree-node variables `x_i` and a small separator `y`, the module solves

```text
[ H_tree  C ] [x] = [h_x]
[ C^T     G ] [y]   [h_y]
```

where `H_tree` contains only node-diagonal blocks and parent-child precision
blocks. The separator can hold the physical-state coefficients and genuinely
global nuisance variables while each gauge window remains a fixed-size tree
node.

`TreeSeparatorGaussianSystemV1` stores:

- one parent index per node;
- one node-diagonal precision block per node;
- one parent-cross precision block per non-root node;
- one node-to-separator precision block per node;
- a small dense separator precision;
- node and separator information vectors.

Parents must precede children. Roots use parent index `-1` and a zero
parent-cross block.

## Exact elimination

`solve_tree_separator_gaussian` performs:

1. reverse-order leaf-to-root block elimination;
2. a dense solve only for the small separator Schur complement;
3. root-to-leaf mean back-substitution; and
4. forward marginal-covariance propagation.

The returned result contains all node means, all node marginal covariances,
node-to-separator cross-covariances, the separator mean and covariance, and the
precision log determinant. It does not construct the complete tree covariance
or precision matrix.

For fixed node block size `d` and separator size `q`, storage is
`O(K(d^2 + dq) + q^2)`. The dominant arithmetic is approximately
`O(K(d^3 + d^2 q + d q^2) + q^3)`.

## Dense compatibility boundary

`TreeSeparatorGaussianSystemV1.to_dense` exists only for small compatibility
checks and diagnostics. It requires an explicit byte budget and raises
`MemoryError` before allocation when the complete precision would exceed that
budget.

```python
precision, information = system.to_dense(maximum_bytes=8_000_000)
```

Production tree-sparse execution should call:

```python
from bayesian_phystwin.tree_separator_gaussian import (
    solve_tree_separator_gaussian,
)

result = solve_tree_separator_gaussian(system)
```

## Integration boundary

A later claim-bearing adapter must still prove all of the following before this
primitive replaces the historical accepted-update solve:

- exact coefficient and covariance parity on small frozen fixtures;
- identical robust-IRLS convergence and strict-v2 admission decisions;
- exact physical fallback for every rejection;
- preservation of source lineage, evidence-separation diagnostics, and
  covariance ownership;
- bounded memory on large accepted tree-sparse updates; and
- installed-wheel Prob4D-to-BayesianPhysTwin-to-Causal4D compatibility.

The primitive itself makes no claim about observation competence, provider
calibration, physical-query benefit, intervention benefit, deployment safety,
or state of the art.
