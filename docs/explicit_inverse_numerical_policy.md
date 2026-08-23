# Explicit-inverse numerical policy

BayesianPhysTwin treats an explicit matrix inverse as an exceptional exported
quantity, not as the default way to solve a linear system. New production code
must prefer a factorization or operator action:

- use `SPDSystem.solve()` for admitted symmetric positive-definite systems;
- use `SPDSystem.whiten()` for Mahalanobis residuals and quadratic forms;
- use `numpy.linalg.solve()` when the system is not represented by
  `SPDSystem`;
- use the structured covariance and tree-block operators for large correlated
  beliefs; and
- reconstruct an inverse only when a versioned public artifact explicitly
  requires the complete covariance matrix.

This policy improves numerical stability and avoids unnecessary cubic work and
memory. It does not imply that every historical explicit inverse is incorrect.
Several existing calls invert small camera transforms, while others belong to
frozen experiment implementations whose numerical outputs are part of retained
evidence.

## Enforced ratchet

`tests/test_explicit_inverse_ratchet.py` parses every Python module under
`src/bayesian_phystwin` and records NumPy inverse calls by file and count. The
baseline currently contains 24 calls in 14 modules. The detector recognizes
common import forms including:

```python
import numpy as np
np.linalg.inv(matrix)

from numpy.linalg import inv
inv(matrix)
```

The test fails when:

1. an inverse is introduced in a new production module;
2. an additional inverse is added to a grandfathered module;
3. a call is moved without updating the reviewable inventory; or
4. an alias is used to bypass the literal `np.linalg.inv` spelling.

When an existing inverse is replaced by a solve or factorized operator, the same
change must lower the expected count. The ratchet therefore makes the permitted
surface monotonically reducible while preventing silent numerical-policy drift.

## Migration guidance

For a system

\[
A x = b,
\]

compute `x = solve(A, b)` rather than `inverse(A) @ b`. For a covariance-weighted
quadratic form

\[
r^\mathsf{T} A^{-1} r,
\]

whiten `r` with a Cholesky factor and take the squared Euclidean norm. For a
low-rank covariance

\[
D + U U^\mathsf{T},
\]

use the existing Woodbury-based structured covariance operator rather than
materializing the dense matrix.

Rigid camera transforms are a separate migration class. A validated rigid
transform can be inverted analytically with a transposed rotation and translated
origin, but frozen calibration paths must not be changed merely to satisfy the
ratchet. Such migrations require parity tests over the complete supported input
contract.

## Scientific boundary

The ratchet changes no estimator, covariance, experiment, provider contract,
frozen artifact, target-access boundary, or scientific claim. It is a
forward-looking software-quality constraint. Existing inverse calls remain
explicitly visible until a separately reviewed, parity-tested migration removes
them.
