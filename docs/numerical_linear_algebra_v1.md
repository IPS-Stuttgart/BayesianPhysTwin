# Numerical linear-algebra policy v1

## Purpose

Prospective BayesianPhysTwin inference often solves covariance or information
systems whose numerical admission is part of the scientific boundary. A direct
matrix inverse hides whether the system was positive definite, rank deficient,
ill conditioned, or silently regularized. The explicit module
`bayesian_phystwin.numerical_linear_algebra_v1` provides one NumPy-only policy
for new symmetric systems without rewriting evidence-bound historical source.

The module exposes two operations:

- `solve_spd` uses a Cholesky factorization for a symmetric positive-definite
  system; and
- `solve_psd` uses an eigendecomposition for a consistent symmetric
  positive-semidefinite system and returns its minimum-norm solution.

Both operations return an immutable solution, an optional covariance, and
`SymmetricSolveDiagnostics`. The diagnostics record the solver, dimension,
numerical rank, two-norm condition number, extreme eigenvalues, relative
residual, and declared regularization.

## Fail-closed semantics

The public functions reject nonnumeric or non-finite arrays, nonsquare or
asymmetric matrices, mismatched right-hand sides, nonliteral Boolean options,
and malformed numerical thresholds. They apply no hidden jitter, clipping, or
ridge term. The reported regularization is therefore exactly zero.

`solve_spd` rejects a non-positive-definite matrix through Cholesky and rejects a
condition number above the caller's declared maximum. A caller may set
`maximum_condition_number=None` only when another frozen boundary owns an
equivalent condition gate; the condition number is still computed and reported.

`solve_psd` requires an explicit relative rank tolerance. It rejects negative
eigenvalues beyond that tolerance, proves that the right-hand side has no
material component in the admitted nullspace, and solves only the retained
range. Its optional covariance is the admitted Moore-Penrose covariance under
the same eigensystem and rank decision.

Neither operation chooses whether a Bayesian candidate is scientifically
admissible. The caller still owns identifiability, calibration, plausibility,
regret, and exact-fallback decisions.

## Example

```python
import numpy as np

from bayesian_phystwin.numerical_linear_algebra_v1 import solve_spd

information = np.array([[4.0, 1.0], [1.0, 3.0]])
score = np.array([1.0, 2.0])
result = solve_spd(
    information,
    score,
    compute_covariance=True,
    maximum_condition_number=1e12,
)

posterior_increment = result.solution
posterior_covariance = result.covariance
assert result.diagnostics.method == "cholesky"
assert result.diagnostics.regularization == 0.0
```

## Incremental inverse ratchet

The changed-source preflight reports `BPTQ003` when a changed package module
introduces `numpy.linalg.inv`, including common import aliases. New numerical
code should use an explicit solve, Cholesky factorization, or admitted
eigendecomposition and retain the relevant diagnostics.

The ratchet is forward-only. Existing historical modules are not rewritten merely
to satisfy the policy. A direct inverse in an exact frozen reproduction may be
suppressed next to the full expression with a specific explanation, for example:

```python
# bpt-quality: allow BPTQ003 -- frozen tagged reproduction
```

A suppression does not authorize the same pattern in a prospective estimator or
claim-bearing provider boundary.

## Evidence boundary

A successful solve establishes only the declared numerical properties of that
system. It does not establish observation quality, posterior calibration,
identifiability, unseen-object transfer, physical-query benefit, Causal4D
intervention benefit, deployment safety, or state of the art.
