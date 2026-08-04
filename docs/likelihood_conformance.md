# Student-t mixture semantics and conformance

Bayesian-PhysTwin exposes two deliberately different robust scores at the
uncertain-gauge boundary. They share the same normalized nominal/outlier
Student-t mixture kernel, but they represent different probabilistic objects and
must not be reported as the same likelihood. Both standalone operations are
NumPy-only and remain available in the base installation.

## Conditional reliability-weighted objective

`conditional_grouped_student_t_mixture_objective` and
`update_prior_aware_gauge_belief` use one latent nominal/outlier state per
declared correlation group. The standalone operation evaluates the solver's
objective at a supplied conditional prediction; that prediction must already
include the evaluated physical-state and nuisance contributions.

For active row `i`, let

```text
z_i = sqrt(s_i) C_i^(-1/2) r_i,
```

where `s_i` is residual-independent prior reliability, `C_i` is conditional
local covariance, and `r_i` is the conditional residual. For group `g`, stack
the active rows into `z_g`, define `q_g = z_g^T z_g`, and let `d_g` be the
stacked dimension. A row with zero prior reliability is excluded from both
`q_g` and `d_g`; it cannot change the responsibility, score, group power, or
association diagnostic of supported rows.

The group mixture kernel is

```text
rho_g t_nu(z_g; 0, ((nu-2)/nu) I)
+ (1-rho_g) t_nu(z_g; 0, ((nu-2)/nu) lambda_out I).
```

The covariance-to-scale factor `(nu-2)/nu` is required because the declared
matrices are component covariances rather than raw Student-t scale matrices.
The posterior nominal responsibility is

```text
alpha_g = rho_g t_nominal
          / (rho_g t_nominal + (1-rho_g) t_outlier),
```

and the exact score precision relative to the reliability-scaled local
precision is

```text
omega_g = alpha_g (nu+d_g) / ((nu-2) + q_g)
        + (1-alpha_g) (nu+d_g)
          / ((nu-2) lambda_out + q_g).
```

Every supported row in one correlation group receives the same `omega_g`.
Only `alpha_g` depends on the residual; the supplied `rho_g` does not.
Association probability remains diagnostic information.

For consumer-owned repeated-evidence capping, the generalized-Bayes group power
is

```text
w_g = composite_g min(cap, n_active) / n_active.
```

For a provider-final artifact, `composite_g` already contains the final per-row
effective-sample power and no second cap is applied. The solver's normal-system
row weight is `w_g s_i omega_g`. The standalone conditional objective uses the
same mixture kernel, active-row definition, and group-power rules.

`PriorAwareGaugeConfigV1.minimum_robust_precision` is zero by default. A
positive value explicitly floors `omega_g` and changes the solver's score in the
far tails. The standalone conditional operation always reports the exact,
unfloored negative log objective; it is intended for conformance, evaluation,
and audit rather than for reproducing an explicitly floored approximation.

Machine-readable semantics:

```text
conditional-reliability-weighted-student-t-objective-v1
```

## Covariance-marginalized diagnostic

`grouped_student_t_mixture_likelihood` preserves the original portable
observation score. It constructs, within each correlation group, a covariance
from block-diagonal local covariance plus the declared shared low-rank factors
and evaluates that covariance with Cholesky and Woodbury identities.

This operation intentionally does **not** use `prior_reliability`. It answers a
different question: how surprising is the residual under a covariance-matched
marginal diagnostic in which the shared factors have been folded into the
covariance? It does not evaluate the conditional generalized-Bayes objective
optimized by the prior-aware solver, where those factors are explicit nuisance
variables with their own priors.

Both operations now call the same normalized Student-t mixture kernel for
component densities and posterior nominal responsibility. Their difference is
therefore explicit in the residual representation and information power, rather
than arising from duplicated mixture formulas.

Machine-readable semantics:

```text
covariance-marginalized-student-t-score-v1
```

The historical function name remains available for compatibility. New analyses
must record the returned `semantics` property and must not use its score as
evidence that prior reliability or explicit nuisance marginalization was
applied.

## Strict minimax solver

`update_gauge_aware_belief` retains the conservative rowwise Student-t IRLS
objective used with strict nuisance-subspace projection. In that mode,
`prior_nominal_probability` is a residual-independent information-power weight;
it is not a latent mixture prior. The result diagnostics state this explicitly,
and this solver is not used as evidence for either grouped nominal/outlier
mixture score above.

Use `update_prior_aware_gauge_belief` when the grouped mixture and independently
calibrated nuisance priors are part of the declared method. Use the strict
solver when nuisance priors are not trusted and minimax projection is required.
Use the covariance-marginalized diagnostic only when its folded-covariance
interpretation is the intended audit quantity.

## Posterior curvature and covariance

The prior-aware solver reports the minimum and maximum eigenvalues of the exact
reduced mixture Hessian at the final fixed point. Its returned
`posterior_covariance` remains the working Gauss-Newton/IRLS covariance: the
mixture score precisions are held fixed while the normal matrix is inverted.
That matrix is generally not the exact mixture Hessian because the posterior
responsibilities and Student-t precisions vary with the residual.

The working covariance is formed from a symmetric Cholesky factor and
triangular solves against the identity; the implementation does not call
`np.linalg.inv(normal)`. Prospective coverage evaluation remains required
before treating this covariance as calibrated uncertainty.

## Executable checks

`tests/test_prior_aware_likelihood_conformance.py` verifies:

- normalized density, posterior responsibility, score precision, and score
  gradient in one and three dimensions against an independent formula;
- the solver MAP against an explicit dense-grid posterior;
- posterior nominal responsibility and one shared precision for a multi-row
  group;
- exact local curvature against a finite-difference posterior curvature;
- reliability-weighted Mahalanobis distance and complete inertness of
  zero-reliability rows;
- the working covariance against the final IRLS normal matrix; and
- symmetry and positive definiteness of the Cholesky-derived covariance.

`tests/test_grouped_likelihood.py` additionally verifies:

- the covariance-marginalized diagnostic against an explicit dense covariance;
- both standalone operations against the shared mixture kernel;
- explicit machine-readable score semantics;
- zero-reliability inertness for the conditional objective;
- consumer-owned and provider-final group-power behavior;
- legacy and canonical Prob4D repository identities; and
- fail-closed handling of unknown composite-weight semantics.
