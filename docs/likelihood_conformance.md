# Student-t mixture semantics and conformance

Bayesian-PhysTwin has two robust objectives at the uncertain-gauge boundary.
They are intentionally distinct and must not be described as the same
likelihood.

## Prior-aware grouped nominal/outlier mixture

`grouped_student_t_mixture_likelihood` and
`update_prior_aware_gauge_belief` use one latent nominal/outlier state per
declared correlation group. For active row `i`, let

```text
z_i = sqrt(s_i) C_i^(-1/2) r_i,
```

where `s_i` is the residual-independent prior reliability and `C_i` is the
conditional local covariance. For group `g`, stack the active rows into `z_g`,
define `q_g = z_g^T z_g`, and let `d_g` be the stacked dimension. A row with
zero prior reliability is excluded from both `q_g` and `d_g`; it cannot change
the responsibility of supported rows.

The group likelihood is

```text
rho_g t_nu(z_g; 0, ((nu-2)/nu) I)
+ (1-rho_g) t_nu(z_g; 0, ((nu-2)/nu) lambda_out I).
```

Equivalently, before whitening, its nominal and outlier component covariances
are the reliability-scaled block covariances and `lambda_out` times those
covariances. The covariance-to-scale factor `(nu-2)/nu` is required because the
input matrices are component covariances rather than raw Student-t scale
matrices.

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

The group composite weight and effective-sample cap define one common
residual-independent generalized-Bayes power

```text
w_g = composite_g min(cap, n_active) / n_active.
```

The solver therefore minimizes `-w_g log p_g` for each group. Its normal-system
row weight is exactly `w_g s_i omega_g`. This separation keeps reliability,
mixture state, and repeated-evidence capping mathematically identifiable.

`PriorAwareGaugeConfigV1.minimum_robust_precision` is zero by default. A
positive value explicitly floors `omega_g` and therefore changes the exact
mixture score in the far tails. Solver diagnostics distinguish the exact and
precision-floored objectives.

## Strict minimax solver

`update_gauge_aware_belief` retains the conservative rowwise Student-t IRLS
objective used with strict nuisance-subspace projection. In that mode,
`prior_nominal_probability` is a residual-independent information-power weight;
it is not a latent mixture prior. The result diagnostics state this explicitly,
and this solver is not used as evidence for the grouped nominal/outlier-mixture
claim.

Use `update_prior_aware_gauge_belief` when the grouped mixture and independently
calibrated nuisance priors are part of the declared method. Use the strict
solver when nuisance priors are not trusted and minimax projection is required.

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
