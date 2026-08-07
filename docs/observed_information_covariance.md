# Exact observed-information covariance

## Purpose

The prior-aware grouped Student-t mixture solver returns a working
Gauss--Newton/IRLS covariance. That covariance is useful for recursion and
conditioning diagnostics, but it treats the final group precisions as locally
fixed. The solver also computes the exact local mixture Hessian to diagnose its
eigenvalues, but historically did not expose the corresponding covariance.

`bayesian_phystwin.observed_information_covariance` adds a prospective,
versioned post-processor. It leaves the accepted point estimate and every frozen
protocol unchanged. For an already inference-admissible result, it:

1. reconstructs the exact reduced state/nuisance design;
2. recomputes every group responsibility, expected precision, and precision
   derivative at the selected solution;
3. reproduces the existing working posterior covariance exactly;
4. forms the exact observed local mixture information;
5. fails closed unless that information is finite, positive definite, and below
   the configured condition-number limit; and
6. maps its Cholesky-solve covariance back to the complete state and nuisance
   domain while retaining the original prior covariance in unidentifiable state
   directions.

## Curvature

For group `g`, let

```text
q_g(beta) = sum_i reliability_i ||r_i(beta)||^2
```

and let `p_g(q_g)` denote the exact expected precision of the normalized
nominal/outlier Student-t mixture. The working information contains

```text
power_g p_g D_g' W_g D_g.
```

The exact observed information additionally contains

```text
2 power_g p'_g s_g s_g',
```

where

```text
s_g = sum_i reliability_i D_i' r_i.
```

The derivative is generally negative, so exact mixture curvature can be smaller
than the working Gauss--Newton curvature. A non-positive or ill-conditioned
observed-information matrix is therefore a real local diagnostic and causes the
post-processor to reject the covariance rather than add jitter, clip
eigenvalues, or silently fall back to the working covariance.

## Usage

```python
from bayesian_phystwin.observed_information_covariance import (
    observed_information_covariance_from_prior_aware_result,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    update_prior_aware_gauge_belief,
)

result = update_prior_aware_gauge_belief(batch, config=config)
observed = observed_information_covariance_from_prior_aware_result(
    batch,
    result,
    config=config,
    metadata={"protocol": "prospective-query-uncertainty-v1"},
)

print(observed.covariance_semantics.method)
# laplace_observed_information
print(observed.covariance_semantics.calibrated)
# False
```

The same `PriorAwareGaugeConfigV1` instance used for inference must be supplied.
The post-processor verifies the final robust weights, working covariance,
mixture-Hessian eigenvalue diagnostics, state/nuisance dimensions, and input
lineage before returning a result.

## Artifact contents

`ObservedInformationCovarianceResultV1` retains immutable, content-addressed
records for:

- the working reduced information matrix;
- the exact observed reduced information matrix;
- its reduced covariance;
- the mapped complete covariance;
- the state prior, declared prior eigenvalue floor, and retained state mapping;
- ordered observation and anchor group identities;
- group likelihood powers;
- expected group precisions and their exact derivatives;
- the observed-information condition number;
- explicit `laplace_observed_information` covariance semantics; and
- caller metadata plus the original result lineage.

The artifact ID hashes array dtypes, shapes, and bytes rather than relying on
filenames or mutable NumPy objects.

## Admission boundary

The operation requires:

- an inference-admissible `GaugeAwareBeliefResult`;
- the exact batch that produced it;
- `minimum_robust_precision == 0`, because a positive precision floor changes
  the objective into a separately defined approximation;
- exact reproduction of ordinary and anchor robust weights;
- exact reproduction of the solver's working covariance; and
- a positive-definite exact observed Hessian;
- PSD state-prior and complete covariance matrices under the declared prior eigenvalue floor; and
- a nonempty retained-state mapping that is orthonormal in the prior-standardized positive-eigenvalue basis and has negligible prior-nullspace leakage.

It does not rescue an inadmissible update and does not affect the separate
baseline-relative regret guard or exact-fallback decision.

## Scientific boundary

This covariance is an uncalibrated local Laplace/observed-information
approximation. It is not a frequentist coverage guarantee, a group-robust
sandwich covariance, or a deployment certificate. Claim-bearing physical-query
intervals still require independent object/session calibration, retained
technical failures, complete interval-width reporting, and an untouched target
cohort. The existing group-sandwich and conformal artifacts remain separate
uncertainty layers.
