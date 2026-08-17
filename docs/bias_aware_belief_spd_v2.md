# Prospective SPD bias-aware belief v2

## Purpose

`bayesian_phystwin.bias_aware_belief_v2` is a prospective numerical successor
to the frozen bias-aware implementation in `bias_aware_belief.py`. It preserves
the same linear state/shared-bias/per-camera-bias model, source reliability,
Student-t IRLS weighting, ambiguity rejection, and exact physical fallback on
well-formed inputs. It changes the numerical contract deliberately.

Historical Deform360 protocols and committed results remain bound to v1. They
must not be relabelled as v2 evidence, and the v1 source file is not modified by
this implementation.

## Numerical contract

`bayesian_phystwin.spd_system.SPDSystem` validates each claim-bearing normal
system before it is solved:

1. require a finite nonempty square matrix;
2. measure the maximum asymmetry against explicit absolute and relative
   tolerances;
3. deterministically replace an admitted matrix by `(A + A.T) / 2`;
4. perform one Cholesky factorization of that admitted system;
5. reject a non-finite or excessive 2-norm condition number;
6. solve vectors or matrices through the retained lower-triangular factor and
   its transpose;
7. reject a solve whose normalized residual exceeds the frozen tolerance; and
8. reconstruct an explicit inverse only when the exported posterior-covariance
   contract requires it.

The backend never adds jitter, clips eigenvalues, substitutes a pseudoinverse,
or silently retries with a different precision.

The same factor object also provides whitening, quadratic forms, and a
Cholesky-derived log determinant for future unfrozen consumers.

## Versioned API

```python
from bayesian_phystwin.bias_aware_belief_v2 import (
    BiasAwareStateUpdateConfigV2,
    update_bias_aware_state_v2,
)

result = update_bias_aware_state_v2(
    camera_innovation_m,
    camera_available,
    state_basis,
    shared_bias_basis,
    prior_reliability=source_only_reliability,
    state_prior_covariance_m2=state_prior_covariance,
    config=BiasAwareStateUpdateConfigV2(
        maximum_condition_number=1e12,
        symmetry_absolute_tolerance=1e-12,
        symmetry_relative_tolerance=1e-10,
        solve_residual_tolerance=1e-10,
        inverse_residual_tolerance=1e-9,
    ),
)
```

A v1 `BiasAwareStateUpdateConfig` is intentionally not accepted by the v2
entrypoint. This prevents an experiment from switching numerical implementations
without an explicit configuration and source change.

Every v2 result records:

- implementation schema, version, and implementation ID;
- SPD backend schema and version;
- final condition number and solve residual;
- inverse-reconstruction residual when a covariance is exported;
- whether the exported covariance passed a positive-definiteness check; and
- explicit `false` values for implicit jitter, eigenvalue clipping, and
  pseudoinverse fallback.

## Failure and fallback semantics

Malformed shapes and invalid nonnumerical inputs remain programming errors.
Numerical admission failures after dimensions and fallback shapes are known
return an exact zero physical correction with a reason code:

| Reason | Meaning |
| --- | --- |
| `invalid-state-prior-covariance` | Supplied state prior was non-finite, asymmetric, non-SPD, too ill-conditioned, or failed inverse reconstruction. |
| `ill-conditioned-posterior` | An IRLS or final posterior system exceeded the frozen condition limit. |
| `non-positive-definite-posterior` | An IRLS or final normal system failed SPD admission. |
| `unstable-posterior-solve` | A triangular solve failed or exceeded its residual tolerance. |
| `unstable-posterior-covariance` | The explicit exported covariance failed reconstruction, residual, or positive-definiteness validation. |

Existing semantic fallbacks such as `no-observation-support`,
`unanchored-common-mode-ambiguity`, and `implausible-state-update` are retained.
No failed numerical update changes physical-state, shared-bias, or camera-bias
coefficients.

## Evidence and compatibility boundary

The accompanying tests establish numerical and compatibility properties only:

- exact Git-blob identity of the frozen v1 source;
- agreement with v1 on a well-conditioned non-claim-bearing fixture;
- solve, whitening, log-determinant, and inverse residuals;
- no silent repair of material asymmetry, singularity, or indefiniteness;
- row and coordinate permutation invariance;
- singular state-prior fallback; and
- fail-closed handling of a deliberately invalid final IRLS system.

These tests do not establish improved physical accuracy, calibrated uncertainty,
provider competence, Deform360 confirmation, Causal4D benefit, deployment safety,
or state of the art.

A future experiment may use v2 only after a new protocol or amendment freezes:

- the exact v2 source revision and distribution;
- all four numerical tolerances and the condition limit;
- the source/calibration/target information order;
- reliability, covariance, nuisance, guard, and fallback artifacts; and
- a separate result identity and evidence ledger entry.

V1 and v2 result rows must remain separately labelled even when their values are
numerically equal on a compatibility fixture.
