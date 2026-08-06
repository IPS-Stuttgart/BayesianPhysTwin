# Prospective directional endpoint v2

## Status and scope

`bayesian_phystwin.phystwin_directional_endpoint_v2` is a prospective numerical
implementation. It does not replace or modify the frozen
`phystwin_directional_endpoint.py` implementation used by historical PhysTwin,
Deform360, MatPhys, or paper evidence. Existing protocols and artifact identities
remain byte-for-byte reproducible. The module is not imported by the root package
or command registry, so adoption requires an explicit versioned import.

The v2 path addresses one specific numerical and uncertainty issue in future
experiments: the frozen implementation moment-matches the robust inlier/outlier
mixture and then, for a full three-dimensional source update, replaces the
resulting covariance by

```text
trace(P) / 3 * I.
```

That replacement preserves average variance but is not a positive-semidefinite
upper bound. Whenever `lambda_max(P) > trace(P) / 3`, it reduces uncertainty in
the dominant direction. Robust mixture uncertainty is commonly anisotropic
because the between-component mean term lies along the innovation direction.

## V2 update

For each inlier or outlier component, v2 forms

```text
S_j = H P H^T + R_j
K_j = P H^T S_j^-1
m_j = m + K_j (y - H m)
P_j = (I - K_j H) P (I - K_j H)^T + K_j R_j K_j^T.
```

`S_j` is factored by the shared `SPDSystem` backend. Cholesky solves provide the
Kalman gain, quadratic form, and log determinant without constructing an
explicit inverse. The Joseph covariance expression is used for each component.
The two Gaussian components are then moment-matched exactly:

```text
m = p m_in + (1 - p) m_out
P = p [P_in + (m_in - m)(m_in - m)^T]
  + (1 - p) [P_out + (m_out - m)(m_out - m)^T].
```

The complete covariance is retained. The scalar `variance` output is
`lambda_max(P)`, so it remains a conservative directional scalar readout without
discarding the covariance needed by downstream calculations.

## Fail-closed numerical contract

Every prior, innovation, component-posterior, mixture, and final covariance must
be finite, symmetric within the declared tolerance, positive definite, and below
the configured condition-number limit. Triangular solves must satisfy the shared
residual contract.

V2 never uses:

- implicit diagonal jitter;
- eigenvalue clipping;
- a pseudoinverse;
- silent covariance repair; or
- trace-average isotropization.

A failed admission raises `DirectionalEndpointNumericalError`. A claim-bearing
caller must then retain its unchanged physical baseline or another explicitly
registered fallback; it must not rescue the same observation by tuning the
condition limit after outcome access.

## Usage

```python
from bayesian_phystwin.phystwin_directional_endpoint_v2 import (
    DirectionalEndpointConfigV2,
    DirectionalEndpointNumericalError,
    robust_directional_endpoint_v2,
)

try:
    posterior = robust_directional_endpoint_v2(
        source_residual,
        source_valid,
        multiview_residual,
        multiview_valid,
        tangent_projectors,
        priority_identities,
        end_frame=prefix_stop,
        process_variance=1e-6,
        observation_variance=1e-4,
        initial_variance=1e-3,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
        config=DirectionalEndpointConfigV2(
            maximum_condition_number=1e12,
        ),
    )
except DirectionalEndpointNumericalError:
    posterior = None  # caller applies its registered exact fallback
```

Returned arrays are immutable. `posterior.diagnostics()` records the schema and
backend versions, maximum admitted condition numbers, Joseph/mixture semantics,
and the absence of hidden regularization.

## Promotion boundary

Unit tests establish numerical contracts, covariance retention,
orthogonal-coordinate invariance, fail-closed conditioning, and compatibility of
the well-conditioned one-step posterior mean. They do not establish improved
physical prediction, calibrated uncertainty, independent-object transfer,
deployment safety, Causal4D benefit, or state of the art.

Before v2 can enter a physical or claim-bearing protocol, the protocol must:

1. name the v2 implementation explicitly;
2. freeze the numerical configuration before target outcomes are opened;
3. retain the corresponding v1 or physical-baseline comparison;
4. report any numerical rejection as an exact fallback; and
5. create a new evidence identity rather than reinterpreting historical v1
   artifacts.
