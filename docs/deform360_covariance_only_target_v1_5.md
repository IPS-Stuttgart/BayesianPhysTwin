# Deform360 covariance-only target v1.5

## Why this amendment exists

Independent source-only review rejected the v1.4 gate before target decode. The
heteroscedastic provider correctly consumes full metric covariance and cue-only
reliability, but one favorable fixture was incorrectly generalized into a global
confidence-monotonicity claim.

V1.5 preserves the v1.4 implementation and scientific method while correcting the
claim. The v1.2 roster, acquisition and quarantine artifacts, registered mean,
covariance donor, horizon scales, support thresholds, fallback, estimands, and
no-replacement rule remain unchanged. No target media, endpoint, or outcome was
opened under v1.4.

## Conditional covariance semantics

For endpoint component `k`, the provider uses

```text
R_tn,k = (R_tn + sigma_obs,k^2 I) / reliability_tn
```

inside one robust inlier/outlier mixture update. Changing `R_tn` or reliability
also changes the robust responsibilities and the evidence weights among the 15
default endpoint components. Consequently, two separately conditioned posterior
covariances do not have a general Loewner ordering. Larger input covariance or
lower reliability can change the posterior in either matrix direction while each
reported covariance remains PSD.

A deterministic source-only counterexample with NumPy seed 260811 is frozen in
the test suite. It prevents future text from inferring a global ordering guarantee
from one favorable fixture. This behavior is a property of conditional robust
model averaging, not evidence that metric uncertainty is ignored.

## Guarantees retained

V1.5 retains and tests these contracts:

- full `3x3` metric row covariance changes the production-default forecast;
- residual-independent cue reliability changes the production-default forecast;
- assignment-mixture spread survives into the scored covariance;
- every candidate covariance is PSD and the registered mean is byte-identical;
- geometry controls association, not prior perception reliability;
- distant rows cannot manufacture empirical updates;
- duplicate correlated windows do not reduce covariance or raise reliability;
- admitted innovations are unclipped and robustified once; and
- unsupported identities and failed cases return exact fallback.

The original frozen v5 materializer is restored byte-for-byte. The new provider
reuses its existing association routine without changing a source file bound by
older protocol hashes.

## Claim boundary

This remains a custom fresh-object Deform360 calibration study. At the pinned Hub
revision there are no genuinely fresh official processed annotations after prior
exclusions, so this cohort cannot establish official Deform360 point SOTA. V1.5
authorizes no target processing until its source commit, regenerated dry run,
protocol lock, complete tests, and independent source-only review all pass.
