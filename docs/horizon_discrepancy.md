# Source-calibrated horizon discrepancy

`bayesian_phystwin.horizon_discrepancy` is an additive uncertainty model for the
model-averaged endpoint posterior. It does not change the historical endpoint
predictor or any frozen provider-v1 behavior.

## Motivation

The endpoint posterior describes discrepancy at the causal-prefix boundary. A
future prediction needs a separate assumption about persistence. Keeping the
endpoint mean fixed and only adding random-walk variance is often too strong:
physical discrepancy can relax toward the simulator baseline, while unresolved
model error can approach a stationary floor and continue accumulating process
uncertainty.

The new contract models these effects as

\[
\rho(h)=\rho_{\min}+(1-\rho_{\min})2^{-h/\tau},
\]

where `tau` is a source-selected half-life and `rho_min` is the asymptotic mean
retention. Per-axis additional variance is

\[
(1-\rho(h)^2)\sigma_{\mathrm{stationary}}^2
+ h\sigma_{\mathrm{process}}^2.
\]

The existing within-component process variance and the between-component
model-average covariance are retained in the total-covariance calculation.

## Information order

Calibration consumes one endpoint/future summary per independent source object
or acquisition session. Frames, cameras, tracks, points, and taxels do not create
additional source groups. Candidate persistence policies are selected by equal
weight per source group.

A calibration artifact fails closed when it declares use of interval-calibration,
confirmation, or target outcomes. This keeps dynamics selection separate from
subsequent conformal interval calibration and prospective confirmation.

## Example

```python
from bayesian_phystwin.horizon_discrepancy import (
    fit_horizon_discrepancy_calibration,
    predict_horizon_conditioned_endpoint,
)

calibration = fit_horizon_discrepancy_calibration(
    source_group_ids,
    endpoint_mean_m,
    future_mean_m,
    horizon_steps=(4, 8, 16, 32),
)

prediction = predict_horizon_conditioned_endpoint(
    endpoint_posterior,
    calibration,
    horizon_steps=16,
)
```

At horizon zero, the propagated mean and covariance reproduce the supplied
`ModelAveragedEndpointPosteriorV1`. For positive horizons, the result records the
content-addressed calibration ID, mean retention, component moments, and explicit
per-axis added variance.

## Claim boundary

This module provides model and provenance infrastructure. Source fit alone does
not establish prospective coverage, physical transfer, deployment safety, or
state-of-the-art performance. Those claims require a frozen interval calibration
and an unopened confirmation cohort.
