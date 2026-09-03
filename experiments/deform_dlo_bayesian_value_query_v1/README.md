# DEFORM DLO Bayesian-value nonlinear-query diagnostic v1

This retrospective diagnostic asks a narrowly isolated Bayesian question on the
already opened DLO4/DLO5 evaluation trajectories:

> When the coordinate prediction is held fixed, does propagating its frozen
> Bayesian coordinate belief through a nonlinear deformation query improve the
> query-space point decision or predictive score?

No model is retrained or refitted. The workflow consumes the immutable
`target_predictions.npz`, prediction seals, evaluation manifests, and public
trajectory files from successful parent workflow `33361441865`. Every retained
file and every trajectory is checked against its frozen size and SHA-256 before
scoring.

## Arms

- **Plug-in:** apply the query directly to the candidate coordinate mean,
  `Q(E[Y])`.
- **Full Bayesian:** transform draws from the frozen source-calibrated 3-D
  coordinate belief and use `E[Q(Y)]`, `median(Q(Y))`, and the full scalar query
  distribution.
- **Diagonal marginal-matched:** zero only the off-diagonal coordinate
  covariances. Point means and all coordinate marginal variances remain exactly
  unchanged.

## Queries

At the terminal frame and across the final quarter of the forecast, the
workflow evaluates free-node distance and squared distance from the line through
the contemporaneous left and right clamp centers. Scores are first averaged
within each complete trajectory. DLO4 and DLO5 then receive equal weight.

The squared-loss contrast compares the plug-in query with the posterior mean of
the query. The absolute-loss contrast compares it with the posterior median.
Scalar CRPS compares the full query distribution with both the marginal-matched
diagonal distribution and the plug-in point mass.

## Execution

The path-triggered workflow runs the data job on:

```yaml
runs-on: [self-hosted, Linux, X64, gpuserver4090]
```

The exact parent identities, query definitions, sample count, random seeds,
bootstrap, and interpretation boundary are frozen in `protocol.json`.

This is intentionally a retrospective diagnostic, not fresh target evidence.
A positive result would establish value only for these nonlinear queries under
the available per-point 3x3 coordinate belief. It cannot establish cross-node
or cross-time dependence, physical-state identification, unseen-object
transfer, universal calibration, or deployment safety.
