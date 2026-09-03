# Deform360 held-out-query Bayesian-value study v7

## Outcome

The file-triggered workflow completed successfully on the clean Deform360 copy
at `/mnt/seagate10tb/florianpfaff/datasets/deform360` using the
`gpuserver4090` runner label. It reproduced the exact frozen v3 point result for
all 92 registered objects and evaluated 64 deterministic low-frequency tactile
field queries, split before execution into 32 source-calibration and 32 held-out
evaluation queries.

No new measurement, camera image, geometry, point cloud, or unbound numeric
payload was opened. The target episodes were already opened by the parent study,
so this is retrospective mechanism evidence rather than fresh confirmation.

## Strict same-mean, same-marginal comparison

The full, diagonal, and scrambled arms had the same predictive mean by
construction. The maximum discrepancy between their raw coordinate marginal
variances was `2.220446049250313e-16`, and all three used exactly the same
source-derived variance scale and interval radius.

| Arm | 90% coverage | Query NLL | Event Brier | Decision loss |
|---|---:|---:|---:|---:|
| Full low-rank covariance | 72.240% | 0.388121 | 0.170694 | 0.103176 |
| Diagonal, marginal matched | 65.430% | 3.41928 | 0.174884 | 0.108675 |
| Scrambled, marginal matched | 64.811% | 4.07148 | 0.175571 | 0.109410 |

Negative paired differences favor the full covariance:

| Comparator | Metric | Full minus comparator | 95% object bootstrap |
|---|---|---:|---:|
| Diagonal | Decision loss | -0.00549957 | [-0.00708425, -0.00402809] |
| Diagonal | Event Brier | -0.00419010 | [-0.00498345, -0.00340905] |
| Scrambled | Decision loss | -0.00623370 | [-0.00785782, -0.00475196] |
| Scrambled | Event Brier | -0.00487761 | [-0.00571810, -0.00404054] |

Thus the registered strict matched-marginal dependence test is positive.

## Robustness to source-only recalibration

The conclusion changes when each covariance arm receives its own scalar
calibration from source episodes and the 32 calibration queries.

| Arm | 90% coverage | Query NLL | Event Brier | Decision loss |
|---|---:|---:|---:|---:|
| Full low-rank covariance | 72.240% | 0.388121 | 0.170694 | 0.103176 |
| Diagonal, independently calibrated | 73.012% | -0.082167 | 0.172733 | 0.100965 |
| Scrambled, independently calibrated | 73.117% | -0.059987 | 0.173893 | 0.101363 |
| Local diagonal only | 73.082% | -0.123295 | 0.172684 | 0.100748 |
| Diagonal, source-width matched | 73.901% | -0.204592 | 0.173848 | 0.100922 |

Compared with the independently calibrated diagonal arm, the full covariance
has **higher** decision loss by `0.00221073`, with 95% object-bootstrap interval
`[0.000648163, 0.00372869]`, and higher query NLL and event log loss. Its Brier
score is lower by `0.00203979`, but that comparison narrowly fails the registered
strict interval gate because the upper endpoint is `7.76104e-07`.

Compared with the scrambled, local-diagonal, and source-width-matched controls,
the full covariance retains a statistically lower Brier score, but has
statistically higher decision loss, query NLL, and event log loss. Therefore:

- independent-calibration robustness: **not supported**;
- source-width robustness: **not supported**;
- isolated low-rank-component decision value: **not supported**.

## Absolute decision and calibration diagnostics

The fallback cost was fixed at `0.1`, so an always-fallback policy has decision
loss exactly `0.1`. Every evaluated arm has loss above this baseline. The best
arm is local diagonal only at `0.10074777`; the full arm is `0.10317581`.
Consequently, the study does not show positive absolute decision utility over
exact fallback.

The source-defined adverse-event rate averages `10.12%`, while the corresponding
target rate averages `31.90%`. This action/episode shift is consistent with the
large calibration failure: the best observed nominal-90% target coverage is
only `73.90%`, and the full covariance reaches `72.24%` with query nANEES
`7.278` rather than the ideal value near one.

The score pattern is informative. Full dependence slightly improves Brier score,
but loses under logarithmic score and Gaussian query NLL after fair source-only
recalibration. This indicates occasional high-confidence errors or misspecified
tails: the full covariance contains useful ranking information, but is not a
well-calibrated predictive distribution on this transfer.

## Scientific interpretation

The experiment supports only the narrow mechanism statement:

> Holding the predictive mean, every coordinate marginal variance, and the
> common source calibration fixed, the frozen low-rank dependence structure
> improves held-out physical-query probabilities relative to dependence-destroyed
> controls.

It falsifies the stronger practical statement:

> The current full Bayesian covariance improves decisions over a properly
> source-calibrated diagonal uncertainty model or over exact fallback.

This result should therefore not be used as the sole headline evidence that the
Bayesian formulation has practical decision value. It is appropriate as a
controlled dependence ablation and as motivation for better calibration or for
a direct multivariate proper-score study. It does not authorize a paper,
calibration, unseen-object, deployment, or robot-safety claim.

## Reproducibility binding

- Workflow run: `33751475658`
- Trigger commit: `7be19066e521627199394234e0b70784d8b4489e`
- Artifact: `9892001282`
- Artifact digest: `sha256:e8beaa3b6058b3f5a560ae4288535726949123d2e20fe2569072116a60e6b0f0`
- Canonical result digest: `6e2eaedd9c6977fe3f129cae38f8ef4f63f1ea26e3eb11ebfc86215de7106418`
- Result-file SHA-256: `34f07cadad138e242d13fd4349bcf4b2ff1804f0e680fe3f5e3b571f259d6e2c`
- Protocol SHA-256: `4b1fa8a5a305ede21ebaca10be2a42da0cf6c46e835dc26581677e3bda046e8f`

The GitHub artifact contains the complete result JSON, object-level CSV,
protocol, request, report, console log, and checksums. The artifact retention
expiration is 2026-12-02; the compact evidence record in this directory preserves
the registered aggregate results and binding after that date.
