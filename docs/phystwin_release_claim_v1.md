# PhysTwin release-facing claim contract v1

## Purpose

This document fixes the scientific wording that should accompany a
BayesianPhysTwin software release. It consolidates existing frozen evidence; it
is not a new experiment and does not alter any historical artifact, method, or
metric.

The canonical paper-side synthesis is maintained in
[BayesianPhysTwin-Paper](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/bpt_release_claim_synthesis_2026-08-10.md).
Its machine-readable counterpart is
[`evidence/bayesian_phystwin/bpt-release-synthesis-v1/summary.json`](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/evidence/bayesian_phystwin/bpt-release-synthesis-v1/summary.json).
The owning point-result source remains the
[full-22 evidence report](phystwin_sota_22_v1.md).

## Authorized primary claim

> A bounded Bayesian anchor improves the re-evaluated released PhysTwin
> predictor on the official ordered 22-case contract.

The equal-case result is:

| Method | Chamfer distance | Track error |
| --- | ---: | ---: |
| Released PhysTwin | `0.011579 m` | `0.022019 m` |
| Bayesian anchor | **`0.010180 m`** | `0.019205 m` |
| Last residual | `0.010185 m` | **`0.019156 m`** |

Relative to released PhysTwin, the Bayesian anchor improves Chamfer distance by
`12.09%` and track error by `12.78%`. The last-residual method is the principal
matched deterministic comparator: it is approximately `0.005 mm` worse in
Chamfer distance and approximately `0.049 mm` better in track error.

A release must therefore not describe Bayesian anchoring as the unique
best deterministic predictor. The differentiating contribution is a bounded,
uncertainty-bearing readout/model-discrepancy belief with recursive, robust, and
guarded update interfaces.

## Uncertainty claim boundary

The point-estimate improvement does not calibrate the raw posterior covariance.
The archived audit reports:

| Diagnostic | Result |
| --- | ---: |
| Operational mean 3-D NEES | `1355.05` rather than approximately `3` |
| Nominal-90% ellipsoid coverage | `38.31%` |
| Posterior-scaled 90% CD coverage | `75.36%` |
| Posterior-scaled 90% manual-track coverage | `90.63%` |
| Additive conformal median case-mean upper bound, CD | approximately `38.87 mm` |
| Additive conformal median case-mean upper bound, track | approximately `42.68 mm` |

Release notes and API documentation must keep three layers separate:

1. posterior belief structure;
2. raw covariance diagnostics; and
3. conformal coverage together with width and the applicable exchangeability
   and score-shift assumptions.

The software release does not authorize a general claim of calibrated Bayesian
posterior uncertainty.

## Exact-mean covariance-only retrospective evidence

A separate retrospective intervention holds the complete `last_residual` point
mean exactly fixed and changes only the attached covariance. Donor identity and
one scale per early, middle, and late horizon were selected under outer
leave-one-physical-object/session-out folds. The primary cross-fitted result is:

| Quantity | Result |
| --- | ---: |
| Mean Gaussian NLL difference | **`-9.136`** |
| Simultaneous 95% interval | **`[-13.961, -4.312]`** |
| Better / worse / tied units | `17 / 5 / 0` |
| Exact mean identity | `22/22` units |
| Track-error difference | exactly `0 m` |
| Chamfer-distance difference | exactly `0 m` |
| Marginal 90% coverage | `70.6%` to `91.0%` |
| Mean full interval width | `16.45 mm` to `50.94 mm` |
| Width ratio | `3.10×` |

Because the point trajectory is identical, the relative Gaussian-score change
is attributable to the covariance under the frozen scoring model rather than a
point-prediction change. The coverage gain must always be reported with the
`3.10×` width cost.

The 22 outcomes were already open. Cross-fitting prevents each held unit from
selecting its own covariance donor or scale, but it does not create fresh
confirmation. The result is retrospective mechanism evidence only and does not
authorize independent calibration or deployment. It also does not authorize a
new point-accuracy claim because track and Chamfer outputs are exactly
unchanged.

For a separately registered fresh object/session study, the complete-source
candidate is frozen as the exact `last_residual` mean plus
`independent_endpoint_v1` covariance with early/middle/late scales
`[8, 16, 16]`. No target outcome may retune the donor, scales, scoring rule,
guard, fallback, or cohort boundaries.

The detailed paper-side record is the
[exact-mean covariance-only note](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/full22_covariance_only_hybrid_2026-08-11.md),
and the compact source is the
[`evidence/bayesian_phystwin/bpt-release-synthesis-v1/summary.json`](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/evidence/bayesian_phystwin/bpt-release-synthesis-v1/summary.json)
release synthesis.

## Independent-validation boundary

The controlled Prob4D-to-BayesianPhysTwin mechanism is positive on its synthetic
calibration/target split, but the current real-provider evidence is negative at
two distinct boundaries.

### Retrospective MotionCrafter transfer

Across 19 already-open interactions:

- physical fallback RMSE is `6.899 mm`;
- marginal-gauge deployment RMSE is `6.942 mm` (`+0.62%`);
- `11/19` marginal-gauge updates are accepted with `37.3%` nominal-90%
  coverage; and
- the explicit-persistent guard accepts `0/19` updates and exactly reproduces
  fallback.

Those interactions may not be reused to tune a replacement confirmation method.

### Official-Hub Deform360 route

The frozen official-Hub route completed ten-object source preparation and all
`324/324` admitted visual-production jobs. The next complete-stream robot/camera
support gate retained `11` support-negative streams:

- supported streams: `313/324`;
- technical failures: `0`;
- source covariance fit: not run;
- leave-one-object-out source gate: not run; and
- twelve-object confirmation access: not authorized.

The method version is terminal at that source-support boundary. This is not a
fitted-covariance failure and does not establish independent-object transfer.
Deleting cameras, fitting only the supported streams, changing the fixed prefix,
or opening confirmation would violate the frozen information order.

The exact paper-side evidence is in the
[Deform360 source-support result](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/deform360_prob4d_source_support_negative_2026-08-09.md).

## Software versus scientific evidence

The package may continue to improve archive integrity, numerical admission,
structured covariance handling, testing, packaging, and cross-repository
contracts. Such changes are engineering evidence only unless a registered run
binds the exact revision and fresh outcomes.

In particular, green CI, exact fallback tests, accepted or rejected golden-path
fixtures, bounded I/O, stricter solver convergence checks, valid wheel and source
distributions, or a new provider adapter do not by themselves change the
release claim above.

## Required release wording

Every release note that cites the full-22 improvement should also state:

- last residual is the principal matched comparator and is marginally better on
  equal-case track error;
- the exact-mean covariance-only result improves the frozen Gaussian score while
  point outputs remain exactly unchanged, carries a `3.10×` interval-width cost,
  and remains retrospective development evidence;
- raw posterior covariance is severely undercalibrated;
- conformal results are width-bearing and assumption-specific; and
- independent real-provider and independent-object transfer remain unconfirmed.

The current evidence does not authorize claims of a unique deterministic winner,
calibrated raw posterior covariance, fresh-cohort calibration from the
cross-fitted covariance-only result, dynamically identified simulator-state
correction, independent-object transfer, deployment safety, or overall state of
the art.
