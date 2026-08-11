# PhysTwin release-facing claim contract v1

## Purpose

This document fixes the scientific wording that should accompany a
BayesianPhysTwin software release. It consolidates existing frozen evidence; it
is not a new experiment and does not alter any historical artifact, method, or
metric.

The canonical paper-side synthesis is maintained in
[BayesianPhysTwin-Paper](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/bpt_release_claim_synthesis_2026-08-10.md).
The owning point-result source remains the
[full-22 evidence report](phystwin_sota_22_v1.md), and the owning retrospective
covariance source is the
[full-22 covariance-only report](full22_covariance_only_hybrid_v1.md).

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

A release must therefore not describe Bayesian anchoring as the unique best
deterministic predictor. The differentiating contribution is a bounded,
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

## Retrospective covariance-only mechanism result

A registered full-22 analysis asked whether a Bayesian covariance improves the
predictive distribution while preserving the exact caller-owned
`last_residual` mean. Its cross-fitted selected/scaled arm reported:

| Diagnostic | Result |
| --- | ---: |
| Overall Gaussian-NLL difference | `-9.136` |
| Simultaneous 95% CI | `[-13.961, -4.312]` |
| Object-session wins | `17/22` |
| Exact point-mean identity | `22/22` |
| Marginal 90% coverage | `0.706 -> 0.910` |
| Mean full interval width | `0.01645 m -> 0.05094 m` |
| Width ratio | `3.10×` |

The point trajectory, Chamfer distance, and track error were unchanged by
construction. The result identifies useful retrospective uncertainty with a
material width cost. It is not evidence that raw covariance is calibrated, that
point prediction improved, or that the candidate transfers to unseen objects.

The only candidate frozen for a separate fresh study is:

- exact caller-owned `last_residual` mean;
- covariance donor `independent_endpoint_v1`;
- early/middle/late scales `[8.0, 16.0, 16.0]`;
- common observation standard deviation `0.005 m`;
- covariance eigenvalue floor `1e-12 m²`; and
- exact registered comparator/physical fallback on rejection or unsupported
  evidence.

No donor, scale, endpoint, observation-noise, action-family, denominator, or
fallback change is permitted after fresh target access.

## Independent-validation boundary

The controlled Prob4D-to-BayesianPhysTwin mechanism is positive on its synthetic
calibration/target split, but independent real-provider transfer is not yet
established.

### Retrospective MotionCrafter transfer

Across 19 already-open interactions:

- physical fallback RMSE is `6.899 mm`;
- marginal-gauge deployment RMSE is `6.942 mm` (`+0.62%`);
- `11/19` marginal-gauge updates are accepted with `37.3%` nominal-90%
  coverage; and
- the explicit-persistent guard accepts `0/19` updates and exactly reproduces
  fallback.

Those interactions may not be reused to tune a replacement confirmation method.

### Terminal complete-stream official-Hub provider version

The earlier frozen official-Hub route completed ten-object source preparation
and all `324/324` admitted visual-production jobs. Its next complete-stream
robot/camera support gate retained `11` support-negative streams:

- supported streams: `313/324`;
- technical failures: `0`;
- source covariance fit: not run;
- leave-one-object-out source gate: not run; and
- twelve-object confirmation access: not authorized.

That provider version is terminal at its source-support boundary. This is not a
fitted-covariance failure and does not establish independent-object transfer.
Deleting cameras, fitting only the supported streams, changing the fixed prefix,
or opening its confirmation panel would violate the frozen information order.

The exact paper-side evidence is in the
[Deform360 source-support result](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/deform360_prob4d_source_support_negative_2026-08-09.md).

### Separate registered Deform360 v6 confirmation route

A separate existing-data route freezes the exact ten opened development/source
object-sessions and twelve disjoint confirmation object-sessions from the
content-addressed selection artifact. It does not reinterpret or rescue the
terminal complete-stream provider version above.

The v6 route requires, in order:

1. exactly 100 sealed prefix-only source prediction records;
2. a target-blind source decision under the frozen candidate and source panel;
3. authorization, only after a source-positive decision, for one preregistered
   evaluation over all twelve confirmation object-sessions; and
4. a target prediction barrier before scoring.

The paper-side Gaussian-NLL analysis, simultaneous contrasts, 100,000
case-bootstrap procedure, exact `2^12` sign-flip enumeration, point-parity rule,
stratum non-regression checks, worst-object reporting, and positive/negative
wording were preregistered before target access. The twelve confirmation
object-sessions remain closed. A source-negative result is complete evidence and
keeps them closed; no target result may revise the candidate or analysis.

## Software versus scientific evidence

The package may continue to improve archive integrity, numerical admission,
structured covariance handling, testing, packaging, and cross-repository
contracts. Such changes are engineering evidence only unless a registered run
binds the exact revision and fresh outcomes.

In particular, green CI, exact fallback tests, bounded I/O, stricter solver
convergence checks, a source-only residual-history adapter, or a new provider
adapter do not by themselves change the release claim above.

## Required release wording

Every release note that cites the full-22 improvement should also state:

- last residual is the principal matched comparator and is marginally better on
  equal-case track error;
- raw posterior covariance is severely undercalibrated;
- the retrospective covariance-only result preserves exact point predictions
  and improves proper score only with a `3.10×` mean-width cost;
- conformal results are width-bearing and assumption-specific; and
- independent real-provider and independent-object transfer remain unconfirmed,
  with the registered twelve-object confirmation panel still closed.

The current evidence does not authorize claims of a unique deterministic winner,
calibrated raw posterior covariance, dynamically identified simulator-state
correction, independent-object transfer, deployment safety, or overall state of
the art.
