# Deform360 covariance-only independent validation v1

## Scientific objective

This protocol asks one deliberately narrow confirmatory question:

> Does the Bayesian endpoint covariance improve probabilistic forecasts on
> object-disjoint Deform360 sessions when the predictive mean is the exact same
> `last_residual` object?

The development experiment on the already-open 22-session PhysTwin cohort
localized a Gaussian-NLL benefit to covariance while keeping every point
prediction unchanged. That result selected one candidate for this study; it is
not itself the confirmatory evidence.

## Frozen candidate

The candidate is fixed before any confirmation payload or future outcome is
opened:

| Component | Frozen value |
| --- | --- |
| Mean | exact caller-owned `last_residual` `float64` array |
| Covariance donor | `independent_endpoint_v1` |
| Early / middle / late scales | `[8, 16, 16]` |
| Observation standard deviation | `5 mm` |
| Covariance eigenvalue floor | `1e-12 m²` |
| Unsupported or rejected case | exact `last_residual` reference/fallback |
| Point change | forbidden |

The donor, scales, horizon bins, observation model, target roster, and endpoint
cannot be selected again after target access.

## Independent cohort

The study reuses the exact public Stage-0 Deform360 selection at
`protocols/locks/deform360_official_hub_visuotactile_v1_selection.json`.

- The ten previously opened calibration object-sessions remain source-only.
- The twelve separately selected confirmation object-sessions are the target.
- The target contains six sheet and six volumetric objects.
- Source and target are disjoint by physical-object identity.
- Every failed, unsupported, or rejected target remains in the denominator as
  the preregistered exact fallback; there is no replacement.
- No new robot acquisition is required.

This is separate from the immutable sixteen-object v6 challenger study. It does
not reinterpret or rescue that protocol or the earlier official-Hub
source-support result.

## Prediction barrier and information order

Before any target future frame is opened:

1. the protected source execution must retain a complete target-closed receipt,
   ten physical manifests, and 100 source-prediction seals;
2. the exact evaluation distribution, runtime, source artifacts, and twelve-unit
   target roster must be content-addressed;
3. prefix-only predictions for all twelve target units must be sealed atomically;
4. every prediction must prove in memory that the hybrid returns the exact same
   mean object and must retain matching serialized mean content; and
5. the complete prediction batch must state that future frames, target outcomes,
   human selection, and replacement were unused.

The target prefix is frames `[0, 58)`, the scored future is `[58, 76)`, and
`[76, 81)` remains an unscored buffer. The twelve future payloads may then be
opened exactly once.

## Arms

The complete report retains four arms:

1. unchanged physical fallback;
2. deterministic `last_residual`, the primary reference;
3. the preregistered prefix-routing/exact-fallback control; and
4. the same exact mean with the frozen Bayesian covariance.

Only arm 4 versus arm 2 is the confirmatory covariance-value contrast. The other
arms preserve the physical and routing context without widening the primary
hypothesis family.

## Primary endpoint and decision

For each physical object-session, Gaussian negative log predictive density is
computed using the common 5 mm observation model. Early, middle, and late scores
are averaged equally within the object-session, and the twelve complete
object-sessions receive equal weight.

The primary effect is:

```text
frozen covariance hybrid minus zero-covariance last_residual
```

Lower is better. The primary claim is authorized only when the upper endpoint of
a two-sided 95% paired object-session bootstrap interval is below zero. The
bootstrap uses 100,000 replicates and seed `20260812`. Frames, tracks, vertices,
coordinates, or cameras never increase the independent sample size.

Point noninferiority is not estimated: point predictions must be content
identical by construction, so track and Chamfer differences are exactly zero.

## Predeclared secondary analyses

The following analyses are fixed before target access and cannot select a new
candidate:

- simultaneous max-t 95% intervals for early, middle, and late horizons;
- simultaneous sensitivity at observation standard deviations
  `2.5`, `5`, and `10 mm`, with `5 mm` remaining primary;
- marginal 50%, 90%, and 95% coverage, full interval width, and width ratio;
- sheet and volumetric effects, all twelve object effects, and the worst object;
- an exact paired sign test; and
- exact point-artifact identity and fallback accounting.

A calibration-and-sharpness statement additionally requires 90% marginal
coverage in `[0.80, 0.98]` and a mean width ratio no larger than `4.0`. A
robustness statement requires all three observation-noise sensitivity intervals
to favor the hybrid. A horizon-wide statement requires all three horizon
intervals to favor it. Failure of these qualifications does not rewrite the
primary result.

## Claim boundary

A positive result supports only the following statement:

> On the twelve locked, object-disjoint Deform360 confirmation sessions, the
> frozen Bayesian endpoint covariance improves object-session-level
> probabilistic forecasts over zero-covariance `last_residual`, while point
> predictions remain exactly unchanged.

The protocol does not identify a latent physical-state correction, authorize
deployment, establish performance outside the locked cohort, demonstrate a
Causal4D intervention benefit, establish benchmark parity, or support a
state-of-the-art claim. A negative or inconclusive result is complete and cannot
be rescued by target-side retuning.
