# Fresh Deform360 Bias Guard: Post-Open V1

## Status

This is a post-open stress test on the exhausted 12-object fresh-pairwise
cohort. It applies the existing source-v4 configuration and prospective lock
without changing a threshold, feature, state model, or source certificate. It
is not prospective confirmation, calibration, selector tuning, or SOTA
evidence.

## Question

The frozen pairwise update regressed against persistence and its selected raw
backbone on every object. The earlier selective-camera diagnostic showed that
source v4 safely abstained when the sealed physical response was exactly zero.
This cohort asks the harder question:

> Does source v4 remain safe when PhysTwin predicts a nonzero response but the
> real object remains nearly static?

The diagnostic uses:

- the sealed selected raw trajectory as the unchanged baseline;
- the sealed physical-minus-persistence rollout as causal response support;
- the original source-v4 bias model and dynamic-evidence thresholds;
- the original source-v4 certified group-regret lock;
- residual-independent view redundancy and reprojection reliability;
- the source-v4 variance floor because no cycle-covariance sidecar was sealed.

Candidate construction accepts no target. Already-open targets are joined only
for this diagnostic score.

## Result

| Quantity | Result |
| --- | ---: |
| Cases / physical objects | 12 / 12 |
| Total update intervals | 36 |
| Candidate-available intervals | 4 |
| Accepted intervals | 4 |
| Exact-fallback intervals | 32 |
| Harmful accepted intervals | 4 |

Object-balanced means:

| Arm | Hidden identity RMSE | Hidden Chamfer |
| --- | ---: | ---: |
| Selected raw baseline | 0.899 mm | 0.772 mm |
| Guarded source v4 | 0.945 mm | 0.836 mm |
| Relative change | +5.07% | +8.29% |

There are zero object-level wins, nine exact ties, and three regressions for
both metrics. The object-bootstrap difference intervals are `[0.000, 0.101]`
mm for identity RMSE and `[0.000, 0.143]` mm for Chamfer.

The four accepted intervals occur in three objects. Their physical-response
RMS is `0.780–4.332` mm, observed-motion RMS is `3.960–16.486` mm, and the
frozen causal-agreement score is `0.501–1.000`. Every inferred correction is
harmful despite passing all source-v4 eligibility checks.

## Interpretation

Source v4 solved the zero-response case but not physical-model misspecification.
When the simulator predicts motion that the object does not execute, coherent
triangulation bias can align with that simulated response and pass:

1. nonzero physical response;
2. nonzero observed motion;
3. physical-response agreement;
4. state/bias identifiability; and
5. the source-wide group-regret certificate.

The source regret bound changed sign under transfer. It was calibrated from
four eligible source objects and cannot support a 90% claim; this diagnostic
now provides concrete negative transfer evidence rather than only an
arithmetic caveat.

## Decision

Do not deploy source v4 unchanged on another fresh camera-only cohort. Preserve
its useful exact-fallback and bias-decomposition machinery, but replace its
admission evidence.

The next candidate should require gripper-excluded evidence upstream of 3-D
triangulation:

1. per-camera 2-D object motion above a source-calibrated tracking floor;
2. cross-view agreement of projected motion direction and temporal onset;
3. agreement with the projected physical response, not only triangulated 3-D
   displacement;
4. a feature-conditional upper regret bound with enough independent source
   groups, otherwise exact fallback;
5. an independent depth, tactile, or registered material-point anchor for any
   common global component;
6. sealed predictive covariance if calibration is claimed.

Absent independent evidence, the safe camera-only policy on these windows is
exact persistence.

## Provenance

- Unchanged source-v4 lock SHA-256:
  `5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6`
- Diagnostic implementation commit:
  `6bc448d3eec9d1a04e3f763890d4f4e2b2607f54`
- Result SHA-256:
  `6dceeb23dc4a766bb1e2fd84acf175b8bc822dcd20afe7dc122671ce54ea2a3f`
- Evidence:
  `results/sota/deform360_fresh_bias_guard_postopen_v1/result.json`
