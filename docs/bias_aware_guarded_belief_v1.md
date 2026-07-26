# Bias-Aware Guarded Belief v1

## Motivation

The prospective Deform360 selective-virtual-sensing study established a hard
limit for camera-only updates. A coherent triangulation can represent either
real object motion or shared camera/reconstruction bias. These worlds are
observationally identical without an additional assumption or independent
measurement. The sealed update therefore regressed by more than an order of
magnitude while camera-internal quality statistics remained plausible.

This milestone implements the next method candidate without reopening any
prospective outcome:

> infer only physically supported state modes, model coherent observation bias
> explicitly, and apply the candidate only when a source-calibrated upper bound
> predicts improvement over the unchanged baseline.

## Update Model

For camera or camera-derived observations `v`, the innovation is

```text
y_v = S a + B b + C_v d_v + e_v.
```

- `S` is an action-conditioned state basis built from the causal simulated
  physical response and declared contact/action support.
- `a` is the physical state correction.
- `B b` is low-rank spatial bias shared across views.
- `C_v d_v` is a per-camera offset.
- an independent depth, LiDAR, tactile, or registered actuator anchor observes
  `a` without the camera-bias terms.

The implementation is intentionally linear at this boundary. It exposes the
state/bias subspace cosine and posterior cross-correlation before a nonlinear
twin consumes the state correction. A global state translation and a global
camera bias without an independent anchor trigger exact abstention. A local
physical response orthogonal to the shared bias can still be estimated.

## Correlation And Reliability

Prior reliability may use source confidence, visibility, association entropy,
mask distance, multiview geometry, and sensor provenance. It does not use the
innovation against the current twin. The innovation enters once through a
Student-t IRLS likelihood.

Pixels within one view contribute at most a fixed effective sample count.
Views receive equal covariance-intersection weight when their cross-correlation
is unknown. Per-camera bias-prior precision is scaled by the same view count;
duplicating an identical camera therefore cannot increase state confidence
through repeated nuisance priors. Observation and anchor variances are metric
variances in square metres.

The update returns the exact zero correction when evidence is absent,
state/bias support is indistinguishable without an anchor, the posterior is
ill-conditioned, or the decoded state displacement exceeds its physical cap.
The caller then preserves the baseline trajectory and its uncertainty.

## Baseline-Relative Guard

The candidate's regret is defined as candidate loss minus unchanged-baseline
loss. A ridge model is fit only on source groups, with equal total weight per
group. Leave-one-group-out residuals set a one-sided finite-sample residual
quantile. A candidate is selected only when

```text
predicted regret + source residual quantile < -minimum improvement.
```

Feature vectors outside the axis-aligned source support return the baseline
bit-for-bit. This is a source-group exchangeability calibration device, not a
uniform guarantee under arbitrary domain shift. Candidate features must be
target-free and frozen before a prospective outcome is opened.

## Synthetic Controls

The deterministic benchmark uses seed `20260720`. Its result is stored at
`results/synthetic/bias_aware_guarded_belief_v1/summary.json`, SHA256
`7029b5a61c3992a12e0e0598b2d92638f4be827b4ba23f026f5f1c70ada4f58f`.

| Control | Result |
| --- | ---: |
| Unanchored common-mode exact fallback | 128/128 |
| Anchored state RMSE | 0.846 mm |
| Naive camera-as-state RMSE | 27.470 mm |
| Action-local state RMSE with shared bias | 0.393 mm |
| Four duplicated views / one-view variance | 1.0000000000000004 |
| Synthetic target regret-bound coverage | 99.22% |
| Supported target acceptance | 73.63% |
| Harmful accepted updates | 0 |
| Out-of-support exact fallback | yes |

All gates pass. These are implementation, identifiability, and positive-control
results only. They are not empirical accuracy, calibration, or state-of-the-art
evidence.

## Open-Source Development Result

The frozen v4 source candidate passes the transfer gates on 27 already-open
episodes from five objects. Against the selected raw backbone, the
object-balanced hidden identity RMSE changes from 8.807 to 8.683 mm (-1.414%)
and hidden Chamfer from 7.888 to 7.783 mm (-1.330%). Both object-cluster
intervals exclude zero. There are 7 episode wins, 20 exact ties, and no losses
on either metric.

This is not confirmation. The physical-agreement threshold was chosen after
source outcomes were open. Moreover, leave-one-object-out regret fits attain
only 75% finite-sample coverage and the full four-group source lock only 80%,
not the requested 90%. The exact result and claim boundary are recorded in
`docs/deform360_bias_aware_guarded_belief_source_v4.md`.

## Prospective Path

The next development stage may use only the already outcome-open Deform360
source objects. It must freeze:

1. a dynamic update-window rule requiring target-free evidence of contact,
   observed causal motion, and predicted physical response;
2. the physical-response rank and action-support construction;
3. the shared spatial-bias basis and metric prior scales;
4. the target-free eligibility rule and direct source-group regret bound;
5. exact fallback and source-support behavior.

The empirical ladder is:

| Arm | State support | Bias model | Regret guard |
| --- | --- | --- | --- |
| Baseline | no update | none | exact reference |
| Physical support only | causal physical response | none | no |
| Bias-aware state | causal physical response | shared + camera | no |
| Guarded bias-aware state | causal physical response | shared + camera | yes |
| Independent-anchor arm | same | same | yes |

Only after source choices are frozen may metadata select genuinely fresh
objects that do not overlap any opened or reserved Deform360 cohort. The
prediction artifacts must be sealed before targets are downloaded or opened.
Primary evaluation is object-clustered hidden-identity RMSE and hidden Chamfer
relative to the exact baseline, with coverage, interval width, acceptance,
harmful-update frequency, and bit-exact fallback reported separately.

## Claim Boundary

The previously opened Deform360 and PokeFlex results remain mechanism evidence
and cannot confirm this method. Prob4D remains a versioned observation feeder;
it does not own the state update or its safety claim. A fresh
accuracy/non-regression run is now justified because the guarded method beats
the exact baseline under object-held-out source cross-fitting with no accepted
harm. A 90% calibration or safety claim is not justified: the source lock has
only four eligible object groups and exact 80% finite-sample resolution. Until
a fresh outcome is opened, the strongest confirmed result remains the
camera-only impossibility/failure result plus controlled positive mechanisms.

## Later Fresh-Cohort Stress Test

The unchanged source-v4 lock was subsequently applied, without retuning, to
the already outcome-open 12-object fresh-pairwise cohort. It accepted four
intervals and all four were harmful, yielding +5.07% identity RMSE and +8.29%
Chamfer relative to the selected raw baseline. This is post-open stress
evidence rather than prospective confirmation, but it invalidates the earlier
recommendation to deploy source v4 unchanged on another camera-only cohort.

See `docs/deform360_fresh_bias_guard_postopen_v1.md`. The next candidate must
replace triangulated-3-D agreement with gripper-excluded per-camera motion
evidence or an independent modality, and must use a feature-conditional regret
bound with adequate source-group support.
