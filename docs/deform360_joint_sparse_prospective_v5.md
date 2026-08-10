# Deform360 joint-sparse prospective benefit v5

## Purpose

Version 5 is the proposed confirmatory experiment after the frozen Deform360
v1-v3 support negatives and the development-only joint-sparse observability v4
protocol.

The earlier experiments asked whether every camera stream was independently
sufficient. That is the wrong unit for a multi-view Bayesian model. A camera can
be only partially informative while the object-level collection of cameras,
causal windows, metric anchors, contact factors, and gauge priors is jointly
identifiable.

Version 5 therefore asks the claim-bearing question directly:

> On the twelve still-sealed Deform360 confirmation objects, does a frozen,
> guarded, joint-sparse visuotactile Prob4D update improve future physical-query
> prediction over both the unchanged physical fallback and the last causal
> residual, without harmful accepted updates?

The machine-readable design lock is
`protocols/locks/deform360_official_hub_joint_sparse_prospective_v5.json`.

## Why this is a genuinely new experiment

Version 5 does not delete the eleven unsupported v1 cameras, weaken a completed
gate, refit v1-v3, or reinterpret their results. It changes the scientific unit
and factorization before any confirmation payload is opened:

- the primary admission unit is one physical object and one registered physical
  query, not one camera;
- every registered camera remains in the provenance and denominator;
- an unavailable partial factor contributes zero likelihood rather than causing
  camera or object deletion;
- complementary partial factors are accumulated in one tree-sparse explicit
  gauge model;
- gauge, shared visual bias, view bias, and contact-anchor bias are marginalized
  before query observability and deployment risk are assessed; and
- every rejected or unsupported candidate returns the unchanged physical
  fallback exactly.

This is a new protocol version. The completed v1-v3 evidence remains immutable.

## Cohorts and information order

The existing Stage-0 selection remains fixed:

- ten development objects: five sheet and five volumetric;
- twelve confirmation objects: six sheet and six volumetric; and
- one previously selected episode per physical object.

The ten development objects have already been opened and may be used only for
v4 structural development and the fully frozen v5 source-fit procedure. They
cannot contribute to the final effect estimate. The twelve confirmation
payloads remain closed until all of the following exist:

1. a valid v4 result with status `development-design-supported`;
2. the committed v5 policy and exact execution revisions;
3. a content-addressed source calibration fit;
4. leave-one-physical-object-out source evaluation meeting the v5 source gate;
5. source-frozen guard and interval artifacts; and
6. the existing claim-bearing confirmation-opening authorization.

Source and confirmation must use identical implementation, provider, model,
factor, guard, endpoint, and analysis revisions. Confirmation-side adaptation is
forbidden.

## Factor model

For partial visual factor `i`, the registered linearized design is

```text
H_i = [H_state,i, H_gauge,i, H_shared_bias,i, H_view_bias,i].
```

Contact factors use

```text
A_i = [A_state,i, A_anchor_bias,i]
```

and contain no camera-gauge term.

The causal gauge tree is retained in precision form. Conditional observation
covariance whitens each factor. Dependence groups receive a fixed effective
sample cap, so duplicated pixels, frames, taxels, or tracks cannot manufacture
precision. After accumulating all admitted partial factors for an object, gauge
and bias nuisances are marginalized with a Schur complement in the registered
physical-query subspace.

A valid object-level update requires the unchanged v4 structural conditions:

- at least two cameras;
- at least two causal windows;
- at least eight spatial clusters;
- full registered-query rank;
- weakest query precision at least `1e-9`;
- query condition number at most `1e10`;
- no camera contributing more than `85%` of query precision;
- at least `75%` rank retention after removing any one camera; and
- at least `75%` rank retention after removing any one causal window.

Unavailable partial factors are retained with zero likelihood. Technical
failures are retained and deploy exact physical fallback. Nothing is replaced.

## Causal cutoff and independent endpoint

The registered action window is `[0, 81)`. The update may use only the causal
prefix `[0, 58)`. The next 18 frames, `[58, 76)`, are the future evaluation
interval. Frames `[76, 81)` are an unscored terminal buffer. A source fit,
deployment guard, or acceptance decision that uses any frame at or after 58 is
invalid.

Exactly two endpoint camera views are reserved for every physical object. They
are selected by ascending SHA-256 rank of
`object_id\0camera_id\0v5-endpoint-view-v1`, using camera identities and released
camera-calibration metadata only. The rule is applied before any confirmation
pixel or outcome is opened and is identical for development and confirmation.
Reserved endpoint views remain in the complete provenance but contribute no
likelihood term to any method.

The primary target is released future geometry from the two reserved views,
processed by the frozen official Deform360 revision
`d8522a4403b766aeb387510c04e89032a56fdf35`. Its definition may not reuse a
contact or tactile refinement factor. Future-frame and reserved-view losses are
averaged within object before equal-object aggregation. This prevents the same
camera evidence from both changing and judging the posterior.

## Source calibration

After the v5 design and exact execution software are frozen, the ten development
objects are processed with a nested leave-one-object-out procedure.

For each held-out development object, the other nine fit:

- parallel and lateral point-variance factors;
- Sim(3) scale, rotation, and translation factors;
- shared and view-specific visual-bias priors;
- contact-anchor bias; and
- the tie-preserving operational guard threshold.

Each physical object receives equal weight regardless of camera, frame, point,
track, or taxel count. The source report must pass on at least eight of ten
objects and at least four of five objects in each stratum. Only then is one final
source artifact fitted on all ten development objects for the locked
confirmation run. This final refit changes no hyperparameter or decision rule.

The guard risk score has the fixed semantics
`lower-is-safer-inclusive-threshold-v1`. Threshold-native risk coverage accepts
complete tied score blocks; object IDs may not be used to split a tie.

Intervals use source-only grouped split conformal calibration. No confirmation
prefix, outcome, future frame, or target loss may tune a variance factor, bias
prior, guard, threshold, or interval.

## Compared methods

All seven methods are evaluated for every confirmation object:

| ID | Method | Role |
| --- | --- | --- |
| `B0_physical_fallback` | Unchanged physical prediction | Baseline and exact fallback |
| `B1_last_causal_residual` | Last causal readout residual | Registered simple reference |
| `V1_joint_sparse_visual_guarded` | Joint-sparse explicit-gauge visual update | Visual mechanism comparator |
| `T1_contact_anchor_only` | Independently calibrated contact update | Contact mechanism comparator |
| `VT1_joint_sparse_visuotactile_guarded` | Joint-sparse visual plus contact update with source-frozen guard | **Primary candidate** |
| `VT2_joint_sparse_visuotactile_unguarded` | Same update without the guard | Safety diagnostic |
| `VT3_joint_sparse_visuotactile_anchor_bias` | Guarded update with explicit shared anchor-bias nuisance | Bias diagnostic |

Causal4D is deliberately not part of the primary experiment. It is evaluated
only after a positive BayesianPhysTwin v5 result, so intervention logic cannot
hide an observation or state-estimation failure.

## Statistical unit and endpoints

The physical object is the sole independent statistical unit. Cameras, causal
windows, views, frames, points, tracks, and taxels are averaged within object and
cannot increase the sample size.

The primary endpoint is future held-out-view geometry error whose target is
independent of tactile refinement. Within-object registered horizons and the two
reserved views are averaged first; the twelve objects are then equally weighted.

Secondary endpoints are:

- future control-point track error;
- operational acceptance coverage and exact-fallback frequency;
- harmful accepted-update frequency;
- predictive coverage and full interval width;
- nonlinear closure error;
- 90th-, 95th-, and worst-object regression; and
- separate sheet and volumetric summaries.

Every method is present for every object. A rejected candidate records its raw
result, but its deployed loss is the byte-exact physical fallback. Rejected
objects are never removed from the denominator.

## Positive decision

`VT1_joint_sparse_visuotactile_guarded` is positive only if **all** frozen checks
pass:

1. object-balanced primary loss improves by at least `10%` versus
   `B0_physical_fallback`;
2. it improves by at least `5%` versus `B1_last_causal_residual`;
3. paired equal-object bootstrap 95% upper bounds for absolute loss difference
   are below zero against both comparators;
4. at least `10/12` objects improve against each comparator;
5. at least `5/6` objects improve in each stratum against each comparator;
6. at least `10/12` objects, including at least `5/6` per stratum, are accepted
   by the source-frozen guard;
7. no accepted object exceeds the physical fallback by more than the frozen
   `2%` harmful-update margin;
8. neither stratum's mean regresses by more than `2%`;
9. the contact-augmented primary method improves at least `2%` over the otherwise
   identical visual-only guarded method; and
10. every rejected or unsupported object reproduces the physical fallback
    exactly.

The paired bootstrap uses 10,000 group replicates, seed `20260810`, and 95%
confidence. Requiring at least 10 improvements among 12 objects also gives an
exact one-sided sign probability of approximately `0.0193` under a 50/50 null;
the bootstrap remains the registered effect-size interval.

Zero observed harmful accepted objects is a cohort decision rule, not a
population safety certificate: twelve confirmation objects are too few to
certify a small general harm probability at 95% confidence. The claim boundary
therefore explicitly excludes deployment safety.

A valid negative result is complete evidence. No target-side threshold change,
camera deletion, object replacement, provider substitution, or second opening is
allowed.

## Negative controls

The frozen report includes:

- metric-anchor ablation;
- contact-factor ablation;
- camera-identity permutation;
- dependence-group duplication;
- the unguarded update; and
- an explicit assertion that future frames were unavailable to source fitting
  and deployment decisions.

Controls diagnose the mechanism but cannot rescue a failed primary decision.

## GitHub execution plan

This design PR adds only target-closed contracts. It cannot schedule a
self-hosted scientific job and cannot open data.

After v4 has a valid development result, a separate reviewed execution PR must:

1. bind the exact v4 result and policy identities;
2. freeze exact BayesianPhysTwin, Prob4D, MotionCrafter, and model revisions;
3. materialize and cross-validate the source calibration artifact;
4. produce confirmation-opening authorization;
5. run exactly once on protected `main` and `workstation2`;
6. upload compact, checksummed source and confirmation evidence before
   enforcement; and
7. publish either a positive or negative object-level result.

## Claim boundary

A positive result supports only prospective object-level improvement of the
frozen guarded joint-sparse visuotactile BayesianPhysTwin method over the
unchanged physical fallback and last causal residual on the twelve locked
Deform360 confirmation objects.

It does not establish general unseen-object performance beyond this cohort,
deployment calibration, Causal4D intervention benefit, safety, official
benchmark parity, or state of the art.
