# PhysTwin TAPNext++ Prefix Competence v1

## Purpose

The strongest remaining source headroom is an automatic material-identity
observation that is accurate enough to support a guarded Bayesian state
update. This one-case control asks whether the public causal TAPNext++ tracker
can supply that observation. It stops before simulator assimilation.

TAPNext++ receives one manual query frame, the three released calibrated RGB-D
camera streams, and object masks over an allowed prefix. Its 3D output is
sealed before the separately staged manual trajectories over that prefix are
opened.

## Frozen Source Window

The case is the already-open `single_lift_cloth` source interaction. The
numeric interval is `[68, 88)`, wholly before the released training boundary
at frame 121. It was selected without manual trajectories: among all
20-frame intervals in the released training prefix, it maximizes first-to-last
RMS displacement of the selected PhysTwin nodes, at 62.565 mm.

The query identities remain benchmark identities 3, 4, 6, and 8. Cameras 0,
1, and 2 are used. The comparator is exact persistence of their frame-68
positions.

## Tracker And Metric Lift

The tracker is the official TAPNext++ PyTorch implementation at revision
`c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`, using the public 512-pixel
checkpoint with SHA-256
`6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f`.
Inference is recurrent and frame-by-frame. Every real query uses 64 local
support points within the official 32-model-pixel radius; support trajectories
are discarded.

Per-camera tracks are lifted by calibrated triangulation. A view supports a
candidate only when tracker visibility, object-mask support, reprojection, and
RGB-D depth agree. Two-view observations are permitted because one of the
three views can be occluded, but their geometric covariance is inflated
fourfold. A 5 mm shared metric bias floor is added once, and spread among
admissible cross-view assignments is included in covariance.

Unsupported observations fall back to the exact query position with zero
update support. Dense pixels and cameras are never accumulated as independent
confidence. The PhysTwin state residual is not an input to prior perception
reliability; a later assimilation method may process the state innovation
once through its robust likelihood.

## Frozen Gate

The route advances only if all four conditions hold:

- at least 75% of eligible point-frames are supported;
- identity RMSE improves over exact persistence by at least 10%;
- overall identity RMSE is at most 15 mm;
- RMSE over the final five prefix frames is at most 15 mm.

A pass authorizes only a separately locked, baseline-relative guarded
assimilation smoke. It does not authorize a larger cohort or any confirmatory
claim. A failure stops this route without tuning.

## Information And Claim Boundaries

Prediction cannot read the withheld manual prefix target, any observation
outside `[68, 88)`, any frame at or after 121, or a simulator future metric.
No held-v8 or sealed PokeFlex artifact is involved. The source staging process
loads the public manual-track and mask containers once, then writes disjoint
prediction-visible and withheld artifacts.

This is an already-open one-case competence test. It cannot establish a
Bayesian-PhysTwin gain, calibration, independent transfer, or state of the
art. Even a competence pass would not resolve coherent camera bias; that risk
belongs to the later physical/action-supported regret guard with exact whole-
belief fallback.

## Prelock

The method, source-window rule, public checkpoint hash, multiview thresholds,
uncertainty treatment, acceptance gates, and stopping rule are frozen before
source staging. The protocol remains
`prelocked-awaiting-source-staging` until prediction-visible and withheld
artifacts have been created, hashed, and bound by a separate commit.
