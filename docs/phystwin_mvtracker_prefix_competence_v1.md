# PhysTwin MVTracker Prefix Competence v1

## Purpose

The remaining PhysTwin headroom is dominated by the gap between a useful
manual sparse-prefix identity channel and weak automatic camera observations.
This control asks whether a released joint multiview 3D tracker can close
enough of that observation gap to justify simulator assimilation.

The test is deliberately upstream of Bayesian inference. MVTracker receives
the three released calibrated RGB-D views and four benchmark query locations
at frame 90. It predicts those identities through frame 120. Its prediction is
sealed before the separately staged manual prefix target is opened.

## Why MVTracker

MVTracker operates jointly in calibrated multiview 3D and consumes sensor
depth directly. This tests a materially different observation model from the
failed independent AllTracker/CoTracker plus triangulation paths. No fixed
Prob4D blend or camera-only confidence accumulation is introduced.

## Frozen Test

- Case: `single_lift_cloth`, already-open source development only.
- Prediction frames: `[90, 121)`, wholly inside the released training prefix.
- Cameras: 0, 1, and 2.
- Identities: benchmark identities 3, 4, 6, and 8, the four identities finite
  at frame 90, initialized only at that frame.
- Tracker: MVTracker revision
  `ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072`.
- Checkpoint SHA-256:
  `a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6`.
- Comparator: exact persistence of the frame-90 query positions.

The tracker must retain at least 75% supported point-frames, improve identity
RMSE over persistence by at least 10%, remain below 15 mm overall RMSE, and
remain below 15 mm over the final six prefix frames.

## Boundaries

The prediction process cannot read the withheld manual prefix target, any
frame at or after 121, or any simulator future metric. A pass authorizes only
a separately preregistered assimilation smoke. A failure stops this tracker
route without parameter tuning. No held-v8 or sealed PokeFlex artifact is
involved.

The source staging step reads the public manual-track file once, then writes
two disjoint artifacts: a prediction-visible frame-90 query file and a
withheld `[90, 121)` evaluation file. Future rows are never retained.

## Prediction Lock

Source staging was executed from Bayesian-PhysTwin commit
`269f0e34ef9cb08b4aa85bb02fa1e2d1d32a0581`. Before any MVTracker prediction,
the prediction-visible query archive was locked at SHA-256
`1aafdb6074d6e53643e086a0ef4aea2caf28be8a2218fce9b315abf9fd621f11`
and the inaccessible prefix target at SHA-256
`a7657e40fd26811f0fff2b49e29b32b82fda251d616454f7dfc9c62620782488`.
The source report is bound by SHA-256
`998f51e9b31cda044f7f2bef5122b4751ca41fc5e7f537b1aa0be98c9e5424a7`.
The protocol status is now `locked-before-mvtracker-prediction`; the frozen
prediction may proceed, but withheld scoring remains forbidden until the
prediction archive has been sealed.
