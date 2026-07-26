# PhysTwin MVTracker Prefix Competence v1

## Purpose

The remaining PhysTwin headroom is dominated by the gap between a useful
manual sparse-prefix identity channel and weak automatic camera observations.
This control asks whether a released joint multiview 3D tracker can close
enough of that observation gap to justify simulator assimilation.

The test is deliberately upstream of Bayesian inference. MVTracker receives
the three released calibrated RGB-D views and nine benchmark query locations
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
- Identities: all nine benchmark manual-track queries, initialized only at
  frame 90.
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
