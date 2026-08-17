# PokeFlex Conservative Shrinkage Target v2

## Pre-outcome amendment

Version 1 produced seven valid prediction seals and then stopped before any
target mesh was opened. The eighth registered take exposed a prediction-input
schema difference: its robot stream contains `T_WT`, force, and frame values but
does not contain `T_WE`. The selected action-local correction uses both poses to
define its action axis. Treating the two transforms as interchangeable would
silently change the source-selected method.

Version 2 therefore makes the existing unsupported-update rule explicit for
robot metadata. If any frame in the frozen action-field history lacks a finite
`T_WT`, `T_WE`, or force value, that prediction frame is unsupported and returns
the released checkpoint vertices exactly. No pose is imputed. This rule was
locked before any target deformed mesh, CD, or Jaccard outcome was opened.

The seven v1 seals remain archived as superseded technical evidence. They are
not mixed with v2. All eight registered takes must be rerun from scratch at one
clean v2 implementation revision before the all-case barrier can authorize
scoring. The cohort, selected arm, scale, observations, metrics, gates, and
claim boundary are unchanged.

## Custody and interpretation

Prediction at frame `f` uses only Kinect frames `f-5` through `f-1` and robot
history through `f-1`. No deformed target mesh is read during prediction.
Scoring remains unavailable until eight checksummed v2 prediction seals pass
the common-revision barrier.

A take with no usable `T_WE` receives exact baseline predictions rather than a
technical failure or an invented action axis. It still remains in the target
accounting, so this conservative amendment cannot remove a difficult object or
manufacture an improvement on it.
