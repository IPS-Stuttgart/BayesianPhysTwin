# Deform360 projected-view action-response source v3

## Purpose

V2 established that the balanced frame-zero planner exposes physically moving
material identities in all three camera panels, but dynamic 3-D triangulation
remains fragile and the surviving panels disagree. V3 asks a narrower,
target-free question:

> Does causal per-camera image motion contain the non-rigid response predicted
> by the sealed physical rollout after a camera-specific translation nuisance
> is removed?

This is an admission smoke only. It does not construct a state update, inspect
a hidden identity, or score a future outcome.

## Frozen observation path

The opt-in planner is `projected-view-response-v3`; the default remains the
legacy V1 path.

V3 retains V2's target-free balanced selection of 16 physically responsive
graph identities and its eight-camera plan. For each selected camera and each
causal update at frames `19`, `38`, and `57`, it:

1. projects the sealed physical trajectory into that camera;
2. runs AllTracker forward using only RGB frames `0` through the update;
3. runs AllTracker over the same prefix in reverse to measure cycle error;
4. converts physical and observed pixel displacement to metric camera-tangent
   displacement using frame-zero depth and focal length;
5. removes a per-camera shared translation nuisance across material points;
6. evaluates the remaining shape response with the unchanged
   `ActionResponseAdmissionConfig`.

Forward and reverse visibility determine whether a row exists. The
forward/backward cycle error determines association probability. The tracker
runtime does not expose per-point continuous confidence through this frozen
interface, so prior reliability is the residual-independent binary source
acceptance indicator. The residual against PhysTwin is processed exactly once
inside the robust action-response certificate and never feeds prior
reliability.

Metric covariance uses a frozen `2 px` standard deviation, inflated by `4` for
model mismatch, and is propagated in square metres using the frame-zero
depth-to-focal scale. Cameras remain separate evidence groups; the gate
requires the frozen passing fraction across at least three views. No
triangulated 3-D observation is used for admission.

## Provenance and information boundary

The physical certificate binds:

- the sealed 3-D physical rollout array;
- the projected camera-tangent physical response;
- sampled frame indices and selected camera identities;
- intrinsic and extrinsic calibration file hashes.

The observation certificate binds positions, validity, metric covariance,
prior reliability, association probabilities, and cycle errors.

Only the already-open source case `059-shoe-ep0000` is permitted. The runner
may read RGB frames no later than frame `57`, frame-zero mask/depth support,
calibration, and the known action. It may not read a future object point cloud,
future material identity, target metric, held-v8 artifact, or sealed V1 target.

## Decision boundary

- Rejection preserves the exact physical baseline and closes this fixed
  projected-view certificate on the examined source case.
- Admission authorizes only construction of a candidate update on an
  already-open, multi-object source panel.
- Advancement beyond source development additionally requires a frozen
  baseline-relative regret upper bound, bit-exact fallback, and transfer over
  disjoint hidden identities.

Neither outcome alone supports an accuracy, calibration, or state-of-the-art
claim.
