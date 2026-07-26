# SpatialTrackerV2 persistent-identity competence control v1

Status: locked before inference and scoring.

## Motivation

The opened PhysTwin-22 headroom audit separates dense geometric correction
from sparse material identity. Released dense pseudo-tracks plus manual prefix
identities reach `7.891873/13.429357` mm CD/manual-track error, but that manual
arm is an online-supervised capacity ceiling, not a deployable method.
Automatic CoTracker3 reaches `10.627040/20.414981` mm and does not close the
identity gap.

Two alternative persistent 3D trackers have already failed source competence
controls:

- MVTracker supplied broad support but had `18.209` mm displacement RMSE; and
- TAPIP3D supplied `85.49%` support but had `10.168` mm displacement RMSE and
  improved only `1.11%` over strict CoTracker3 on shared support.

SpatialTrackerV2 is tested as a different observation model rather than another
confidence rule around the same triangulations. The official offline RGB-D
model uses depth and calibrated camera motion while tracking persistent custom
queries.

## Question

On the already-open `single_lift_cloth` source interaction, can official
SpatialTrackerV2 recover nine persistent material identities with both:

1. materially broader support than strict-three-view CoTracker3; and
2. materially lower displacement error on the identity-frame pairs where both
   methods provide observations?

This remains an association-oracle competence control. The model receives the
nine released manual identity coordinates only at frame zero. The later manual
trajectory is unavailable until the prediction has been sealed.

## Frozen model and input

The external checkout is pinned to
`7e12274c52077860cebfe007a6290777db43b63c`; it remains external under CC
BY-NC 4.0 and is not vendored. The official offline checkpoint revision is
`76e275b00f9c57dab71d46544df5255d4538106d`, with model SHA-256
`f1236958b274867ca9a743303eb2cf48a9d217a7d005e163b45a9ab87ed2e723`.

The prediction receives camera-0 RGB-D frames `[0, 121)`, known intrinsics and
extrinsics, and the same nine frame-zero queries used by the TAPIP3D control.
Inference is frozen to 756 internal tracks, four tracking iterations, replacement
ratio `0.2`, stage 1, offline mode, and visibility threshold `0.5`.

SpatialTrackerV2's official RGB-D path expresses reconstructed tracks in a
first-camera gauge. The wrapper follows the official transformation returned by
the model, then applies only the locked camera-0-to-PhysTwin-world transform.
No score-time translation, rigid, similarity, affine, or scale fit is allowed.

## Comparator and gates

The comparator is strict-three-view CoTracker3 with nearest-node association
fixed from frame-zero geometry and displacement re-anchored to the identical
manual query coordinate.

The shared-support comparison prevents either tracker from winning by reporting
a different subset. All six frozen gates must pass:

- prefix support at least 70%;
- at least 20% lower displacement RMSE than CoTracker3 on shared support;
- overall displacement RMSE at most 5 mm;
- frame-zero anchor RMSE at most 2 mm;
- late-third support at least 50%; and
- late-third displacement RMSE at most 10 mm.

Passing authorizes only an automatic graph-query source study. Failure stops
SpatialTrackerV2 as this sparse-identity feeder without tuning on the opened
trajectory.

## Artifact boundary

The external runner accepts only the frozen RGB-D input, checkpoint, and
inference constants. The `seal-prediction` parser exposes no case directory,
manual trajectory, or CoTracker3 cue. It verifies checkpoint and input hashes,
recomputes the frame-zero pixel projection, validates the world and pixel query
bindings, and writes a compact carrier without copied RGB-D tensors.

Only `score`, after verifying that seal, may open the released source
trajectory and CoTracker3 cues. Frames after 120 are not used.

No automatic retry is permitted. Any memory, resolution, track-count, camera,
or window change after a technical failure requires a committed amendment
before another forward pass.

## Claim boundary

This is one opened source case with oracle frame-zero association. It tests
prefix observation competence, not future forecasting. It cannot establish
transfer, calibration, confirmation, or state of the art, and it grants no
authority to inspect held-v8.
