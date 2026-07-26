# TAPIP3D persistent-identity competence control v1

Status: completed negative one-case source control. TAPIP3D passed the support
and anchoring gates but failed all three displacement-accuracy gates. The exact
feeder is stopped. See `docs/phystwin_tapip3d_competence_v1_result.md`.

## Motivation

The opened PhysTwin-22 headroom audit separated two observation roles:

- released dense pseudo-tracks provide most of the Chamfer-distance gain; and
- persistent sparse material identities provide most of the manual-track gain.

Supplying manual identity trajectories through the prefix reaches
`7.891873/13.429357` mm CD/track, but this is an online-supervised capacity
ceiling rather than a deployable method. The automatic strict-three-view
CoTracker3 arm reaches only `10.627040/20.414981` mm and fails its advancement
gate. Its supported query observations can be accurate, but support is sparse.

TAPIP3D is therefore tested as a different observation model, not as another
confidence formula around the existing triangulations. Its official model
tracks persistent query identities directly in metric world-space 3D from
RGB-D, intrinsics, and extrinsics.

## Question

On the already-open `single_lift_cloth` source interaction, can official
TAPIP3D recover nine persistent material identities with both:

1. materially broader support than strict-three-view CoTracker3; and
2. materially lower displacement error on the frames where the two methods can
   be compared fairly?

This is an association-oracle competence control. The nine query locations are
released manual identity coordinates at frame zero only. Later manual
coordinates cannot enter prediction generation or sealing.

## Frozen input and model

The prediction receives camera-0 RGB-D frames `[0, 121)`, calibrated
world-to-camera extrinsics, intrinsics, and nine frame-zero world queries.
These frame-zero queries agree with the sensor depth to within 1 mm.

The official TAPIP3D checkout is pinned to
`4cb7e69a1687f67d56ec3e506768f51f2c581b46`; the released checkpoint SHA-256
is `3a9514d526559838e6158360af2b857da596d064b06a94fae7fd3b85134b2b1e`.
Inference uses its native `384 x 512` resolution, six iterations, a 16-point
support grid, visibility threshold `0.9`, and bidirectional inference.

## Comparator

The comparator is the frozen CoTracker3 observation archive. Each frame-zero
manual query is associated with its nearest PhysTwin graph identity using
frame-zero geometry only. Strict three-view CoTracker3 displacement is then
re-anchored to the identical manual query coordinate.

The relative accuracy statistic is computed only on identity-frame pairs valid
for both trackers. TAPIP3D cannot win merely by reporting a different subset.
Separate support gates require it to cover the prefix and its late third.

## Locked gates

All conditions must pass:

- prefix support at least 70%;
- at least 20% lower displacement RMSE than CoTracker3 on shared support;
- overall displacement RMSE at most 5 mm;
- frame-zero anchor RMSE at most 2 mm;
- late-third support at least 50%; and
- late-third displacement RMSE at most 10 mm.

Passing authorizes only an automatic graph-query source study. It does not
authorize a fresh target, Bayesian integration, or a state-of-the-art claim.
Failure stops TAPIP3D as the PhysTwin sparse-identity feeder without tuning on
this opened trajectory.

## Artifact boundary

`seal-prediction` can receive only the locked protocol, locked RGB-D input
manifest, and official TAPIP3D result. Its parser exposes no case directory,
manual-trajectory, or CoTracker3-cue argument. It verifies all hashes, frame
count, query count, and exact frame-zero query binding, then writes a compact
prediction carrier that excludes copied RGB-D inputs.

Only `score`, after verifying that seal, may open the released source trajectory
and frozen CoTracker3 cues. Frames after 120 are not used.

An inference failure is recorded without scoring or automatic retry. Any
resolution or memory-setting change requires a committed amendment before a
new model run.

## Claim boundary

This is one opened source case with oracle frame-zero association. It tests
observation competence, not future prediction. It cannot establish transfer,
calibration, confirmation, or state of the art, and it grants no authority to
inspect held-v8.
