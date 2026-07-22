# PokeFlex Regret-Guard Calibration v1

## Purpose

The source-calibrated D405 regret guard passed its registered same-object/new-take
replication: 6.418 to 6.222 mm CD_UL1, a 3.06% improvement, with both objects and
all three takes improving. This protocol asks whether the exact same deployment
artifact transfers to previously unseen objects.

No coefficient, feature, support range, selector correction, candidate arm, or
threshold is refit on calibration data.

## Cohort

The four calibration takes were fixed in the original PokeFlex split and exist
in the public download:

- `3dPrintedPyramid_T2`;
- `Beanbag_T2`;
- `FoamCylinder_T2`;
- `PlushMoon_T2`.

Only archive names and sizes were checked before lock. No archive member, sensor
frame, mesh, or outcome was read. There is no replacement. The eight target
objects remain sealed.

## Gates

The no-refit calibration result passes only if all conditions hold:

- at least 1% object-balanced improvement over the released checkpoint;
- at least three of four object means improve;
- no object regresses by more than 1%;
- at most 10% of accepted frames are harmful;
- at least one frame is accepted.

The published 6.498 mm Kinect reference is reported but is not a gate because
this calibration cohort is not the official target cohort.

Passing allows a target protocol to be drafted. It does not open target objects
automatically and is not itself a state-of-the-art claim.

## Lock

- Protocol:
  `configs/sota/pokeflex_independent_depth_regret_guard_calibration_v1.json`
- Protocol SHA-256:
  `cef28df26d7710670aa3d73883462e896ec2d903c3be5faf00fee6a8d00a2644`
- Frozen source result SHA-256:
  `6065d1178796eba949b0411fbd57b53184e39d38f6559357c740d63b8b47398b`
- Parent prospective result SHA-256:
  `0fc409a1bf85d4ef0e697bdb5604689094914ac57d32652660685007bfd67b98`
