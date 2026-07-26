# SpatialTrackerV2 persistent-identity competence result v1

Status: completed negative one-case source control.

SpatialTrackerV2 is stopped as the direct PhysTwin sparse-identity feeder. No
automatic graph-query arm or fresh-object evaluation is authorized.

## Evidence order

The implementation was committed at `c219ee9`, after 839 tests passed. The
protocol was then committed and pushed at `8ea0d29`, while the designated
server output directory was empty. Only afterward was one official inference
run executed.

The raw result was sealed before the released manual prefix trajectory or
CoTracker3 cue archive was opened:

- raw result SHA-256:
  `c188d601fd665ccb671c9e1b4823fa6e1155f78176df11d5faea51aee8f3b375`;
- canonical prediction SHA-256:
  `7013b3a3865b19f0f9800e0744a73825eae681ade05cc4113c2c7cd51671242d`;
- prediction manifest SHA-256:
  `3c51a765a84e1bcd02189267d344d4dda88848a43c9413ec0383562bbdc4f020`;
- score result SHA-256:
  `762b2ccddfddcb8f8662a046694f10e278b1679e5a799d4e0664c5aa3dd8b62f`.

No retry or inference-setting change occurred. The prediction used RGB-D frames
`[0, 121)`, known camera calibration, and nine manual identity coordinates at
frame zero only. It did not access frames after 120, later manual coordinates,
or held-v8.

## Result

| Gate | Frozen requirement | SpatialTrackerV2 | Outcome |
| --- | ---: | ---: | --- |
| Prefix support | at least 70% | 85.83% | pass |
| Shared-support improvement over CoTracker3 | at least 20% | -21.90% | fail |
| Prefix displacement RMSE | at most 5 mm | 60.98 mm | fail |
| Frame-zero anchor RMSE | at most 2 mm | 3.14 mm | fail |
| Late-third support | at least 50% | 78.13% | pass |
| Late-third displacement RMSE | at most 10 mm | 117.84 mm | fail |

On the 159 identity-frame pairs supported by both methods, SpatialTrackerV2
has `10.149` mm displacement RMSE and strict-three-view CoTracker3 has `8.325`
mm. SpatialTrackerV2 therefore regresses by `21.90%` rather than achieving the
required improvement.

The broad-support error is not explained by one removable global translation.
Its framewise common-translation RMSE is `51.54` mm, but the residual RMSE after
removing that translation remains `42.83` mm. Re-anchoring the frame-zero point
would repair the 3.14 mm anchor gate only; displacement metrics are already
anchor invariant and remain decisively outside the gates.

## Interpretation

SpatialTrackerV2 provides the coverage that strict multiview CoTracker3 lacks,
but not reliable metric material identity on this PhysTwin source trajectory.
The result agrees with the earlier MVTracker and TAPIP3D controls: a generic
persistent 3D tracker can report broad support while its errors remain too
large for millimetre-scale Bayesian state updates.

This does not reject SpatialTrackerV2 on its published benchmarks. It rejects
this exact official offline RGB-D model, calibrated gauge conversion, and
direct sparse-identity role under the frozen PhysTwin competence gates.

No confidence threshold, score-time alignment, query change, or selective
fusion will be tuned on this opened trajectory. The next method effort should
return to the stronger Bayesian-PhysTwin lead: baseline-relative recursive
belief updates admitted by physical/action support, redundant observations,
explicit common-mode bias, calibrated regret bounds, and exact fallback.

## Claim boundary

This was an opened one-case association-oracle observation control. It is not a
prediction, transfer, calibration, confirmation, or state-of-the-art result.
It provides no authority to inspect or alter held-v8.
