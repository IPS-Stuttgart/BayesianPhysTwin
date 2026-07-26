# TAPIP3D persistent-identity competence result v1

## Decision

The locked competence gate failed. Do not build the automatic graph-query arm
or integrate this off-the-shelf TAPIP3D observation into Bayesian-PhysTwin.
Do not tune visibility, resolution, iteration count, query selection, or the
acceptance thresholds against this opened result.

This is a negative observation-feeder result on one already-open source
interaction. It is not a future-prediction, transfer, confirmation, calibration,
or state-of-the-art result.

## Result

All errors are Euclidean RMSE in millimetres over the causal RGB-D prefix.

| Locked quantity | TAPIP3D | Strict 3-view CoTracker3 | Gate | Result |
| --- | ---: | ---: | ---: | --- |
| Finite-target support | **85.49%** | 18.17% | at least 70% | pass |
| Shared-support displacement RMSE | 8.233 | 8.325 | at least 20% relative gain | fail: 1.11% |
| Own-support displacement RMSE | 10.168 | 8.325 | at most 5.000 | fail |
| Frame-zero anchor RMSE | **0.116** | 0.000 | at most 2.000 | pass |
| Late-third support | **65.63%** | not gated | at least 50% | pass |
| Late-third displacement RMSE | 13.771 | not gated | at most 10.000 | fail |

TAPIP3D therefore solves the sparse-support problem but not the fine material
identity problem. Its support is 4.70 times the comparator's on this case, yet
on their 159 shared identity-frame pairs it is effectively tied rather than
materially more accurate.

## Error structure

The TAPIP3D displacement error has:

- 4.423 mm framewise common-translation RMSE; and
- 9.420 mm RMSE after removing the framewise common translation.

The remaining error is therefore predominantly identity-specific or
non-rigid, not a single camera/world translation that the existing
gauge-aware update could remove. Its late-prefix increase to 13.771 mm is also
incompatible with the intended persistent-anchor role.

CoTracker3 shows the opposite pattern on its sparse strict-support subset:
9.246 mm common-translation RMSE but 4.928 mm after translation removal. That
supports retaining the existing bias-aware treatment for highly redundant
CoTracker3 observations, while confirming that support remains its bottleneck.

## Evidence boundary

The prediction was generated from camera-0 RGB-D frames `[0, 121)`,
calibration, and nine released manual identity coordinates at frame zero only.
The prediction seal was written before the later manual source trajectory or
CoTracker3 cues were available to the scorer.

The raw official result SHA-256 is
`4e8538b2d8f4abf1fdf573b7dff84805bd3e29ca443625fd08ad949435bbc5d3`.
The compact sealed carrier SHA-256 is
`f491da4ed3bb4e418b594b3d5404306e03db214455b50703160f4423bcd5cc7d`;
its manifest SHA-256 is
`939287b2eadcb154c1a8dd4bee03a6bc2213b70d719efa9728e2769640347744`.
The scored result SHA-256 is
`44833b68ffd56742f3a020dd711e86c879be37a11145ebe6929f11da5f72eb8b`.

No frames after 120, fresh-object result, or held-v8 artifact were used.

## Implication

The opened PhysTwin evidence still points to a missing deployable sparse
identity channel, but a broad-support tracker is not sufficient. It must also
preserve deformable material identity at approximately 5 mm or better over the
full prefix and avoid late drift.

The next observation candidate should add genuinely independent metric
evidence or task-specific adaptation validated on source objects. Another
post-hoc confidence or gauge correction around these TAPIP3D coordinates is
not justified by this gate.
