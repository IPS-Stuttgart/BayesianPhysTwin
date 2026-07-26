# PhysTwin Rendered-AllTracker Prefix Competence v1

## Purpose

The strongest opened PhysTwin source ceiling uses nine manual material
identities during the allowed response prefix. Generic temporal point trackers
have not closed that gap: strict automatic CoTracker3 remains materially worse
than the manual ceiling, TAPIP3D accumulates late displacement drift, and a
static-background CoTracker3 gauge improves background tracks while worsening
moving object identities.

This control tests a narrower interface. At each allowed response-prefix frame,
the selected PhysTwin trajectory renders the same material graph point into
each camera. AllTracker then solves a direct two-frame correspondence from
that synthetic material appearance to the simultaneous real image. The method
does not carry a generic image identity through a long video, so it avoids
temporal drift by construction.

The experiment is deliberately an **association-oracle competence control**.
It receives the same nine frame-zero manual world positions used by the prior
TAPIP3D and SpatialTrackerV2 controls, but it never receives their later
trajectory before producing and sealing its prediction. A pass can justify
building an automatic graph-query interface. It is not itself deployable or a
state-of-the-art result.

## Frozen Prediction

- Case: `single_lift_cloth`, already opened for source development.
- Images and masks: frames 114 through 120 only, before the exclusive endpoint
  at frame 121.
- Cameras: `cam0`, `cam1`, and `cam2`.
- Identities: nine frame-zero manual world positions only.
- Material carrier: nearest PhysTwin graph node at frame zero, within 5 mm.
- Image pair per frame and camera: PhysTwin render followed by real RGB.
- Cycle check: real RGB followed by the PhysTwin render.
- AllTracker: revision `61f5b21`, four iterations, 512-pixel maximum side.
- View support: rendered alpha, object mask, tracker quality at least 0.5,
  forward/reverse cycle error at most 5 pixels.
- Metric reconstruction: at least two distinct calibrated camera poses and at
  most 3-pixel reprojection error.

All support decisions are independent of the PhysTwin state innovation. The
candidate geometry determines the material source query, but the residual
between the triangulated observation and physical state is not recycled into
perception reliability.

## Correlation Boundary

The three cameras are not treated as conditionally independent measurements.
Duplicate poses form one evidence group. Unknown cross-view correlation is
handled by averaging, rather than summing, per-view information in a
covariance-intersection-style update. A 5 mm common-mode covariance floor is
always present, and a two-view triangulation receives an additional 10 mm
floor. Duplicating a camera therefore cannot manufacture arbitrary confidence.

These covariances are conditional diagnostics. One opened case cannot establish
NEES or coverage calibration, and a visually coherent common-mode camera bias
remains observationally unidentifiable without physical or independent-modality
support.

## Sealed Outcome Order

Prediction has sole access to:

1. the already sealed PhysTwin render carrier for frames 114 through 120;
2. the selected physical trajectory;
3. the nine frame-zero world positions;
4. camera calibration and same-frame object masks;
5. the frozen AllTracker source and checkpoint.

It writes `prediction.npz`, `prediction_report.json`, and `PREDICTION_SEAL`.
The scorer first validates all three artifacts, the prediction hashes, and the
recorded information-boundary flags. Only then may it open:

- the later manual identity trajectory for frames 114 through 120;
- the frozen strict CoTracker3 source comparator.

The physical comparator is the frame-zero oracle position plus the selected
graph-node displacement. CoTracker3 requires three views, all three view
qualities at least 0.5, reprojection error at most 3 pixels, and forward/backward
error at most 5 pixels. Relative comparisons use exact shared identity-frame
support.

## Frozen Gate

All five requirements must pass:

1. candidate support is at least 50%;
2. candidate position RMSE is at most 5 mm;
3. final-frame candidate RMSE is at most 8 mm;
4. candidate RMSE is at least 10% lower than the physical comparator on shared
   support;
5. candidate RMSE is at least 20% lower than strict CoTracker3 on shared
   support.

Failure stops this exact interface without changing frames, identities,
cameras, tracker settings, support thresholds, covariance floors, anchoring,
comparators, or gates. Success permits only a separately preregistered
source-only experiment with automatically selected graph queries and guarded
Bayesian assimilation.

## Lock

The implementation is frozen at commit
`1a1968ad1851295f676060c469dfb5e0a76bfb98`. The observation builder,
competence functions, prediction runner, and outcome runner are bound by
SHA-256:

```text
fb92d0554976f073b87775bdd72f44c6cb310b86009e4af5015c6071dc32398f
9d214abf27be35f8f25af2110783ed45517e806538a6dd42624cb265bfee90e8
c74c60caa3b881083f9153a8e924fa543e9d42067980c72a14ac37af05433235
24a3d9ed31f257b531720d1fc69035277919d118511d892e78d6c5491833a23d
```

The exact source and deferred scoring hashes are recorded in the machine-readable
protocol. No held-v8 artifact may be opened, inspected, modified, or rerun.

## Outcome Parser Amendment

The predictor completed and was sealed before either scoring input was opened.
The first scorer invocation then validated that seal and opened the hash-locked
CoTracker3 comparator, but stopped before computing or writing a metric. The
archive stores full triangulated trajectories for 173 frames and network
quality only for the authorized 121-frame prefix. The frozen parser had
incorrectly required those two frame dimensions to be equal.

Commit `5e3f0b0173b6f903ad59d48d78862e26a6f093ca` changes only that schema
validation: quality may have shape `(camera, prefix_frame, track)` when it
covers every scored frame and has the same track inventory. It does not change
the prediction, identities, comparator, support rule, thresholds, metric, or
gate. The amended competence module and outcome runner have SHA-256
`a058dd18739818a4ca9eed7e739f9726c7e755655f161ad7406f8a9c13eaf044`
and
`29d296ad819a42b3a4efa31661aeb6ebe39c2371d76c245686cc28fbc29ea1fc`;
12 focused tests pass.
