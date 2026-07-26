# PhysTwin Static-Scene Gauge Source Transfer v1

## Purpose

The opened PhysTwin-22 evidence leaves a large gap between manual prefix
identities and deployable automatic observations. The manual result of
`7.891873 mm` Chamfer and `13.429357 mm` future track error is an
online-supervised capacity ceiling: the same manual identity family is
observed in the prefix and scored in the future. It is not a deployable method
or a fair state-of-the-art comparison. The completed automatic CoTracker3 arm
reached `10.627/20.415 mm` and failed its advancement gate.

This source experiment tests an upstream, object-state-independent correction
to those automatic tracks. In a fixed calibrated camera, pixels that remain
outside every object and actuator mask during the allowed prefix belong to the
static scene. Their true image motion is zero. CoTracker3 drift on those
pixels therefore measures a tracker/camera nuisance field without comparing
the observation with PhysTwin state.

## Frozen Method

For each camera and allowed prefix:

1. Select background pixels that remain outside all dynamic masks.
2. Track them with the already frozen CoTracker3 checkpoint.
3. Cluster spatially correlated tracks before estimating support.
4. Fit a local RBF drift field and evaluate it by spatially held-out
   background tracks.
5. Admit the camera only when held-out background error improves by at least
   10%.
6. Correct only object tracks within 48 pixels of admitted background support.
   Every unsupported point and rejected camera uses an exact no-op fallback.

The local residual variance is retained in pixel-squared units and is not
divided by the number of dense or duplicated pixels. The correction never
reads a PhysTwin residual, manual identity, future frame, or future simulator
outcome. Manual tracks are opened only after each gauge artifact is written
and hashed, and only to score prefix competence.

## Cohort And Gates

`single_lift_cloth` is the sole development case. It improved prefix manual
mean error by 7.47%, RMSE by 2.86%, and late-prefix mean error by 2.88% on
common support. Those results froze the implementation and the following
transfer gates before evaluating the other 21 already-open source cases:

- at least 3% lower equal-case mean prefix manual error;
- at least 2% lower equal-case RMSE;
- at least 2% lower equal-case late-prefix mean error;
- mean-error wins in at least 14 of 21 cases;
- no case with more than 10% mean-error regression.

Failure stops this gauge route without changing the rank, background mask,
RBF bandwidth, support radius, quality threshold, or gates. Passing authorizes
only a separately locked simulator-assimilation experiment.

## Admissibility Amendment 1

The first transfer worker sealed a `rope_double_hand` gauge and then stopped
before producing a score because two of nine manual identities are non-finite
at frame zero. One other case, `double_stretch_sloth`, had already written a
score, but its metric values were not inspected and no aggregate was run.

Before transfer resumed, the scorer was amended to admit exactly the manual
identities finite at frame zero and to record their indices and the original
total. Future missingness remains handled by the predeclared common-support
mask. This changes neither the gauge, its admission rule, the cohort, nor any
gate, and it is an identity operation for cases whose frame-zero identities
are all finite.

## Boundaries

This is opened-source observation-feeder development, not independent
confirmation, calibration, or state-of-the-art evidence. It does not inspect
the sealed held-v8 cohort, PokeFlex targets, or any fresh-object target. It
does not establish that static-scene correction improves future simulation:
the transfer score is confined to the allowed observed prefix.

Prob4D remains a separate, optional observation/calibration feeder. No fixed
Prob4D/VGGT blend is used. Causal4D remains a separate downstream intervention
project and is not modified by this experiment.
