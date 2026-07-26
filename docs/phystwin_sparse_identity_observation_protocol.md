# Covariance-aware sparse identity observation path

## Status

This is an opt-in Bayesian-PhysTwin development path motivated by the frozen
Prob4D and CoTracker3 diagnostics. It is not a positive result, an independent
evaluation, or a state-of-the-art claim. All existing observation-source names
and defaults remain unchanged.

The opened PhysTwin-22 cohort may be used only for smoke tests, model-class
diagnosis, and object-disjoint source development. No fresh Deform360 object
may be evaluated until the complete source transfer and calibration gates pass
and every independently supplied hash-only object exclusion is merged.

## Observation boundary

The new source name is
`final_data_plus_cotracker3_sparse_identity`. It deliberately retains the
released dense PhysTwin pseudo-tracks as the geometry channel and introduces a
separate CoTracker3 material-identity channel at directly supported graph
nodes. It does not use a fixed Prob4D/VGGT or dense/sparse point blend.

Only frames before `train_end_frame` may enter an observation. Future cue rows
are neither loaded nor used. Prior perception reliability is computed from:

- CoTracker3 source confidence;
- forward/backward cycle consistency;
- mask-boundary distance;
- multiview reprojection consistency;
- geometrically distinct camera support; and
- ray geometry.

The PhysTwin innovation is not an input to prior reliability. It enters once,
after the observation artifact is fixed, through the existing robust
inlier/outlier likelihood.

## Correlation and covariance

The artifact carries a full `3 x 3` covariance and an isotropic metric variance
in square metres for every point-frame observation.

- Cameras with duplicate poses are collapsed before support is counted.
- Unknown cross-view correlation is handled with normalized ray information;
  duplicate observations cannot accumulate independent precision.
- Three or more distinct views add leave-one-view-out disagreement covariance.
- Exactly two distinct views are an explicit fallback with lower prior
  reliability and additional metric variance.
- A shared camera/time bias floor remains in every covariance. Camera-only
  consistency is not claimed to identify coherent common-mode bias.
- Anchored displacements conservatively retain both the frame-zero and current
  triangulation uncertainty.

Metric endpoint variance is propagated into graph smoothing. Unsupported
identities retain the existing dense path. Prefix rejection produces the
existing exact zero-correction fallback relative to the unchanged physical
baseline.

## Development sequence

1. Run one already-open development case as a schema and numerical smoke test.
2. Compare the unchanged dense path against the opt-in sparse-identity path
   without modifying hyperparameters after future metrics are read.
3. Freeze a candidate family before object-disjoint source evaluation.
4. Require source transfer in both future Chamfer distance and manual identity
   error, non-degraded late horizon, and calibrated uncertainty.
5. Merge all hash-only object exclusions and preregister a genuinely fresh
   cohort only if every source gate passes.

The public PhysTwin-22 outcomes remain exploratory throughout. No held-v8
target, query, score, barrier, or outcome artifact is authorized by this
protocol.
