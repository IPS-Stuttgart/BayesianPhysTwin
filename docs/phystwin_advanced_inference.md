# Advanced PhysTwin Inference

This workflow separates three latent causes that must not be conflated:

1. perception reliability and gross outliers;
2. ambiguity in physical/controller parameters;
3. systematic simulator discrepancy.

All selection uses a fit/validation split inside the released training
interval. Future frames remain untouched until final evaluation.

## Hierarchical interaction pooling

Run the same grouped object/controller profile grid on at least two
interactions. Then combine the saved `parameter_profile.npz` files:

```bash
bpt-combine-phystwin-profiles runs/joint/hierarchical \
  lift=runs/lift/parameter_profile.npz \
  stretch=runs/stretch/parameter_profile.npz \
  --temperature lift=0.75 \
  --temperature stretch=1.25 \
  --object-deviation-std 0.03 \
  --object-deviation-std 0.075 \
  --object-deviation-std 0.15 \
  --object-deviation-std 0.30 \
  --object-deviation-prior-scale 0.075
```

Temperatures can normalize interactions for unequal fit lengths. Without
`--object-deviation-std`, the command enforces one exact shared object scale.
With deviation candidates, it marginalizes a population scale and a discrete
trial-deviation hyperparameter while retaining trial-specific controller
scales. Each output NPZ can be passed back to `bpt-phystwin-refit` through
`--profile-weights` for full-state posterior prediction.

## Raw camera reliability cues

The released continuous CoTracker confidence is unavailable, but the archived
binary visibility and segmentation masks can be recovered when `cotracker/`,
`pcd/0.npz`, and `mask/processed_masks.pkl` are present:

```bash
bpt-build-phystwin-raw-cues \
  final_data.pkl /path/to/raw/CASE runs/CASE/raw_cues.npz \
  --base-cues runs/CASE/motion_cues.npz \
  --summary-json runs/CASE/raw_cues_summary.json
```

Final tracks are matched to raw camera queries by their initial 3D points. The
builder requires a one-to-one mapping within tolerance and stores source camera,
source query, raw visibility, and normalized inside-mask boundary distance.
Use `--boundary-scale` in `bpt-phystwin-refit` to match the cue's normalized
pixel units.

## Automatic MotionCrafter graph association

The dense evaluator pins the official MotionCrafter repository at revision
`1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257`. It consumes the released
`point_map`, `valid_mask`, `scene_flow`, and `deform_mask` arrays. MotionCrafter
predicts Eulerian forward scene flow, not persistent point identities, so the
adapter constructs identities explicitly:

1. resize the calibrated PhysTwin object mask and frame-zero depth with the
   same cover-resize transform used by MotionCrafter;
2. fit a trimmed frame-zero Sim(3) from same-pixel object depth pairs;
3. chain forward scene flow by gated nearest point-map transport, rejecting
   collisions and recording transport error and survival;
4. infer a sparse soft map from PhysTwin surface vertices to MotionCrafter
   trajectories using frame-zero distance, training-prefix persistence, the
   exact spring graph, and a local injectivity penalty;
5. freeze the map before reporting future dense graph error.

The optional training-motion likelihood compares relative MotionCrafter motion
with PhysTwin's automatic CoTracker graph trajectories. Development controls
did not justify enabling it, so the frozen confirmation uses
`--motion-strength 0`, one transport candidate, an 8-candidate graph map, and a
4-pixel seed stride. `gt_track_3d.pkl` is never required. When present, it is
opened only after association and supplies an audit of the automatic map.

Run the pinned native-rate extraction and frozen association with:

```bash
BPT_MOTIONCRAFTER_CAMERAS=0,1,2 \
  bash scripts/remote/run_phystwin_motioncrafter.sh CASE

bpt-select-phystwin-motioncrafter-view \
  runs/motioncrafter-selection.json \
  /path/to/CASE/camera0_native/association_frozen/summary.json \
  /path/to/CASE/camera1_native/association_frozen/summary.json \
  /path/to/CASE/camera2_native/association_frozen/summary.json
```

Camera selection minimizes

```text
training dense graph error / training-end graph coverage.
```

This score uses no future values and no sparse manual tracks. Coverage is in
the denominator because a view that retains only a tiny easy subset can report
a misleadingly low error. The development rule selected camera 0 in all three
interactions; confirmation therefore runs camera 0 only.

Because the primary score compares against the released training simulation,
the report also evaluates a perception-only sensitivity score,

```text
(frame-zero association error + training motion disagreement)
/ training-end graph coverage.
```

It independently selects camera 0 in all three development interactions. The
confirmation decision is therefore unchanged without simulator-error input.

The upstream pipeline supports overlapping temporal windows, but its public
`run.py` currently forwards `overlap=0` regardless of the CLI value. The
`run_motioncrafter_overlap.py` adapter repairs that forwarding at runtime while
checking the pinned revision. Overlap, multi-candidate transport, naive
multiview pooling, training-motion matching, and reverse-video tracking are
development controls, not parts of the frozen method. In development, reverse
tracking restored coverage but raised sparse future audit error to 102.65 mm;
it must not be used as an accuracy result.

The output `summary.json` records every input hash, transform, transport error,
per-frame graph coverage, automatic dense error, and optional manual audit.
The associated NPZ retains graph observations, validity, reliability,
candidate weights, dense trajectories, camera IDs, and any explicitly run
backward control.

On the frozen 19-case camera-0 confirmation, 17 cases retain at least three
paired future manual frames. Their equal-case mean within-sequence correlation
is `0.661 [0.449, 0.843]`, with 15/17 positive correlations and pooled frame
correlation `0.854`. Matched case-mean automatic-minus-manual error is
`-1.28 mm [-11.19, +8.26]`. Mean future graph coverage is only
`31.42% [22.03%, 41.88%]`, and two cases retain no future sparse audited
identity. Describe this as a confirmed automatic visible-surface surrogate,
not a full replacement for manual identities.

## Constrained action residual

Fit a low-rank residual to a baseline or posterior-mean trajectory:

```bash
bpt-fit-phystwin-residual-dynamics \
  final_data.pkl baseline_trajectory.pkl gt_track_3d.pkl \
  runs/CASE/residual \
  --fit-end-frame 44 \
  --train-end-frame 59 \
  --maximum-residual-m 0.01
```

The deformation basis and candidate dynamics use only fit frames. Rank,
persistence, and ridge strength are selected on validation CD and manual-track
error. The chosen dynamics are refitted through the training endpoint, seeded
from the final training observation, and rolled into the future using only
controller position/spread/motion features. A fixed reference-space k-nearest
map lifts tracked residuals to the full simulated object. State and pointwise
corrections are capped.

This is a simulator-discrepancy mean model. It does not calibrate discrepancy
uncertainty and must not be interpreted as a learned physical parameter.

## Hierarchical residual magnitude

The development-only hierarchical alternative removes the hard clipping kink.
For a raw lifted residual vector `r` and positive interaction scale `s`, it
uses smooth radial shrinkage:

```text
r_shrunk = s * tanh(||r|| / s) * r / ||r||.
```

Its magnitude approaches `s` smoothly while preserving direction. For every
held-out development interaction, rank, persistence, ridge, observation noise,
population mean, and population standard deviation are selected using only the
other two interactions. The held-out local scale posterior then uses the
interaction's validation pseudo-track and manual-track residual channels with
equal channel weight. Future frames never enter selection.

```bash
bpt-fit-phystwin-hierarchical-residual \
  /path/to/phystwin-eval runs/phystwin-hierarchical-residual

bpt-compare-phystwin-residual-scales \
  /path/to/phystwin-eval \
  runs/phystwin-hierarchical-residual \
  runs/phystwin-residual-cap-controls \
  runs/phystwin-residual-scale-comparison.json
```

The comparison reevaluates matched 10 mm and 30 mm hard-cap trajectories and
the hierarchical trajectory over the whole released future and contiguous
early, middle, and late thirds. It reports equal-interaction paired
moving-block intervals. With only three interactions from one object, this is
an exploratory prior comparison, not evidence that the selected population
scale transfers to a broader cohort.

## Persistent and Bayesian endpoint anchors

The full benchmark shows that extrapolating residual dynamics is not required
for the strongest transfer. A persistent anchor holds the final temporally
filled training residual under the same 10 mm pointwise cap. The robust
Bayesian version filters each 3D residual with a random-walk state and a
Gaussian/outlier mixture before taking the endpoint posterior mean:

```bash
bpt-fit-phystwin-bayesian-anchor \
  final_data.pkl inference.pkl gt_track_3d.pkl runs/CASE/bayesian_anchor \
  --fit-end-frame 60 \
  --train-end-frame 81 \
  --maximum-residual-m 0.01
```

Binary visibility/motion validity enters as missingness. The output posterior
contains per-track mean, variance, final inlier probability, and update count.
Future variance is propagated without future observations. A direct
manual-track NEES audit rejects calibration of the validation-selected raw
variance; use the strict conformal command below for coverage-bearing outputs.

Retrieve only the compact released inputs and reproduce the locked full-cohort
comparisons with:

```bash
bpt-fetch-phystwin-eval-data /path/to/phystwin-eval
bpt-confirm-phystwin-residual \
  /path/to/phystwin-eval runs/phystwin-confirmatory
bpt-confirm-phystwin-residual-baselines \
  /path/to/phystwin-eval runs/phystwin-baselines
bpt-confirm-phystwin-bayesian-anchor \
  /path/to/phystwin-eval runs/phystwin-bayesian-anchor
bpt-audit-phystwin-calibration \
  /path/to/phystwin-eval runs/phystwin-calibration \
  --anchor-run-dir runs/phystwin-bayesian-anchor
bpt-analyze-phystwin-horizon \
  /path/to/phystwin-eval \
  runs/phystwin-confirmatory runs/phystwin-baselines \
  runs/phystwin-horizon.json
```

The calibration command preserves a strict split. It uses fixed anchor
hyperparameters, fits state only on `[0, fit_end)`, calibrates per-frame metric
scores on `[fit_end, train_end)`, and never refits the predictor on calibration
frames. Finite-sample one-sided coverage is distribution-free only when the
calibration and future scores are exchangeable. With 90% nominal bounds on the
19 nondevelopment cases, the posterior-scaled point estimate is 90.63% macro
coverage for manual-track error and 75.36% for CD; track coverage falls from
98.66% early to 82.28% late. Additive wrapping covers 100.00%/97.84% CD/track
but widens median case-mean bounds to 38.87/42.68 mm.

The same output reports 3D NEES against manual correspondences. The operational
selected posterior has pooled mean NEES 1355.05 versus the calibrated
expectation of 3 and 38.31% coverage under its nominal 90% ellipsoid. Six cases
selected zero process noise and have 0.13% coverage. A separate strict fixed
5 mm process-noise posterior overdisperses instead (mean NEES 0.62; 99.63%
coverage). These are calibration results, not permission to interpret every
Bayesian covariance in the pipeline as frequentist coverage.

The horizon command is a post-hoc mechanism analysis. It splits each official
future interval into contiguous early, middle, and late thirds, reevaluates the
saved action and persistent trajectories, and correlates the 10 mm capped
training-endpoint residual with future residual fields after removing global
translation. It does not refit either trajectory.

## Spring-graph discrepancy posterior

The endpoint posterior can be conditioned jointly on the exact released
PhysTwin object spring graph. The random-walk normalized Laplacian is
dimensionless and leaves constant displacement unpenalized. With endpoint mean
`m`, variance-derived weights `W`, and graph displacement `b`, the solver uses

```text
0.5 * ||W^(1/2) (b - m)||^2 + lambda * ||L b||^2
```

and exposes the implicit spatial covariance

```text
v_ref * (W + 2 * lambda * L.T L + ridge * I)^-1.
```

The graph extra supplies SciPy's sparse matrices and conjugate-gradient solver:

```bash
python3 -m pip install -e ".[data,graph]"
bpt-fetch-phystwin-eval-data /path/to/phystwin-eval
bpt-compare-phystwin-graph-anchors \
  /path/to/phystwin-eval runs/graph-development \
  --cohort development --select-prior-strength
bpt-compare-phystwin-graph-anchors \
  /path/to/phystwin-eval runs/graph-confirmation \
  --cohort confirmation --prior-strength 0.1 --covariance-probes 16
bpt-compare-phystwin-graph-anchors \
  /path/to/phystwin-additional runs/graph-additional \
  --cohort all --prior-strength 0.1 --covariance-probes 16
```

The matched methods share the fixed robust endpoint posterior and 10 mm cap.
Raw leaves untracked state vertices at zero, kNN uses inverse-distance lifting,
and graph smoothing conditions all vertices jointly. The seven-value strength
grid selected `lambda = 0.1` only on the three designated development cases.
Frozen evaluation reduces equal-case Laplacian energy by 67.83% on the 19-case
cohort and 61.29% on the additional cohort. Relative to kNN, however, main
CD/track change by +0.81%/-0.97% and additional CD by +0.22%; every paired
interval crosses zero. The covariance diagonal is a fixed-seed stochastic
diagnostic and has not been calibrated as future coverage.

For the separate label-free cloth release, the protocols use the full released
training interval, no validation labels or selection, and no future inputs:

```bash
bpt-fetch-phystwin-eval-data /path/to/phystwin-additional --additional
bpt-confirm-phystwin-additional-anchor \
  /path/to/phystwin-additional runs/additional-anchor
bpt-confirm-phystwin-additional-bayesian \
  /path/to/phystwin-additional runs/additional-bayesian
bpt-confirm-phystwin-additional-anchor \
  /path/to/phystwin-additional runs/additional-se3 --spatial-mode se3
bpt-confirm-phystwin-additional-anchor \
  /path/to/phystwin-additional runs/additional-sim3 --spatial-mode sim3
bpt-confirm-phystwin-additional-anchor \
  /path/to/phystwin-additional runs/additional-affine --spatial-mode affine
```

The additional-cohort per-point and fixed Bayesian anchors improve all 11
future CD results. Translation, SE(3), Sim(3), and affine are post-hoc spatial
controls; use `--global-translation` for the backward-compatible translation
alias or `--spatial-mode` for any control. All confirmation commands write
`locked_protocol.json` before evaluating a case.

Compare the frozen per-point run directly with all controls using:

```bash
bpt-compare-phystwin-additional-controls \
  runs/additional-anchor runs/additional-spatial-comparison.json \
  runs/additional-global runs/additional-se3 \
  runs/additional-sim3 runs/additional-affine
```

## Bias attribution diagnostic

Before interpreting simulator residual as perception drift, test the implied
correction on held-out manual correspondences:

```bash
bpt-diagnose-phystwin-bias \
  final_data.pkl baseline_trajectory.pkl gt_track_3d.pkl bias.json \
  --fit-end-frame 44 \
  --train-end-frame 59
```

The diagnostic fits the random-walk bias on fit residuals, holds its final state
fixed, subtracts it from later pseudo-measurements, and reports manual-track
error. A worse result indicates that the residual is dominated by simulator
mismatch rather than sensor bias; an alternating correction/refit loop is then
not identified without an independent discrepancy channel.

## Paired evaluation

Create a JSON manifest with one entry per case:

```json
{
  "cases": [{
    "name": "CASE",
    "final_data": "/path/final_data.pkl",
    "gt_track_3d": "/path/gt_track_3d.pkl",
    "baseline_trajectory": "/path/baseline.pkl",
    "candidate_trajectory": "/path/candidate.pkl",
    "start_frame": 59
  }]
}
```

Then run paired moving-block bootstrap evaluation:

```bash
bpt-compare-phystwin-trajectories manifest.json comparison.json \
  --samples 10000 \
  --block-length 5 \
  --seed 20260710
```

The report includes official per-case means, paired frame intervals, and an
equal-case macro interval that resamples both cases and temporal blocks. With a
small number of interactions, this quantifies the registered subset only.
