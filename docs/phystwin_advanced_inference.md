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
