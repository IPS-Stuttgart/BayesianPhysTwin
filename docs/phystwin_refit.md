# Headless PhysTwin Refit

`bpt-phystwin-refit` runs a fixed-correspondence reliability objective inside
the released PhysTwin Warp simulator without importing its rendering, Open3D,
Gaussian-splatting, or GUI stack. The integration is pinned to official commit
`2b6630528141b9cba5a7677c8b88b2129b4a8390`.

## Reconstructed Contract

The runner rebuilds the official spring graph from the first processed frame,
surface/interior points, controller points, and optimized radius settings. It
casts geometry to float32 before the radius/k-nearest queries and preserves the
official object-springs-first ordering. Loading aborts unless both the total and
object spring counts match the checkpoint.

For released case `single_push_rope_1`, the reconstruction gives:

```text
2,323 simulator vertices
45,472 object springs
26 controller springs
45,498 springs total
```

## Matched Objectives

Every variant starts from the same released checkpoint, uses the same simulator
and training frames, and omits Chamfer so the comparison isolates direct track
correspondences:

- `hard`: PhysTwin's processed `object_motions_valid` gate.
- `visible`: every visible track receives equal weight.
- `cue`: visible tracks receive continuous residual-independent cue weights.
- `mixture`: cue values are prior inlier probabilities in a Gaussian/broad-
  Gaussian mixture.
- `markov_cue`: cue weights pass through a causal persistent-state filter.
- `markov_mixture`: filtered cue probabilities condition the robust mixture.

The mixture NLL is zero-shifted and scaled by the effective inlier variance so
its local quadratic term matches PhysTwin's smooth-L1 scale. Observation noise
and model discrepancy remain separate command-line parameters; the likelihood
uses their sum. Continuous cues must be computed before simulator residuals are
observed.

The Markov variants use fixed inlier/outlier persistence `0.98/0.90`. Filtering
is causal: reliability at frame `t` uses cue values only through `t`, never
future cues or simulator residuals.

Audit static and Markov priors on the exact target-visible refit support:

```bash
bpt-evaluate-phystwin-priors final_data.pkl cues.npz prior_evaluation.json \
  --flow-scale 0.005 \
  --inlier-persistence 0.98 \
  --outlier-persistence 0.90
```

The reported hard-gate calibration is a consistency diagnostic only. The gate
uses related local-motion logic and is not independent corruption truth.

## Runtime

The official low-level simulator requires compatible CUDA builds of PyTorch and
NVIDIA Warp. Keep those heavy dependencies outside the package's default
installation. PhysTwin's pinned source sets `cuda:0` during import; use
`CUDA_VISIBLE_DEVICES` to remap another physical GPU.

```bash
PYTHONPATH=/path/to/torch-warp-runtime:src \
bpt-phystwin-refit \
  /path/to/PhysTwin \
  /path/to/final_data.pkl \
  /path/to/optimal_params.pkl \
  /path/to/best_199.pth \
  /path/to/cues.npz \
  runs/CASE/refit_mixture \
  --variant mixture \
  --fit-end-frame 48 \
  --train-end-frame 64 \
  --epochs 20 \
  --learning-rate 1e-3 \
  --observation-variance 2.5e-5 \
  --model-discrepancy-variance 0 \
  --outlier-variance-multiplier 100 \
  --flow-scale 0.005 \
  --spring-parameterization grouped \
  --early-stopping-patience 3 \
  --released-trajectory /path/to/inference.pkl
```

Set `--epochs 0` for a checkpoint-restoration parity replay. Each output folder
contains `trajectory.pkl`, `refit_checkpoint.pt`, `history.json`, and
`summary.json`. The summary records input hashes, both code commits, graph
hashes, runtime versions, parameter movement, train/test errors, and a common
cue-weighted evaluation shared by all variants.

When `--fit-end-frame` is set below `--train-end-frame`, only the earlier
interval receives gradient updates. The runner evaluates the intervening
frames after every epoch, stops after the requested patience, and restores the
lowest hard-valid validation-RMSE parameters. The final summary reports fit,
validation, and untouched test intervals separately.

## Interpretation Boundary

The runner supports the released per-spring parameterization (`dense`), two log
scales around the released checkpoint (`grouped`), regularized principal-axis
material bands (`regional`), part-pair groups (`part_pair`), or a smooth
canonical spring basis (`canonical_basis`). The canonical mode uses normalized
Gaussian-RBF weights over object-spring midpoints and one separate controller
coefficient. It is exactly centered on the released checkpoint and writes a
complete `canonical_spring_basis.npz` artifact. It remains a low-rank point
refit, not a spatial posterior or a reproduction of NeuSpring's neural field.

Dashpot and drag damping are non-differentiable scalar inputs in the official
kernel. They can be changed between runs with `--dashpot-log-scale` and
`--drag-log-scale` for causal profile sweeps, but cannot receive Warp gradients.
The grouped grid below is a two-scale profile posterior, not a posterior over
dense springs, damping, contact, or topology. The processed motion cue is not a
calibrated replacement for raw tracker confidence or mask-boundary uncertainty.

The input files are Python pickles. Load only trusted official or locally
generated artifacts.

## Grouped Parameter Profile

A zero-update grouped run can evaluate a two-dimensional profile posterior over
object- and controller-spring log scales:

```bash
bpt-phystwin-refit ... \
  --variant mixture \
  --fit-end-frame 48 \
  --train-end-frame 64 \
  --epochs 0 \
  --freeze-collision \
  --spring-parameterization grouped \
  --profile-grid-count 9 \
  --profile-prediction-mass 0.999
```

The profile uses only fit frames for its likelihood. It averages track NLLs
within each frame before summing frame contributions, making the spatial
correlation tempering explicit. Independent zero-mean Gaussian priors apply to
both log scales. `parameter_profile.npz` stores the grid, posterior weights,
posterior mean trajectory, and epistemic variance; `summary.json` adds parameter
credible intervals and 90% observation-predictive coverage on fit, validation,
and test intervals. Temperature and prior scales are recorded configuration,
not hidden calibration constants.

`--profile-prediction-mass` evaluates the smallest highest-probability particle
set reaching the requested mass and renormalizes it for prediction. The summary
records requested mass, actual retained mass, and evaluated particle count.
This avoids spending full simulator rollouts on numerically irrelevant or
unstable extreme grid corners.

## Causal Model Discrepancy

Calibrate a saved profile without rerunning the simulator:

```bash
bpt-calibrate-phystwin-discrepancy \
  final_data.pkl parameter_profile.npz runs/CASE/discrepancy.json \
  --fit-end-frame 48 \
  --test-start-frame 64 \
  --observation-variance 2.5e-5 \
  --reference-trajectory inference.pkl
```

The fixed observation variance remains the perception term. A separate model
discrepancy variance is estimated from residual moments after subtracting the
observation and epistemic terms. For frame `t`, the estimate uses residuals only
through frame `t-1`; current-frame observations never set their own interval.
The exponential decay is selected by validation NEES from an explicit candidate
list, with smoother estimates winning exact ties.

This output is online one-step calibration. It may update during the test
sequence from prior test observations and must not be described as open-loop
future uncertainty. Static zero-discrepancy metrics remain beside it in the
summary so the contribution is auditable.
