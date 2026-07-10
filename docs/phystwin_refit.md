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

The mixture NLL is zero-shifted and scaled by the effective inlier variance so
its local quadratic term matches PhysTwin's smooth-L1 scale. Observation noise
and model discrepancy remain separate command-line parameters; the likelihood
uses their sum. Continuous cues must be computed before simulator residuals are
observed.

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

The runner supports either the released per-spring parameterization (`dense`)
or two log scales around the released checkpoint (`grouped`): one for object
springs and one for controller springs. Both are point refits, with optional
contact parameters. Dashpot and drag damping remain fixed because
the official kernel takes them as non-differentiable scalar inputs. It is not
yet a parameter posterior, and the processed motion cue is not a calibrated
replacement for raw tracker confidence or mask-boundary uncertainty. Those are
explicit next-stage requirements, not claims supplied by this integration.

The input files are Python pickles. Load only trusted official or locally
generated artifacts.
