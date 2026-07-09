# Official PhysTwin Integration

The adapter is grounded in the official
[`Jianghanxiao/PhysTwin`](https://github.com/Jianghanxiao/PhysTwin) repository at
commit `2b6630528141b9cba5a7677c8b88b2129b4a8390`.

## Exact Tracking Contract

PhysTwin stores the processed tracked points in `final_data.pkl`:

```text
object_points          (T, N, 3)
object_visibilities    (T, N)
object_motions_valid   (T, N) or (T-1, N)
```

Its spring-mass initialization places those `N` original tracked points first
in the simulator vertex array. At frame `t`, `compute_track_loss` compares
simulator vertex `i` directly with `object_points[t, i]` and gates it with
`object_motions_valid[t-1, i]`. The adapter reproduces that indexing exactly.
The `nearest` correspondence option exists only to mirror PhysTwin's separate
evaluation script and is not the default optimization contract.

Official artifact locations for case `CASE` are normally:

```text
data/different_types/CASE/final_data.pkl
experiments/CASE/inference.pkl
```

PhysTwin writes `inference.pkl` as a NumPy trajectory with shape `(T, M, 3)`.

## Export and Replay

```bash
bpt-export-phystwin-residuals \
  data/different_types/CASE/final_data.pkl \
  experiments/CASE/inference.pkl \
  runs/CASE/residuals.csv \
  --summary-json runs/CASE/export.json \
  --replay-summary-json runs/CASE/replay.json \
  --scored-csv runs/CASE/scored.csv
```

By default, rows rejected by PhysTwin's `object_motions_valid` gate are omitted.
Use `--include-invalid` to retain them for failure analysis. Visibility supplies
the fallback confidence/occlusion cue.

Optional cue arrays can be passed in one `.npz` file. Each array may have shape
`(T, N)` or `(T-1, N)` and use these keys:

```text
confidence
occluded
boundary_distance
flow_inconsistency
```

Build a simulator-independent continuous motion-consistency sidecar directly
from the tracked geometry:

```bash
bpt-build-phystwin-cues \
  data/different_types/CASE/final_data.pkl \
  runs/CASE/cues.npz \
  --summary-json runs/CASE/cues.json
```

The cue compares each visible track motion with the median motion of its
first-frame neighbors. It retains the magnitude that PhysTwin's binary local
motion filter discards. Use a meter-scale replay decay appropriate for the
case, for example:

```bash
bpt-export-phystwin-residuals \
  data/different_types/CASE/final_data.pkl \
  experiments/CASE/inference.pkl \
  runs/CASE/residuals.csv \
  --cues-npz runs/CASE/cues.npz \
  --replay-flow-scale 0.01 \
  --replay-summary-json runs/CASE/replay.json
```

The processed artifact still discards richer raw CoTracker confidence and
mask-boundary distance. A later preprocessing patch should preserve them rather
than infer them from simulator residuals.

## Model Boundary

The first Bayesian model covers tracked 3D correspondences. PhysTwin's
single-direction Chamfer term chooses a nearest simulator point for each visible
observation every frame; it is useful for initialization but is not treated as
a normalized fixed-correspondence likelihood here. Render loss also remains
outside the Bayesian physical refinement stage.

## Security

`final_data.pkl` and `inference.pkl` are Python pickle files. Load only trusted
artifacts produced by an official or locally controlled PhysTwin run.
