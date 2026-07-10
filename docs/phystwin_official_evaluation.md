# Official PhysTwin 3D evaluation

`bpt-evaluate-phystwin-official` reproduces the released PhysTwin definitions for
the two 3D metrics in the paper's quantitative tables:

- `chamfer_distance_m`: per-frame, one-way L1 nearest-neighbor distance from the
  visible observed point cloud to the predicted surface, averaged over frames.
- `track_error_m`: frame-0 nearest-vertex correspondences for the manually
  annotated tracks, with Euclidean error averaged over points and then frames.

Run it with the released case data and any full-state trajectory:

```bash
bpt-evaluate-phystwin-official \
  trajectory.pkl final_data.pkl gt_track_3d.pkl split.json evaluation.json
```

The output records metric definitions, split boundaries, absolute input paths,
and SHA-256 hashes. This supports matched case-level comparisons. The PhysTwin
paper's Table 1 values are averages over 22 scenarios, so a smaller case subset
must not be presented as a replacement aggregate row.
