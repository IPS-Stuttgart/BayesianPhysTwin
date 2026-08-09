# Deform360 robot metric prefix

This source-only stage supplies sparse metric gauge evidence for Prob4D from
public Deform360 measurements. It projects the released synchronized gripper
taxel grid through the released camera calibration over exactly the registered
causal prefix, then maps those pixels to MotionCrafter's cover-resized grid.

It does **not** use `rendered_depth.h5`. That public depth is a privileged
full-sequence reconstruction and is reserved for explicitly labeled
reconstruction controls. The stage also reads no camera image, tactile value,
physical prediction, state innovation, confirmation payload, or target outcome.
Only robot values inside the registered prefix contribute to the output.

The output contains:

- `metric-prefix.npz`: exact `frame_indices`, sparse `points_world_m`, and
  `valid_mask` arrays accepted by the Prob4D sample materializer;
- `metric-calibration.json`: camera matrices and exact cover-resize provenance;
- `metric-prefix.json`: content-addressed identity, source hashes, support
  accounting, and information boundary; and
- `SHA256SUMS`.

Colliding projected taxels use the nearest camera-depth point with a stable
taxel-index tie break. Dense taxels are not declared independent evidence:
downstream fitting uses spatial dependence clusters and object-balanced
calibration. Visual residuals never alter prior source reliability; they enter
once through the robust metric-gauge fit.

This is a gauge-calibration input, not an object-state observation. A source
gate still has to establish adequate support, calibration, and transfer before
any confirmation payload can open.
