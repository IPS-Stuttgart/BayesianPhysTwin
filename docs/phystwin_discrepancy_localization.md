# PhysTwin Discrepancy Localization

## Purpose

The hierarchical structural calibration benchmark recovers controlled frame,
rest-geometry, and state perturbations, preserves the frozen identity path, and
then rejects every nonzero structural candidate on the three released sloth
interactions. This diagnostic therefore does not tune rest geometry again. It
asks where a low-rank correction must enter to explain the post-action response:

1. observation/readout;
2. simulator state at the O-plus prefix endpoint;
3. generalized force inside the simulator;
4. rest geometry, retained only as an information-matched negative control.

The result is diagnostic-only. The released cases have been repeatedly
examined and cannot select the final mechanism for the locked multi-action
protocol.

## Frozen comparison

Every branch uses:

- the first four symmetric normalized graph-Laplacian modes;
- the O-minus endpoint plus exactly six O-plus frames;
- four saved Bayesian-PhysTwin parameter particles and their fixed weights;
- the released controller trajectory;
- the same untouched continuation;
- fixed regularization declared before all released-case runs;
- official nonlinear Warp reruns with deterministic spring accumulation.

No future observation or manual track enters fitting. Manual tracks and future
point clouds are opened only after every correction artifact has been written.

The readout and state branches share the final graph coefficient field. A local
linear slope through the seven prefix states supplies the velocity update. The
force and rest branches use fixed one-step finite-difference responses from one
declared reference particle, fit a dimensionless ridge solution on the prefix,
and then rerun the inferred correction over all four particles. This reduces
the sensitivity cost without reducing the evaluation support.

Finite-difference steps are specified as maximum per-node force or displacement,
then converted to modal coefficients using each normalized mode's maximum
amplitude. This keeps the dimensionless ridge prior stable as graph size changes.

## Typed artifact

`DynamicDiscrepancyCorrection` stores:

- graph basis and eigenvalues;
- position and velocity coefficients;
- constant generalized-force coefficients;
- matched rest-geometry coefficients;
- prefix interval and frame period;
- fixed regularization and plausibility-limit diagnostics;
- source checksums and an explicit information boundary.

The JSON manifest hashes a non-pickled NPZ payload. The artifact constructor
requires rank 4, six O-plus frames, and declarations that future frames and
manual tracks were not consumed.

## Force boundary

The deterministic Warp subclass always captures an external-force kernel, but
the kernel reads a device-side enable flag and performs no write when disabled.
Calling `set_external_forces()` with an all-zero array leaves that flag off.
Every case reruns one reference particle after explicitly setting zero force
and requires bitwise identity with the baseline trajectory.

Nonzero external force is intentionally rejected when deterministic spring
forces are disabled. The released atomic-force path remains unchanged.

## Outputs

For every method, the summary reports:

- future Chamfer distance and manual-track error;
- early, middle, and late horizon results;
- far-graph observation error;
- particle-mixture coverage and NEES with the fixed variance floor;
- field magnitude, graph roughness, and mechanism-specific residual energy;
- framewise Chamfer/track correlation.

The aggregate promotes no mechanism. It reports whether force beats readout on
track, Chamfer, late horizon, and far graph without hitting its force limit;
whether a state restart matches readout; and whether cross-view evidence can
support an observation-bias interpretation.

## Observation audit

The released `final_data.pkl` normally stores fused 3D object tracks. Without
per-view material identities, continuous confidence, object-frame transforms,
or matched surface normals, cross-camera transfer, confidence regression,
object-frame consistency, and point-to-plane tests are not identifiable. The
audit records those tests as unavailable. If future artifacts provide per-view
3D tracks in a common calibrated frame, the same code fits each view and runs
leave-one-view-out correction transfer.

## Commands

Run one case:

```bash
bpt-diagnose-phystwin-discrepancy-location \
  /path/to/PhysTwin \
  /path/to/case/final_data.pkl \
  /path/to/case/inference.pkl \
  /path/to/case/optimal_params.pkl \
  /path/to/case/checkpoint.pth \
  /path/to/parameter_profile.npz \
  /path/to/known.twin_belief.npz \
  /path/to/case/gt_track_3d.pkl \
  /path/to/output/case \
  --train-end-frame 30
```

Aggregate completed case summaries:

```bash
bpt-aggregate-phystwin-discrepancy-location \
  /path/to/output/aggregate.json \
  /path/to/output/single_lift_sloth/summary.json \
  /path/to/output/double_lift_sloth/summary.json \
  /path/to/output/double_stretch_sloth/summary.json
```

## Claim boundary

A successful force branch means only that a constant low-rank force is a better
location for the released predictive correction than an output offset. It does
not identify friction, support, self-contact, or viscoelasticity. Mechanism
selection requires the locked same-object protocol with measured actuation,
registered support/contact geometry, reversals, rates, holds, and slip trials.
