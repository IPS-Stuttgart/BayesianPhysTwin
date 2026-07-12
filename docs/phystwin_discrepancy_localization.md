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

## Released diagnostic result

All three cases pass exact zero-force parity. An independent single-lift rerun
also reproduces the correction artifact ID and full rollout archive hash
exactly. The equal-case result is:

| Method | Future CD | Future track | Late track | Far-graph error |
| --- | ---: | ---: | ---: | ---: |
| Graph-persistent readout | **-17.42%** | **-13.52%** | **-8.59%** | **18.51 mm** |
| Prefix state | -3.43% | +1.04% | +2.08% | 26.78 mm |
| Constant generalized force | -1.89% | -1.27% | -1.16% | 26.95 mm |
| Matched rest control | -1.40% | -2.66% | -1.74% | 26.86 mm |

Relative to readout, force is `1.144x` worse on track, `1.209x` worse on CD,
and `1.432x` worse far from contact. It also reaches the predeclared force limit
in two cases. State is `1.173x` worse on track and degrades track overall. The
rest control reaches its geometry limit in every case. None is promoted.

The aggregate conclusion is
`readout_is_best_but_physical_vs_observation_location_unresolved`. The exact
aggregate SHA-256 is
`ec3d7c21e706d2ef2f9fb447730d3416067a5a42702644fd5b24fcc5cb9333f2`.

## State-correction decay audit

The frozen state rollouts support one additional post-hoc diagnostic without
opening observations or selecting a model. The audit measures the RMS distance
between the prefix-state and nominal trajectories, its component along the
injected field, and the orthogonal component. It fits an exponential only to
the transient above a tail floor and requires log-space `R^2 >= 0.80` before
reporting that fit as adequate.

| Case | Peak state deviation | Final/peak | Final aligned retention | Transient half-life | Tail floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_lift_sloth` | 23.46 mm | 59.47% | 61.80% | 81.5 ms | 15.01 mm |
| `double_lift_sloth` | 24.80 mm | 9.54% | 1.39% | 45.0 ms | 2.39 mm |
| `double_stretch_sloth` | 9.12 mm | 37.59% | 7.18% | 32.3 ms | 3.57 mm |

The state perturbation is therefore not uniformly erased by one contractive
time constant. All cases have a short transient toward a nonzero empirical
floor, but `single_lift_sloth` retains a large component in the injected
direction while the other two rotate and largely lose it. This strengthens the
claim that a single state reset cannot reproduce persistent readout correction;
it does not establish that no state error existed.

### Mode and constraint resolution

A second trajectory-only audit rebuilds the frame-zero spring/controller graph,
requires an exact match to the frozen spring and basis hashes, and projects the
state difference onto the same four graph modes. It never reads future
residuals, manual tracks, or outcome metrics.

| Case | Nodes within 5 attachment hops | Final in rank-4 basis | Dominant retained mode | Far-graph directional retention |
| --- | ---: | ---: | ---: | ---: |
| `single_lift_sloth` | 7.01% | 88.65% | mode 0 (75.73%) | 49.34% |
| `double_lift_sloth` | 17.84% | 26.44% | modes 0/2, tiny absolute retention | 0.90% |
| `double_stretch_sloth` | 11.92% | 75.27% | mode 3 (84.86%) | 47.98% |

Across only three cases, attachment coverage has descriptive correlation
`-0.882` with absolute global directional retention and `-0.903` with
far-graph retention. This is consistent with stronger constraint coverage
suppressing an injected direction, but has no sampling interpretation.

The result does **not** support an inexpressibility explanation for single
lift. There, state injection captures 82.67% of the readout CD gain and 87.38%
of the readout track gain, and the surviving component is mostly the lowest
graph mode. State error is plausible for that interaction. In double lift, the
injected rank-4 field is largely contracted and 73.56% of the remaining state
difference lies outside the original basis. Double stretch instead shows
mode-3 retention and spatial sign cancellation: near, middle, and far
directional retentions are `+5.55%`, `-36.38%`, and `+47.98%`, respectively.

The frozen synthesis is therefore interaction-dependent contraction, rotation,
and mode transfer. Online state estimation remains justified for slow retained
modes, but one generic prefix reset is not a transferable discrepancy model.

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

Audit one already-frozen state rollout:

```bash
bpt-audit-phystwin-state-decay \
  /path/to/case/summary.json \
  /path/to/case/localization_rollouts.npz \
  /path/to/case/dynamic_discrepancy_correction.json \
  /path/to/case/state_correction_decay.json
```

Resolve the same trajectory by graph mode and distance from controller
attachments:

```bash
bpt-audit-phystwin-state-modes \
  /path/to/case/summary.json \
  /path/to/case/localization_rollouts.npz \
  /path/to/case/dynamic_discrepancy_correction.json \
  /path/to/case/dynamic_discrepancy_correction.npz \
  /path/to/case/final_data.pkl \
  /path/to/case/optimal_params.pkl \
  /path/to/case/state_correction_modes.json
```

## Claim boundary

A successful force branch means only that a constant low-rank force is a better
location for the released predictive correction than an output offset. It does
not identify friction, support, self-contact, or viscoelasticity. Mechanism
selection requires the locked same-object protocol with measured actuation,
registered support/contact geometry, reversals, rates, holds, and slip trials.
