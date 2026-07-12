# Hierarchical structural calibration

Status: synthetic mechanism recovery passes; released-data Warp evidence is
diagnostic only; confirmatory same-object acquisition has not started.

## Model boundary

The opt-in structural path separates:

- object-persistent rest geometry in a low-rank graph basis;
- session frame, placement, settled-state, and gravity corrections;
- intervention variables retained by Causal4D;
- dynamic graph discrepancy retained as a residual baseline.

For graph basis `U` and persistent MAP coefficients `a`, corrected material
positions are `r* = r0 + U a`. Object spring rest lengths are recomputed from
this embedded geometry. They are never fitted independently and never clipped
edge by edge. A candidate is rejected when its edge strain, support anchors,
validity cells, or supplied surface topology violate the locked limits.

`StructuralTwinCorrection` is a typed JSON/NPZ artifact containing the nominal
geometry hash, vector graph basis and frequencies, persistent coefficients,
optional deferred covariance, session corrections, support model, plausibility
limits, fit-session identities, information boundary, and source checksums.
`StructuralWarpConfiguration` records the exact corrected positions, rest
lengths, initial state, controls, and gravity vector passed to Warp.

The tagged `v0.3.0-causal4d-aip` path is not modified. The structural adapter is
opt-in and delegates to the same simulator restart functions. Identity artifacts
must preserve every backend input and the resulting deterministic Warp
trajectory byte for byte.

## Basis and MAP fit

The candidate ranks are fixed to `4`, `8`, and `16`. A symmetric normalized
spring Laplacian supplies low-frequency scalar modes. Their vector fields are
projected out of the six rigid translation/infinitesimal-rotation modes and are
exactly zero on declared support nodes.

The MAP design uses separate simulator sensitivities for persistent rest
geometry, session settled state, frame rotation/translation, and gravity. It
applies frequency-weighted priors, Huber measurement weights, and a relative
edge-strain penalty. Invalid/missing observations have zero validation weight.
Rank and mechanism use a one-standard-error rule on source `O-minus` validation
frames, preferring fewer parameters and lower rank. Coefficient covariance is
not estimated until the mean correction passes the transfer gate.

## Synthetic gate

Run:

```bash
bpt-structural-recovery-benchmark runs/structural-recovery/summary.json
```

The controlled benchmark includes frame, gravity, graph-smooth rest geometry,
initial state, combined rest/state/frame/gravity, and omitted-physics cases. It
also checks support anchoring, strain/geometry validity, Causal4D-style contact
hypothesis updating, byte-identical identity application, and invariance to
withheld-future mutation. All recoverable families currently select their
intended mechanism at rank 4; the omitted-physics case is rejected.

## Official Warp diagnostic

Run one released case with:

```bash
bpt-diagnose-phystwin-structure \
  /path/to/PhysTwin /path/to/final_data.pkl /path/to/inference.pkl \
  /path/to/optimal_params.pkl /path/to/checkpoint.pth \
  /path/to/gt_track_3d.pkl runs/structural/CASE \
  --train-end-frame FRAME \
  --graph-persistence-archive /path/to/rest_geometry_injection.npz
```

The runner first verifies identity parity. It then settles the nominal graph
under gravity and fixed initial controls. Sixteen rest sensitivities each
recompute geometric rest lengths and re-settle before the action. Sixteen state
sensitivities perturb the settled state without changing material rest lengths.
Six physical frame sensitivities transform state and controls before rerunning
Warp. Only tracked points selected from `O-minus` enter the MAP system. Every
selected ladder variant is subsequently rerun nonlinearly in official Warp.

The released sloth interaction has been inspected repeatedly and cannot lock
the final rank or priors. It is a mechanism diagnostic. Gravity is represented
in the typed artifact and synthetic benchmark, but the pinned official kernel
hard-codes gravity; a nonzero real gravity correction therefore fails closed
until an opt-in structural simulator kernel exposes it. Released point clouds
also lack trusted surface faces, so their diagnostic validity check covers
anchors and edge strain, not a confirmatory self-intersection claim.

The official `667`-substep diagnostic is complete on the three released sloth
interactions. The source-`O-minus` one-standard-error selector chooses baseline
and rank 4 in all three cases. Equal-case future changes versus released
PhysTwin are:

| Method | CD | Track | Late track |
| --- | ---: | ---: | ---: |
| Graph-persistence readout | -10.59% | -13.76% | -11.48% |
| Initial state only | +0.09% | -2.26% | -7.01% |
| Rest geometry only | +0.18% | +1.38% | -1.73% |
| Rest + state | +0.83% | -0.28% | -1.62% |
| Full hierarchy | +2.64% | +0.54% | +1.15% |

The selected physical methods have `1.147x` graph-persistence track error,
`1.060x` late track error, and `1.251x` far-graph observation error. Coverage
cannot be evaluated from these deterministic released runs. The candidate
therefore fails all locked comparison gates and is not promoted. The useful
mechanism clue is that initial state can improve late manual-track error while
leaving CD neutral, whereas persistent rest geometry does not improve both
metrics. A missing support/contact, topology, friction, self-contact, or
material-dynamics mechanism remains more plausible than this low-rank
equilibrium correction on the available interactions.

Aggregate completed diagnostics with:

```bash
bpt-aggregate-phystwin-structure runs/structural-aggregate.json \
  runs/CASE_A/summary.json runs/CASE_B/summary.json runs/CASE_C/summary.json
```

## Acquisition amendment

Install or audit the measurement-only amendment before acquisition:

```bash
bpt-structural-protocol protocol.json DATASET_ROOT
bpt-structural-protocol protocol.json DATASET_ROOT --audit-only
```

The amendment hashes the unchanged actions, conditions, executions, outcomes,
and splits. It adds three static seconds after settling, support geometry,
gravity/world calibration, reset metadata, a post-reset static scan before each
action, and structural slip-pilot questions. It preregisters transfer,
late-horizon, far-graph, stability, plausibility, and later calibration gates.
No covariance calibration or robot execution is allowed before those gates.
