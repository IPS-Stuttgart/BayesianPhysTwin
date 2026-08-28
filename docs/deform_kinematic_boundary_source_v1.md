# Frozen hard-position boundary source screen

## Question

Does a different treatment of prescribed boundary positions improve the
existing sparse DEFORM state update, without changing learned weights or the
observation budget? This is a new source-only physical-model ablation, not a
retry or reinterpretation of the failed reference-centering experiment.

The preceding code/control-only diagnostic finds that native DEFORM preserves
the direction of each prescribed end segment while projecting its length to
the rest length. In the fourteen opened DLO2 trajectories, node 1 moves by up
to 11.297 mm in Euclidean distance; its componentwise maximum is the previously
reported 10.869 mm. The first point of each prescribed pair stays fixed.
This does not establish that native behavior is wrong: measured pair lengths
may be noisy, and the projection can be a useful modeling assumption.

The alternative explicitly honors all four supplied positions. The only
projection change skips an edge when both endpoints are prescribed. Every
other native edge update, edge order, iteration count, force, learned weight,
twist/material memory, and time step is unchanged. Prescribed segments may
therefore differ from their rest lengths. We make no claim that this is a
globally inextensible rod or a more correct material/actuator model.

The opt-in replacement is installed on a new model instance only, after its
unmodified native rest-state initialization. Derived rest curvatures and
masses must reproduce the old model exactly. Upstream
files/classes, existing backends, successful predictions, and prior receipts
remain unchanged. Disabling the replacement preserves the original callable;
disabling the corresponding readout returns the original incumbent object.

## Four fixed arms

Let B be the frozen incumbent and A its archived native component. Let H be
the hard-boundary native trajectory. There is no parameter fitting.

1. Unchanged incumbent B.
2. Existing paired update, reproduced byte-for-byte from its frozen archive.
3. Hard-boundary baseline C = H + (B - A).
4. Hard-boundary paired update: C + H_updated - H, the sole primary.

The fixed learned readout offset B-A is not refitted or re-centered. Its
boundary entries must be exactly zero. The new sparse pose/velocity innovation
is measured against C in the permitted prefix, then injected into H's own
endpoint state. Native velocity, twist, material frame and previous positions
are retained. The diagnostic baseline cannot rescue a failed primary. A
backend gain without sparse-update value is not a Bayesian-update result.

## Inputs and order

Use only the fourteen already-open DLO2 trajectories; exclude the existing
design case 103.pkl from all metric aggregates. Every updated arm receives
the same two initial full states and eight 3D prefix measurements: nodes
2,4,6,8 at archive frames 41,49. The hidden scored nodes remain 3,5,7,9.
Forecast frames are 50:170; early/middle/late bins each contain forty frames.
Raw dataset frame 2 is archive frame 0. Known future end-node trajectories
are permitted controls under this released-data contract, not a claim of
independently measured real robot commands. All other future truth is excluded
from prediction. Whole already-open source containers can be decoded, but
only the permitted slices enter the simulator/inference functions.

Freeze clean source hashes, exact inputs/runtime, and this plan before one
native CPU attempt at the registered fresh root. Retain every technical
failure without retry or replacement. Seal all fourteen predictions and all
controls before source future truth is scored. A source-only second arithmetic
implementation validates custody, readout, sparse increments, clamp equality,
metrics, and the decision; it is not independent human review.

## Controls and gate

The incumbent and old paired output must be byte-identical. The patched native
endpoint restart must exactly reproduce its own unperturbed continuation.
Both new native branches must preserve the float32 supplied clamp positions
exactly. Zero innovation must return the unchanged hard baseline. The legacy
native/archive replay remains within the already registered 2 mm maximum
coordinate difference and 0.2 mm coordinate RMSE tolerances. All state arrays
must stay finite; no failed trajectory is dropped.

The primary uses the preceding source screen's six value thresholds:

- At least 2% lower equal-trajectory mean coordinate L1 and point RMSE than
  the existing paired update.
- Non-increasing mean late RMSE relative to that update.
- At least 8/13 joint L1/RMSE wins.
- Worst trajectory RMSE ratio at most 1.05.
- Paired trajectory-bootstrap RMSE-difference 95% upper bound below zero
  (10,000 draws, seed 260929, lexicographic non-design order).

Additionally, it must improve both mean L1 and RMSE over the hard-boundary
baseline without sparse observations. All controls and all seven value checks
must pass. Report all arms/horizons regardless of the decision. Intervals are
conditional on one already-open object, not independent confirmation.

This screen authorizes no DLO1/DLO3 transfer, DLO4/DLO5, official DLO3 evaluation,
reserved/fresh Deform360, PokeFlex continuation, held-v8, physical Causal4D,
GPU, new recording, push, or main merge. A source pass motivates a separately
registered next experiment; it does not establish UQ, SOTA, counterfactual
identification, or a significant paper contribution by itself.
