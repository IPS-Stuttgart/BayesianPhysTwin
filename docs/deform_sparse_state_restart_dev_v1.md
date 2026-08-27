# DEFORM Sparse State Restart: Opened-Data Development Test

## Question

The predictive-coupling experiment did not establish a useful active-sensing
gain. Test a different mechanism: sparse prefix positions correct the actual
DEFORM position/velocity state, whose influence is propagated through the fixed
rod simulator and its existing learned dynamics. This is not another fit to the
readout covariance and is not a new official evaluation.

The original successful DLO2 checkpoint, local-residual readout, and all frozen
results remain unchanged. No DLO4/DLO5, held-v8, fresh Deform360 target, or new
physical acquisition is used.

## Fixed Inputs and Information

The config binds the already-open 14-trajectory DLO2 archive, physical checkpoint,
and upstream revision. The first lexicographic trajectory, `103.pkl`, remains a
design/capability case and is excluded from the 13-trajectory aggregate. All
trajectories belong to one physical object; trajectory intervals do not establish
cross-object transfer.

Archive indices 0 onward correspond to dataset frames 2 onward. Four material
identities (2, 4, 6, 8) are observed at archive frames 41 and 49: eight 3D point
observations, with no later non-clamped measurements. The forecast covers archive
frames 50 through 169. Score only disjoint identities 3, 5, 7, 9. Future clamped
positions are known actuator inputs, as in the original DEFORM contract.

Sparse residuals are linearly interpolated along material-node index with zero
updates at the four clamped nodes. The endpoint position residual gives the pose
increment. Its slope over the fixed 80 ms interval gives a velocity increment
added to the simulator's instantaneous velocity; it does not replace that
velocity with an eight-frame average. There is no future tuning or learned
interpolator.

## Comparisons

- Unchanged frozen incumbent and unchanged physical DEFORM.
- Matched sparse readout persistence using the same eight observations.
- Physical state updated with sparse pose, or sparse pose plus residual velocity.
- Privileged full-free-node prefix pose and pose/velocity reference controls.
  These are not sparse-sensing methods and are not an absolute oracle over all
  possible state estimators or internal material states.
- Incumbent plus the paired physical response to its remaining sparse residual,
  with pose, pose/velocity, and a fixed quarter-gain pose/velocity arm. This keeps
  the frozen readout unchanged and avoids counting its correction twice. It is
  an empirical hybrid, not a newly calibrated posterior.

All arms are reported. No best-arm result becomes a confirmation claim. The
quarter-gain arm is fixed before execution, not selected after viewing outcomes.

## State and Runtime Controls

Preserve positions, returned velocity, previous positions, transported material
frame, and rod twist. Reuse the same rest/material tensors and checkpoint. Do not
reset twist or reconstruct an unrelated material frame at the prefix boundary.

Use CPU in the original Torch 2.0.1 runtime because the GPUs are occupied. This
is a separately declared development runtime, not an assertion of GPU bitwise
parity. Before scoring, require:

1. Adapter agreement within 1 micrometre with the original uninterrupted rollout
   on the same CPU runtime.
2. Archived GPU replay agreement within 2 mm maximum and 0.2 mm coordinate RMSE.
3. Byte-identical zero-update continuation and incumbent fallback.
4. At least 99% recovery of a fixed synthetic 1 mm pose / 10 mm/s velocity state
   perturbation when the true state is supplied. This validates the injection and
   recovery path, not real-data effectiveness.

A failed runtime/control gate is retained and prevents scientific scoring. The
runner accepts only the fixed opened-data roster, verifies raw trajectory hashes,
and compares their saved suffix to the already-open archive. It never runs the
old one-shot official evaluator or changes its records.

## Evaluation and Claims

Seal every arm's predictions before computing metrics. Report hidden-identity
coordinate L1, Euclidean point RMSE, FDE, and equal-length early/middle/late bands.
Use equal-trajectory means and 10,000 paired trajectory bootstrap draws with seed
260829. Report the design case separately and all 13 remaining trajectories, with
no replacement. No uncertainty-calibration claim follows from these deterministic
updates. Any genuine gain would justify a later separately frozen sensing/state
study; failure would close this simple position/velocity update class only.
