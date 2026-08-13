# MatPhys part-aware reconstruction control

## Scope

This is a deliberately noncausal capacity test on the already-open
`single_lift_sloth` case. It fits all released frames, including the future
evaluation interval. It cannot support a predictive or state-of-the-art claim.

The method is not the published MatPhys method. MatPhys does not release its
final per-case semantic bundle, so Bayesian-PhysTwin supplies a deterministic
DINO graph partition and a new part-conditioning adapter. The adapter projects
each part descriptor into the released decoder's material embedding. Both the
proxy bytes and adapter implementation are bound by the training audit.

## Pre-execution correction

The earlier `matphys-official-reconstruction-control-v1` protocol loaded the
graph-part proxy but failed to install the adapter. Because all parts of this
case share one material row, that runner would have ignored the DINO
descriptors and duplicated the completed one-part public-artifact control. No
training or outcome was produced under that protocol.

The replacement protocol is
`configs/sota/matphys_part_aware_reconstruction_control_v1.json`. It requires
the `simple-videomae-dino-part-conditioning-v1` adapter at positive scale in
both training and export. Validation rejects a checkpoint whose audit does not
bind that contract.

A second source-independent preflight found that the staged byte-bound proxy
uses the registered compact edge-semantics contract. Version 1.1 names that
contract explicitly. Its `train_ready.pt`, 1024-dimensional part descriptors,
assignments, material distributions, and graph provenance are inherited from
the full proxy; only node/edge semantic tensors unused by the simple decoder
are compacted. No training or outcome was produced under version 1.

## Decision

Run the separately locked
`configs/sota/matphys_part_aware_reconstruction_smoke_v1.json` one-epoch
protocol first to verify finite optimization, full future-access labeling,
strict checkpoint export, and official metric evaluation. Continue to the
fixed 200-epoch terminal checkpoint only if the smoke passes mechanically.

A positive reconstruction result would show capacity only. A later predictive
experiment would need a separately frozen source-only training design, target
prefix selection, pinned-PhysTwin replay parity, and an independent cohort.
