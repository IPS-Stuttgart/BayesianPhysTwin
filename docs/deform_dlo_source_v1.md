# DEFORM DLO Source Reproduction v1

## Purpose

This protocol tests whether the public DEFORM implementation can be reproduced
before Bayesian-PhysTwin uses it as an external physical prior. DEFORM is not
vendored or reimplemented. The runner imports an independently obtained checkout
at commit `b73b8b8ecc033caefa693fab7898741d4e6dbeff`.

The source stage opens only the 56 official `train` trajectories for `DLO1`.
The official `eval` trajectories are unavailable to training, checkpoint
selection, calibration, and source metrics. A previous outcome-blind schema and
range audit did deserialize the public eval arrays, so this is not described as
pristine dataset custody; no prediction or error was computed from them.

## Frozen Source Split

Trajectory filenames are ordered by a seeded SHA-256 mapping and divided into:

- 40 fit trajectories;
- 8 validation trajectories;
- 8 held-out source-test trajectories.

The fit partition determines gradients. Validation selects among checkpoints at
updates 0, 40, 80, 160, and 280. The source-test trajectories are evaluated once
after checkpoint selection.

## Runtime Contract

Training uses the public DEFORM model and simulator, its four clamped endpoint
nodes, its SGD parameter groups, a batch size of 32, and known future clamped-node
motion. The unroll is 50 frames. This follows the upstream author's current
stability guidance after a report of NaNs around checkpoint 400. A compatibility
shim satisfies the unused `sksparse.cholmod` import; the selected Theseus path
must remain the dense Cholesky solver and the shim raises if called. CUDA runs
bind `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing PyTorch.

The upstream repository has no explicit license file at the locked commit.
Consequently this repository records provenance and orchestration only and does
not redistribute DEFORM code, data, or checkpoints.

## Advancement Gate

The selected checkpoint advances only when all conditions hold on the eight
held-out source trajectories:

1. mean coordinate-wise L1 is at most 1.1 times the published DLO1 error
   (11.11 mm);
2. it beats exact coordinate persistence in at least six of eight trajectories;
3. all inputs, split assignments, checkpoints, and outputs are hash-bound.

Failure closes this reproduction route without opening official evaluation
metrics. Success permits a separately locked all-train reproduction and
identical-information official evaluation.

Any later online prefix assimilation is a different observation contract. It
must be reported separately and cannot be called an identical-information SOTA
comparison.
