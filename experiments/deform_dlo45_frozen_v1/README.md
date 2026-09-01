# Frozen DEFORM DLO4/DLO5 transfer

This experiment applies the source-fitting and Bayesian local-residual recipe
frozen by the completed DLO3 study to the official DEFORM DLO4 and DLO5
train/evaluation operators.

The experiment is a fresh **procedure-level replication**, not zero-shot
object generalization. Each DLO uses its own official initialization and its
own train partition. The method family, seed, hybrid DEFORM training budget,
residual operator, shrinkage, covariance construction, source gate, target
metrics, and no-retry policy are fixed before either evaluation partition is
read.

## Information order

1. `inventory` checks the public rosters, frozen upstream initialization, CUDA
   runtime, and one source-only training update. It does not deserialize an
   evaluation trajectory.
2. `source` partitions each 56-trajectory train set into 39 fit, 9 covariance
   calibration, and 8 source-test trajectories. The method seal is written
   before the source-test trajectories are opened.
3. `authorize` requires both DLO source gates to pass.
4. `predict` refits the unchanged procedure on all 56 train trajectories,
   writes the final method seal, then predicts all 14 evaluation trajectories.
5. `seal` binds both prediction archives into one joint seal.
6. The workflow uploads that joint seal before `score` opens either target
   outcome for metric computation.

Any execution failure after evaluation payload access is terminal for this
request. Target replacement, target-side calibration, target-side model
selection, and workflow retry are forbidden.

## Primary comparison and baseline semantics

The registered primary metric is the DEFORM-style mean coordinate L1 over all
nodes and all forecast frames. The candidate is compared with the identically
trained update-6400 **DEFORM hybrid checkpoint**. This is not a bare-physics
baseline: the public `DEFORM_sim` combines differentiable rod dynamics with
vertex and displacement GCNs plus a residual decoder, and the source-closed
optimizer trains all five learned modules (`vert_conv1`, `vert_conv2`,
`delta_vert_conv1`, `delta_vert_conv2`, and `fc`) together with the physical
parameters. Recursive evaluation calls the complete model in evaluation mode,
so DEFORM's learned residual remains active.

The immutable result contract uses the historical internal key
`matching-update-6400-physical-checkpoint`. Reader-facing reports must describe
that arm as a **source-closed retraining of the released DEFORM hybrid
architecture and optimizer**. The public repository contains training code but
no authors-released pretrained DLO4/DLO5 checkpoint, so this is not claimed to
be a byte-identical reproduction of the authors' original training run.
BayesianPhysTwin adds a second, frozen Bayesian local-residual layer to this
already learned hybrid.

Free-node-only errors, compute-matched continuation of the same complete DEFORM
hybrid, action-aware persistence, and seven point-identical covariance arms are
reported as secondary diagnostics. A separately retrained physics-only ablation
would answer a different decomposition question and is not part of the frozen
DLO4/DLO5 result.

## Runner

The maintained workflow uses:

```yaml
runs-on: [self-hosted, Linux, X64, gpuserver4090]
```

with the data root fixed to:

```text
/mnt/seagate10tb/florianpfaff/datasets/deform/data_set
```

Original evaluation checkpoints, models, and prediction arrays are retained
privately on the runner under
`/home/github-runner/.cache/workflows/deform-dlo45/runs/`. Only compact evidence,
seals, and logs are uploaded as GitHub Actions artifacts.

## Claim boundary

A positive result supports frozen-procedure replication across the exact
released DLO4 and DLO5 operators. It establishes incremental value beyond a
source-closed retraining of DEFORM's own physics-plus-GCN predictor, not beyond
bare rod physics. It does not establish arbitrary deformable-object
generalization, use of the authors' pretrained checkpoint, camera-based Prob4D
competence, causal intervention identification, robotic control safety, or
universal state of the art.
