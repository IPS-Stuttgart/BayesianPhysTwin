# Frozen DEFORM DLO4/DLO5 transfer

This experiment applies the source-fitting and Bayesian local-residual recipe
frozen by the completed DLO3 study to the official DEFORM DLO4 and DLO5
train/evaluation operators.

The experiment is a fresh **procedure-level replication**, not zero-shot
object generalization. Each DLO uses its own official initialization and its
own train partition. The method family, seed, physical training budget,
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

## Primary comparison

The registered primary metric is the DEFORM-style mean coordinate L1 over all
nodes and all forecast frames. The candidate is compared with the identically
trained update-6400 physical checkpoint. Free-node-only errors, the
compute-matched physical continuation, action-aware persistence, and seven
point-identical covariance arms are reported as secondary diagnostics.

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

A positive result would support frozen-procedure replication across the exact
released DLO4 and DLO5 operators. It would not establish arbitrary deformable
object generalization, camera-based Prob4D competence, causal intervention
identification, robotic control safety, or universal state of the art.
