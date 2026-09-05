# Fixed-mask partial-observation joint-belief test

Retrospective mechanism test on real public DEFORM DLO4/DLO5 trajectories. No new hardware, robot action selection, or adaptive observation selection. Only the canonical `DLO4/train` and `DLO5/train` directories are accessed. The official `eval` partitions and other reserved cohorts are not enumerated or opened.

## Question

At an identical sparse observation budget and a shared prior mean, does retaining a joint space-time residual belief improve hidden-region reconstruction and future free-node prediction? Does it beat independently calibrated empirical covariance and strong deterministic conditional controls, rather than just a deliberately diagonalized distribution?

This first mechanism test uses a source-fitted ridge correction of endpoint-transported damped-velocity geometry. **It is not a run of the released DEFORM physics/GCN hybrid or an existing BayesianPhysTwin checkpoint.** A positive result would support this specific conditioning mechanism; native-simulator benefit remains a separate question.

## Locked evaluation

Each object's 56 public training trajectories are hash-partitioned into 32 fit, 12 calibration, and 12 held-out test trajectories. All windows of a trajectory stay together. Four-fold source-only cross-fitting supplies residuals for covariance estimation. Full measured geometry is supplied at two initialization frames; eight frames later a fixed mask reveals 1, 2, or 4 of the eight internal nodes. Prediction uses the same recorded boundary positions for every arm. The three mask layouts are evenly spaced, a hidden middle block, and a hidden right block. With only eight nodes, some low-budget mask layouts coincide; each registered mask/budget cell retains its declared equal weight.

The primary metrics are hidden-current coordinate RMSE and all-free-node coordinate RMSE 16 frames after the sparse observation. The +4-frame metric is secondary. There are 20 windows per test trajectory, 9 mask/budget cells, 8 arms, and 34,560 scored rows. Rows/windows/coordinates are repeated measurements, not independent objects. Primary estimates equally average window RMSE within each mask and trajectory, then masks and trajectories. Paired bootstrap intervals resample 12 complete test trajectories within each DLO. No unseen-object inference is claimed from two objects.

The arms are prior/no correction; low-rank-plus-diagonal joint Gaussian conditioning; independent coordinates; sign-scrambled dependence with exactly preserved marginal variances; independently calibrated empirical shrinkage covariance; tuned spatial interpolation; mask-aware conditional ridge; and an independently solved deterministic MAP equivalent. The MAP equivalent must agree numerically and is not a superiority comparator. Source calibration selects rank/noise, empirical shrinkage/noise, interpolation strength, and conditional-ridge penalty. Noise is a likelihood/regularization floor, not added synthetic measurement noise. The low-rank model is reused across masks; mask-aware ridge has a separately fitted model for each mask.

The conjunctive positive criterion requires the upper paired-bootstrap endpoint below zero against prior, interpolation, empirical covariance, and masked ridge, for both primary metrics and separately for DLO4 and DLO5. Other positive or mixed mechanisms are reported without changing this criterion. Negative scientific results do not fail the workflow.

## Data and information boundary

The evaluator loads complete trusted local training pickle arrays but exposes only the allowed initialization, recorded boundary motion, and fixed visible-node vector to predictions. An adversarial test changes hidden and future free nodes and requires unchanged predictions. Predictions are sealed before metrics are computed, not before complete files are loaded. This is retrospective recorded-data replay, not byte-level sealed prospective acquisition. Measured coordinates are preserved without clipping. Imposed missingness is not validated real-camera occlusion. No physical-parameter, causal, or safety claim follows.

## Execution and evidence

A push changing `.github/requests/partial-observation-belief-v1.json` on `science/partial-observation-belief-v1` triggers the workflow. No workflow dispatch is used. The request binds the protocol SHA-256; the data job uses `[self-hosted, gpuserver4090]`. Input files are read without mutation or duplication. The isolated NumPy runtime and compact output live under the runner temporary directory.

The artifact contains input hashes and full splits, calibration choices made before each object's test predictions, all predictions and marginal variances, row metadata, all per-case errors and available marginal uncertainty metrics, paired trajectory-bootstrap contrasts, by-mask results, a prediction digest, and a Markdown summary. Raw trajectories are not published. Eight synthetic software tests plus a local synthetic end-to-end rehearsal check the code; neither is real-data evidence.

Run locally on an authorized dataset view:

```sh
OPENBLAS_NUM_THREADS=1 python experiments/partial_observation_belief_v1/run.py \
  --dataset-root /mnt/seagate10tb/florianpfaff/datasets/deform/data_set \
  --output /path/to/new-output-directory
```
