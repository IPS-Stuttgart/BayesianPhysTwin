# Real-data result: a modest GP gain over the existing ridge residual

**The offline comparison completed successfully.** On the historical DLO2
validation panel, the fixed nonlinear GP improves mean coordinate L1 over the
existing local ridge residual by **1.6317%**, with **7/8 trajectory wins**.

Executed on `workstation1` / `gpuserver4090` in
[Actions run 34205498883](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34205498883),
using execution commit `51698c190d749365c81cbdfc96a7085db83093dc`.
The numerical/boundary test step and empirical comparison both passed. The
experiment itself took 158.60 seconds. No robot experiment or new data was used.

## Accuracy

Metric: equal-trajectory mean absolute coordinate error over all 498 future
steps and all 12 nodes, in millimetres. All methods receive the same two initial
states and known future clamped-node motion.

| Method | Mean coordinate L1 (mm) | Improvement over DEFORM hybrid |
| --- | ---: | ---: |
| Frozen DEFORM physics-plus-GCN hybrid | 7.912029 | reference |
| Existing local ridge residual, shrinkage 0.25 | 7.240300 | 8.4900% |
| Fixed nonlinear GP residual, shrinkage 0.25 | **7.122158** | **9.9832%** |

The incremental GP improvement is **0.118142 mm**, or **1.6317%**, over ridge.
The GP wins **7/8** trajectories against ridge and **8/8** against the hybrid.
Its one loss to ridge is `106.pkl`: `5.425760` versus `5.417606 mm`, a difference
of only `+0.008154 mm`. This loss is retained, not filtered out.

The experiment's descriptive 95% paired trajectory-cluster bootstrap interval
for GP minus ridge is **[-0.184397, -0.055848] mm**, using 5,000 resamples.
There are eight distinct validation causal queries. This is a small historical
validation panel on **one physical DLO2 object**, not eight independent objects.

All eight trajectory outcomes are in
[`results/trajectory_metrics.csv`](results/trajectory_metrics.csv). The
machine-readable result and artifact identities are in
[`results/summary.json`](results/summary.json).

## What was tested

The same historical 40 fitting trajectories and eight validation trajectories
were used. The checkpoint was not retrained. Both corrections have ridge 1 and
shrinkage 0.25. The GP adds a fixed Matérn-3/2 Nyström block to the original
linear feature block: 128 trajectory-balanced anchors per node, amplitude 3,
length equal to sqrt(92) after feature standardization, seed 20260907, and
kernel jitter 1e-9. All fitting targets remain in the regression; the anchor
approximation reduces the basis only.

The GP configuration was fixed before this validation comparison and was not
tuned to its outcomes. The original checkpoint and ridge recipe had already
been selected using historical development data; this is **not a new blind
confirmation panel**. It tests the nonlinear mean correction, not whether
Bayesian uncertainty itself improves accuracy.

## Integrity checks

The zero-nonlinearity GP and the actual repository ridge predictor agree to
**1.3600e-15 m maximum absolute difference** on these real validation queries.
All clamped-node coordinates are bitwise identical to the supplied baseline.
The historical baseline metric is reproduced. The exact 40 fitting and eight
validation file paths account for every recorded trajectory read; no historical
source-test or official-evaluation trajectory was opened.

After downloading the artifact, a separate local check verified its SHA256,
linked model and prediction hashes, all eight trajectory aggregates, all 498
horizon aggregates, prediction names/shapes/finiteness, exact clamped-node
preservation, and the allowed-file list. These are independent compact-artifact
checks, **not** an independent raw-data or simulator rerun.

Artifact: `10047661130`, named `deform-gp-development-34205498883-1`.
Archive SHA256:
`86a08e983f90e626684bd28a5c1fc9c798836701d5b49d289ba2ce0e2de298be`.
Original report SHA256:
`776027d03f27aef275c241d896daa0758285f407697ef3088f29b4e114ccabda`.
The full artifact includes model arrays, validation predictions, per-trajectory
and per-horizon CSVs, preflight metadata, the fit seal, and the original report.

## Interpretation

This is a **positive first real-data accuracy result**: the GP provides a small,
consistent improvement beyond the already useful ridge correction. It supports
continuing this residual-model direction. It does not by itself establish a
standalone new methodological contribution, cross-object generalization,
state of the art, calibrated uncertainty, or uniquely Bayesian superiority.
Graph coupling was not tested in this trial.

The earlier execution attempts `34204163720` and `34204714437` stopped at
runtime-path checks before data reads. They are retained technical failures, not
negative scientific results. Metadata/schema integration corrections preceded
the completed trial; the GP settings and the 40/8 scientific split did not
change. No parent protocol, official evidence, or main-branch code was modified.
