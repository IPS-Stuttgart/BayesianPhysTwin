# DEFORM GP residual pilot v1

This is an exploratory comparison of residual predictive means, requested as a
concrete GP trial without new robot experiments. It does not modify the stable
package, official evaluation protocols, published results, or claim registries.

## Comparison

All methods receive the same frozen DEFORM **hybrid** checkpoint (differentiable
rod plus its trained GCN correction), two observed initial states, prescribed
future boundary motion, and the existing causal local feature construction.
Free-node future positions are labels, never prediction features.

The original DLO2 training-side split supplies 40 fitting, eight validation, and
eight historical source-test recordings, each with 498 predicted steps. Whole
recordings remain grouped; exact duplicate fit queries are collapsed and
cross-partition exact-query overlap is rejected. These historically studied
partitions are exploratory, not fresh independent confirmation.

The families are the uncorrected hybrid, existing ridge=1/shrinkage=0.25, a
validation-selected ridge comparator, nodewise sparse Matern-5/2 GPs, and a shared
Matern-5/2 dynamics times material-coordinate RBF GP. Hyperparameters and the GP
family are selected on validation before source-test export. All candidate
validation scores are retained. No official `eval` trajectory is opened.

`gp.py` analytically solves the Gaussian-likelihood sparse variational posterior
for fixed hyperparameters. The three local coordinates are conditionally
independent output GPs. All fit labels enter sufficient statistics; 128 inducing
inputs per node or 256 shared inducing inputs are selected from fit inputs only.
The unequal inducing capacities and 27/54 GP versus nine ridge candidate settings
are explicit pilot limitations, not a compute-matched architecture study.
The first pilot scores means only: neither calibration, joint probabilistic value,
physical parameter recovery, nor a uniquely Bayesian benefit is claimed.

## Reproduction

The file-change-triggered workflow `.github/workflows/deform-gp-residual-pilot-v1.yml`
runs on `[self-hosted, gpuserver6000]`. Its request file selects export-development
or gp-comparison; official evaluation and dataset mutation must remain false.
The old environment's Python 3.10 packages are used with an explicit Python 3.10
interpreter, avoiding mutation of the legacy virtual environment.

The export verifies the source manifest and checkpoint SHA-256 identities and
requires validation L1 agreement with the historical hybrid within 1 micrometre.
The observed export disagreement was 2.33e-10 metres. Verified fit/validation
prediction caches live at `/home/github-runner/.cache/bpt-gp-residual-pilot-v1`.
This new prediction cache does not modify the dataset or old checkpoints.

Numerical controls in `gp.py` check exact-GP mean/covariance parity when inducing
inputs equal training inputs, positive covariance, a zero-target case, the
material product kernel, and invalid-noise rejection. These controls are software
tests, not real-data accuracy results.

The workflow retains `result.json`, per-recording errors, selected settings,
`selection_before_source_test.json`, the complete validation table, prediction
arrays, source identities, and `REPORT.md`. Final paper-facing interpretation
belongs in the paper repository rather than a release claim update here.
