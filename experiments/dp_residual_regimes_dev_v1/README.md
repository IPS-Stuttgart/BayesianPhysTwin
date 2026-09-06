# Conditional DP residual-regime development screen

Retrospective development on the sixteen previously opened DLO4/DLO5 source-test trajectories (eight per object). This is not a fresh confirmation and does not reopen any official evaluation trajectory.

The experiment models the remaining error of the frozen Bayesian local-correction candidate. It is a low-rank, blockwise conditional Gaussian-mixture readout, not a fully Bayesian switching dynamics model or a change to latent physical state. Each six-frame block shares one component across all eight free nodes. Components use a joint input/output density; prediction conditions on inputs only. DP inference is truncated variational inference with at most eight components. Mixture means/covariances are plug-in estimates, not exact posterior-predictive distributions. Correlated blocks are density-fitting units, not independent evaluation replicates.

## Matched controls

All extra models use the same fit-only input/output PCA, baseline outputs, and trajectory split. Compare ridge, a single conditional Gaussian, finite Gaussian mixtures with 2/4/8 components, finite Bayesian mixtures with eight components, and DP mixtures with maximum eight components. DP concentration and finite-prior total concentration each use 0.1/1/10. Correction shrinkage uses 0/0.25/0.5/1. All selections use inner validation trajectories only; zero returns the original candidate unchanged. Nonconverged models are retained in logs but cannot supply selected nonzero corrections.

Four outer folds keep complete trajectories and exact duplicate causal queries together. Each fold uses eight trajectories for fit, four for validation, and four for test. World-coordinate marginal medians are used for coordinate-L1; mean and hard-assignment diagnostics are retained separately. No action selection, extra response prefix, new data collection, or robot is involved.

## Execution

With the checksum-verified source export from Actions run 34052556765, artifact 9994987518:

```bash
python -m pip install numpy==2.2.6 scipy==1.15.3 scikit-learn==1.8.0 pytest==8.4.2
python -m pytest -q experiments/dp_residual_regimes_dev_v1/test_model.py
python -m experiments.dp_residual_regimes_dev_v1.run \
  --cache-dir /path/to/extracted/pinned-source-cache \
  --output outputs/dp-residual-regimes-dev-v1
```

On gpuserver4090, `--source-root` can instead point to the original source-only cache. That route verifies the pinned manifests and prediction archives, then reads only their exact named train trajectories. The portable cache route needs no simulator, torch, GPU, or raw dataset pickle access.

The runner retains protocol/input/code hashes, all validation curves and convergence records, per-trajectory scores, and complete selected-model metadata. The bootstrap intervals are descriptive trajectory intervals on two objects with overlapping training folds; they do not establish a population-level or independent-object confidence claim.

## Initial local result

The first execution used Python 3.11.8, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. Exact runtime versions are authoritative in its result.json. An independent Actions reproduction uses the workflow-pinned runtime and must be reported separately.

| Method | DLO4 coordinate L1, mm | DLO5 coordinate L1, mm | Equal-case mean, mm |
|---|---:|---:|---:|
| DEFORM hybrid baseline | 12.530179 | 9.183285 | 10.856732 |
| Frozen existing local correction | 11.524244 | 8.917156 | 10.220700 |
| Additional matched ridge | 11.397038 | 8.862328 | 10.129683 |
| Additional single conditional Gaussian | 11.393614 | 8.861541 | 10.127578 |
| Additional finite GMM | 11.433780 | 8.965846 | 10.199813 |
| Additional finite Bayesian mixture | 11.476424 | 8.921833 | 10.199128 |
| Additional DP mixture | 11.473057 | 8.915413 | 10.194235 |

DP improves the existing correction by only 0.2589% and is worse than the simpler single Gaussian. Against the existing correction it records seven wins, five losses and four exact-fallback ties. The descriptive 95% paired trajectory bootstrap interval for DP-minus-existing mean error is [-0.100686, +0.048618] mm. The unshrunk selected DP models reach 11.341041 mm, worse than the unchanged candidate. Active component counts are 8/7/7/8, and selected shrinkage is 0.25/0/0.25/0.25 across the four folds.

This screen does not provide convincing DP-specific added value. It does not reject all DP residual formulations, but does not justify a DP-centered paper claim. All original models, official benchmark evidence and main-branch scientific claims remain unchanged.
