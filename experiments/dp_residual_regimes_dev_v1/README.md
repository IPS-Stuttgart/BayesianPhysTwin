# Conditional DP residual-regime development screen

**Completed result: no convincing DP-specific added value.** This is a retrospective screen on sixteen already-opened DLO4/DLO5 source-test trajectories, not fresh confirmation. No official evaluation trajectory was opened, no original model was retrained or overwritten, and no new physical recording or robot action was required.

## Matched experiment

The prototype models remaining error of the frozen local-correction candidate. It is a low-rank, blockwise conditional Gaussian-mixture readout, not a fully Bayesian switching physical-dynamics model. Each six-frame segment shares a component across all eight free nodes. A joint input/output density is conditioned on inputs only at prediction time. The DP uses truncated variational inference with at most eight components and plug-in component distributions, not an exact posterior predictive. Correlated blocks are density-fitting units, not independent evaluation replicates.

All additional models use the same baseline, features, fit-only scaling/PCA, and grouped split. Controls are ridge, one conditional Gaussian, finite GMMs with 2/4/8 components, finite Bayesian mixtures with eight components, and DP mixtures capped at eight. Both Bayesian families use total concentration candidates 0.1/1/10. Every family selects correction shrinkage from 0/0.25/0.5/1 on validation trajectories only; zero preserves the original candidate exactly. Nonconverged models cannot supply selected nonzero corrections.

Four outer folds each use eight trajectories for fit, four for validation, and four for test. Complete trajectories and exact duplicate causal queries stay together. Inputs are the existing initial-state, recorded clamped-action and baseline-rollout features; no future free-node truth is supplied to prediction. World-coordinate marginal medians are scored by coordinate-L1. The bootstrap is descriptive on sixteen trajectories from only two objects with overlapping training folds, not an independent-object confidence interval.

## Results

| Method | DLO4 coordinate L1, mm | DLO5 coordinate L1, mm | Equal-case mean, mm |
|---|---:|---:|---:|
| DEFORM hybrid baseline | 12.530179 | 9.183285 | 10.856732 |
| Frozen existing local correction | 11.524244 | 8.917156 | 10.220700 |
| Additional matched ridge | 11.397038 | 8.862328 | 10.129683 |
| Additional single conditional Gaussian | 11.393614 | 8.861541 | 10.127578 |
| Additional finite GMM | 11.433780 | 8.965846 | 10.199813 |
| Additional finite Bayesian mixture | 11.476424 | 8.921833 | 10.199128 |
| Additional DP mixture | 11.473057 | 8.915413 | 10.194235 |

DP improves the existing correction by only 0.2589%; the simpler single Gaussian improves it by 0.9111%. DP records seven wins, five losses and four exact-fallback ties against the existing correction. The descriptive 95% paired trajectory-bootstrap interval for DP-minus-existing error is [-0.100686, +0.048618] mm. Unshrunk selected DP models yield 11.341041 mm, worse than leaving the candidate unchanged. Active component counts are 8/7/7/8; selected shrinkages are 0.25/0/0.25/0.25.

These are source-development errors, not replacements for official DLO4/DLO5 results. This screen does not rule out all DP residual formulations, but does not justify making a DP the paper's central contribution. It establishes neither calibrated uncertainty nor physical-regime identification.

## Executed provenance

The local run used Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. The independent CPU replay in Actions run **34053409199** completed successfully at commit `9b163b026a51b26fad749e0454483655eb8ff87f`, using Python 3.12.3, NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.8.0. All seven tests passed, and every aggregate method error agreed with the local run within 8e-15 mm. This is computational reproduction on the same data, not independent empirical confirmation.

The complete Actions evidence is artifact **9995252800**, `dp-residual-regimes-dev-v1-34053409199-1`, ZIP SHA-256 `c559cefc44f5e4f3d077a0fd4d1c89243de4d51a607c3ffa98deb2baab24de5f`. It retains the protocol, input and code hashes, all validation curves, convergence records, per-trajectory scores, selected-model metadata, logs and formatting-only source copies. This follow-up source-formatting commit does not change the experiment. The exact executed revision above remains the reproduction reference.

The data came from the existing source-only export run **34052556765**, artifact **9994987518**, ZIP SHA-256 `10c8a84a8978e90c4725222fbaec7b203fb5dd49abfd26c4cc9f33c55888cdd5`. Both extracted NPZ files were checksum-verified against the receipt and the original source manifest/prediction identities are pinned in `run.py`. The export artifact has finite retention; retain its downloaded copy for long-term reproduction. The direct source-cache route remains available on gpuserver4090.

## Reproduction

```bash
python -m pip install numpy==2.2.6 scipy==1.15.3 scikit-learn==1.8.0 pytest==8.4.2
python -m pytest -q experiments/dp_residual_regimes_dev_v1/test_model.py
python -m experiments.dp_residual_regimes_dev_v1.run \
  --cache-dir /path/to/extracted/pinned-source-cache \
  --output outputs/dp-residual-regimes-dev-v1
```

Alternatively, `--source-root` points to the original checksum-bound source cache on gpuserver4090. That route reads only the exact named train trajectories. The portable cache route needs no simulator, torch, GPU or raw pickle access. Existing official models and main-branch scientific claims remain unchanged.
