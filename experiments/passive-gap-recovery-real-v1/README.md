# Passive gap and recovery on real cloth trajectories

## Question

Does recursively propagating discrepancy uncertainty improve boundary-conditioned free-marker predictions during observation gaps and recovery, compared with equally informed deterministic temporal corrections?

This is **not** active probing or robot action selection. No new physical data are collected. The study uses the existing limited equal-mass spring-mesh pilot, not released PhysTwin, DEFORM, a new material simulator, or an RGB-D perception model.

## Data and scope

Tracking Cloth Deformation, Zenodo record 14644526, is read from the canonical gpuserver4090 cache. The complete 64-recording free-hanging panel consists of 32 shaking source recordings and 32 twisting evaluation recordings, over eight material/size specimens. These recordings have already been opened in earlier studies; the new missing-observation question is a **retrospective real-trajectory mechanism test**, not independent fresh-object confirmation or a replacement for a previous negative result. All 56 collision recordings remain numerically unopened by this experiment. The existing integrity audit hashes archive/file bytes, including unused files, without parsing their measurements.

Native motion-capture positions serve as observations and reference values. Only observations are hidden; no synthetic motion, noise, drift, or outliers are generated. This isolates missing-data inference but does not establish robustness to RGB-D errors.

The two driven corners remain continuously available current-time conditioning inputs and are excluded from the score. The free markers are hidden for 0.1, 0.3, or 0.6 seconds, three times during each six-second clip. Each gap is followed by a 0.3-second recovery window. This is not a total sensing blackout: boundary observations remain available. No future free-marker or future-boundary input enters a prediction at time t. Spring rollouts can be precomputed because their recurrence is causal in the boundary input.

## Frozen comparisons

All methods receive the same observations, initial information, validity mask, and physical baseline. No method receives an exclusive outlier/reliability gate. Scores use the forecast **before** assimilating the current free-marker measurement, including the first returning observation.

- Source-selected spring prior without correction.
- Last residual and source-tuned exponential correction.
- Source-tuned alpha-beta correction of the spring residual.
- Raw-position persistence, constant velocity, and source-tuned alpha-beta filtering.
- Recursive Gaussian discrepancy/increment belief.
- Stationary-gain control with the same Gaussian dynamics/noise parameters.
- Covariance-reset control, resetting accumulated uncertainty when measurements return.
- Independently implemented information-form sequential MAP under the identical Gaussian model.

The Bayesian grid has 12 candidates; each alpha-beta grid has 60 candidates. Each method is independently selected using only the 32 source recordings, with equal recording and condition weights. The physical parameter grid is selected on source before the inference comparison. The strongest source-selected deterministic method is frozen separately for each specimen before target prediction.

Gaussian mean and mode coincide. The MAP arm must agree numerically with the filter and prevents promotion of a covariance-versus-fixed-gain result into a claim of unique point-estimate superiority over equivalent MAP. A same-prior restart diagnostic additionally compares adaptive and stationary gains from identical pre-update means.

## Endpoints

Primary: equal-specimen, equal-recording, equal-gap-duration mean gap-and-recovery 3-D RMSE, Bayesian minus the source-selected deterministic reference. The frozen positive gate requires a specimen-bootstrap 95% interval below zero, at least 5% relative improvement, and a clean-condition cost upper confidence limit at most 0.5 mm. The material-cluster bootstrap is a separate sensitivity analysis (four materials versus eight material/size specimens).

Secondary: gap-only and recovery-only RMSE, clean RMSE, per-record 95th-percentile marker error, and all individual comparator contrasts. Coordinate coverage, width, and NLL are descriptive for the Bayesian arm; this experiment makes no calibration claim. Frames, coordinates, and repeated masks are not independent statistical units.

All source fits are saved before any target prediction. All target prediction hashes are sealed before target scoring. Invalid data or incomplete cases raise an error rather than silently changing the roster. Original data are read-only and never uploaded. Artifacts contain aggregate/per-record statistics, provenance, hashes, fitted parameters, and logs.

## Run

The workflow `.github/workflows/passive-gap-recovery-real-v1.yml` uses a push path filter on `request.json`, restricted to the experiment branch, and runs on `[self-hosted, Linux, X64, gpuserver4090]`. It does not use workflow_dispatch. Scientific source blob identities are checked before execution.

Initial push-triggered run: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33984801057

```bash
python -m experiments.passive_gap_recovery_v1 --self-test
python -m experiments.passive_gap_recovery_v1 \
  --dataset-root /home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526 \
  --output /tmp/passive-gap-evidence --workers 1
```

The self-test checks Gaussian MAP parity, suffix invariance, pre-update timing, uncertainty growth across missing intervals, exact residual persistence, the covariance-reset intervention, time-only masks, and bootstrap behavior. Passing these checks is software evidence, not a positive scientific result. Workflow success likewise means completion, not hypothesis acceptance. Read `result.json` and `report.md` for the scientific decision.
