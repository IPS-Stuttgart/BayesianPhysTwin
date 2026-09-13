# Partial-observation covariance experiment v1

## Question

Can source-fitted spatial/temporal dependence turn a partial current observation into better hidden-region completion and prediction on recorded DEFORM trajectories?

No robot action is selected. No new physical data are acquired. Observation masks are fixed, contiguous sets of 1, 2, or 4 of the 8 free nodes; every possible contiguous placement is included. The primary endpoint is hidden-node coordinate RMSE at +25 frames with 2 observed free nodes. Current completion and +1/+5/+10-frame forecasts, plus all observation budgets, are retained.

## Data and information boundary

The workflow reads the canonical DEFORM DLO4/DLO5 data on `gpuserver4090` and immutable cached forecasts from parent run `33361441865`. The baseline is the retrained DEFORM physics-plus-GCN hybrid, and the common predictive anchor includes the existing frozen local residual. It is not bare rod physics.

Eight source-test trajectories per DLO were excluded from the parent source model's training. These are the only trajectories used to estimate this new conditional model and to select hyperparameters by leave-one-complete-trajectory-out validation. All 14 official evaluation trajectories per DLO are then scored. They were already opened by earlier studies, so this is a **retrospective new-task evaluation, not fresh confirmation**. The source and evaluation anchors also come from different training budgets (39 versus 56 trajectories), which is retained rather than concealed.

Both objects' source choices are sealed before evaluation trajectories are loaded. The predictor receives cached baseline forecasts and current visible coordinates only. It never receives hidden or future internal-node outcomes. Known recorded clamp trajectories and two full initialization frames are inherited identically from the cached forecasts. The experiment uses artificial measurement masking of real recorded motion, not raw-image occlusion handling.

Original parent outcomes, data, checkpoints, and claim records are never changed. Forecasts, manifests, and raw trajectory files are checksum verified; original source metric parity is checked before fitting.

## Methods

All conditional methods share the same source-fitted residual mean. Methods include empirical joint Gaussian conditioning, a two/four-sine-mode covariance with diagonal remainder and matched coordinate marginal variances, RBF kernel conditioning, separately solved direct ridge, linear residual interpolation, global translation, and no-conditioning baselines. A sign-scrambled covariance preserves coordinate marginals and positive semidefiniteness while destroying useful dependence.

The rod-mode covariance is a source-fitted predictive discrepancy model. It is **not** a simulator parameter posterior or a revised latent physical state. The empirical Gaussian and matched direct ridge conditional means are algebraically equivalent: the study must not claim a unique Bayesian advantage merely by renaming regression.

Independent source-only hyperparameter selection is supplied to the competitive families. The conventional comparator is selected on source validation, not on target performance. Scrambling is a same-hyperparameter mechanism ablation, not a competitive method.

## Outcomes

All methods, masks, horizons, and trajectories are retained. Primary structured advantage requires at least 1% gain, a paired 95% interval excluding zero, and improvement on each DLO. The complete trajectory is the resampling unit; the bootstrap is stratified within these two objects and does not establish arbitrary-object generalization.

The artifact contains `result.json`, `per_trajectory.csv`, `source_selection.json`, `model_seal.json`, `prediction_seals.json`, and `report.md`. A numerical/integrity failure is retained in `failure.json`, not converted to a scientific negative or silently dropped. No checkpoint or raw recording is uploaded.

## Execution

The branch-specific push workflow is triggered **only** by changing `.github/requests/partial-observation-covariance-v1.json`; it rejects a trigger commit that changes other files. `mode=inventory` checks existing paths; `mode=evaluate` validates the protocol checksum, executes the focused tests, and runs the evaluation. It requests `[self-hosted, gpuserver4090]` and uses a small isolated NumPy environment. The cached forecasts make new GPU simulation or data copying unnecessary.

The seven focused tests cover Gaussian gain, zero cross-covariance, direct-ridge parity, positive/marginal-matched covariance, masked/future outcome invariance, complete mask accounting, finite inputs, and source-order invariance. Synthetic unit tests are software validation only.
