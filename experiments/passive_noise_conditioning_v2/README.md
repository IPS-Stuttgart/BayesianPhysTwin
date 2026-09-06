# Passive visibility and measurement-noise transfer

Follow-up to PR #938. The parent current-completion benefit and failed +25-frame primary result remain unchanged.

## Question and stopping criterion

Does using the supplied observation covariance adapt a fixed conditional predictor to unseen visibility patterns and measurement-quality changes better than a strong source-tuned fixed-gain regressor? Separately, does a rod-inspired covariance improve over an equally noise-aware empirical regressor?

A positive first answer supports operational value of measurement uncertainty. It does not establish uniquely Bayesian superiority: noise-aware Gaussian conditioning and noise-augmented direct ridge have equivalent conditional means and are implemented independently as a parity control. A positive second answer would support useful rod-mode structure. If the second answer is negative, this test does not justify a new Bayesian-accuracy paper claim.

## Real data and controlled perturbation

Use the same immutable cached DEFORM DLO4/DLO5 hybrid-plus-local-residual predictions as PR #938. Eight source-test trajectories per object fit/select the new models; all fourteen already-open evaluation trajectories per object are scored. Both objects' choices and uncertainty calibrations are sealed before evaluation data loading. The source and evaluation predictors were originally trained with different data budgets (39 and 56 trajectories); this limitation is retained.

Source selection uses every contiguous mask of two or four of eight free nodes and 0/5/15 mm independent Gaussian noise. Evaluation uses every NON-contiguous subset: 21 two-node and 65 four-node masks, never used for hyperparameter selection. Each is a separate one-step conditioning query, not a recursive changing-mask tracking experiment.

Primary: current hidden geometry, two observed nodes, equal mixture of clean, 2 mm iid, 10 mm iid, alternating 2/10 mm independent quality, and 10 mm shared translation plus 2 mm independent noise. Additional 30 mm iid and quarter/fourfold supplied covariance are separately retained stress conditions. +5 and +25 frames are secondary. No observation or action is selected adaptively.

The motions are real; masks and noise models are artificial. Measurement noise is integrated analytically, not sampled:

`E[(mu + K(y + epsilon - mu_o) - truth)^2] = bias^2 + diag(K R K.T)`.

The primary metric is per-trajectory root expected coordinate MSE, averaged across the two fixed physical objects. This is not an empirical real-camera-noise benchmark. Supplied noise covariance is oracle-known except the explicitly labelled misspecification controls. No robot experiments, data copying, data mutation, or new simulator training occur.

## Matched controls

All arms share the same source residual mean. Controls include source-tuned fixed ridge, pooled-noise-augmented fixed ridge, empirical Gaussian conditioning with full supplied noise covariance, rod-mode conditioning, observation-correlation-blind conditioning, and independently solved noise-aware direct ridge. The stronger of the fixed-gain controls is selected using source validation only. All hyperparameter choices are leave-one-complete-source-trajectory-out, tuned on an equal current/+5-frame MSE objective, never evaluation outcomes.

Every arm additionally compares its source-calibrated conditional predictive variance with source-fit static per-coordinate residual variances, keeping that arm's mean EXACTLY fixed. Gaussian NLL is also integrated analytically over measurement noise. Both uncertainty alternatives receive independent source-only calibration; no calibration claim follows just from a relative proper-score improvement.

## Execution and evidence

The workflow only starts after a request-file change on `science/passive-noise-conditioning-v2`. A `prepare` request formats/tests the two new Python files on GitHub-hosted CPU, with no dataset access; only formatting changes are committed to this branch. An `evaluate` request binds the immediately preceding source revision, then executes on `[self-hosted, gpuserver4090]` with read-only repository permission and run-specific outputs.

Eleven local tests passed before upload, including exact-risk/Monte-Carlo agreement, covariance/adaptive-gain properties, complete disjoint mask rosters, source-only calibration, future/hidden-input poisoning, and independent Gaussian/ridge equivalence. Synthetic unit tests are not physical evidence.

Artifacts retain all per-trajectory metrics, source choices, a source seal, protocol, paired intervals and report. Inference on evaluation observations is separated from scoring by a prediction function accepting only visible current residuals. Complete trajectories, not masks/frames/coordinates, are bootstrapped within DLO; inference is conditional on these two objects. No overall state-of-the-art, unseen-object, physical-state, safety, or fresh-confirmation claim is made.
