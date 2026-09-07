# Conditional posterior-predictive query transfer v1

## Hypothesis and scope

At identical trajectory means, a context-dependent posterior-predictive distribution improves geometric-query NLL and CRPS over a same-model plug-in distribution and credible empirical residual distributions. This is a new, passive, retrospective hypothesis test on public DEFORM DLO4/DLO5 source recordings. It is not a rerun of the official DEFORM simulator or the previously published BayesianPhysTwin benchmark means.

No robot action is chosen. No measurement is requested. No data are collected or modified. The only dataset reads are the 56 `train/*.pkl` files per object under the user-specified canonical root. The official `eval` directories and all other reserved campaigns are untouched. These source recordings have historical analytical exposure; source-held does not mean independent prospective confirmation.

## Frozen data and inference protocol

The executable's `CONFIG` and query definitions are fixed by the request's Git blob identity. SHA-256 ordering of object/name with the registered domain yields 32 mean-fit, 12 scalar-temperature calibration, and 12 source-test trajectories per object. Source sizes 8 and 16 are nested secondary analyses; 32 is primary. Each object/origin/horizon/size model treats complete recordings, not windows or coordinates, as its fitting observations. Origins are 25, 100, 200, 300, and 400; horizons are 5, 20, and 50 frames.

The common mean is endpoint-displacement interpolation plus a source-fitted action-conditioned ridge residual. Inputs comprise three observed states and the **recorded future clamped-node positions** used as exogenous boundary conditions. This evaluates forecasts conditional on realized boundary motion, not prediction of commanded-action realization. The future internal-node values are never arguments to the prediction function. A replacement test changes those outcomes by enormous amounts and verifies identical inputs.

Source-only PCA supplies at most eight features. A fixed low-dimensional log-scale regression supplies heteroscedastic variance. With that scale, preprocessing, ridge, and prior scale treated as fixed empirical-Bayes hyperparameters, matrix-normal/inverse-Wishart regression gives an exact conditional Student-t predictive distribution. It integrates residual coefficients and covariance, but does NOT integrate the hyperparameters. The algebraic leave-one-row residuals used for source tuning and empirical controls condition on the source-fitted preprocessing; they are not a full nested preprocessing-refit cross-validation.

Seven matched-mean arms are retained:

1. `posterior_student`: parameter-integrated predictive Student-t.
2. `plugin_gaussian`: same conditional model and mean, with coefficients fixed and the posterior-mean noise covariance plugged in.
3. `gaussian_posterior_covariance`: Gaussian with parameter-uncertainty covariance, independently temperature calibrated.
4. `same_covariance_gaussian`: Gaussian with EXACTLY the Student-t predictive covariance, sharing its temperature; isolates distributional shape.
5. `global_shrinkage`: source-residual empirical covariance with 50% diagonal shrinkage.
6. `global_residual_bootstrap`: smoothed symmetric whole-vector empirical residual distribution.
7. `local_residual_bootstrap`: nearest-context whole-vector residual distribution, with the same learned heteroscedastic scale.

The empirical distributions resample whole 24-coordinate residual vectors, preserving dependence. Symmetric resampling keeps every arm's mean identical. All arms can project source residuals into newly requested queries. Independent calibration searches the same variance-temperature grid; bootstrap neighborhood and kernel choices also use only source calibration. The exact-moment comparator deliberately shares the Student temperature instead.

Calibration uses global average position and left-versus-right-half contrast in x/y/z (six scalar queries). Evaluation uses left pair, central pair, right pair, and outer-versus-inner contrast in x/y/z (12 different queries), without query-specific target calibration.

## Scoring and decision

Analytic scalar NLL and CRPS are primary; nominal-90% coverage, interval width, and Brier score for absolute query error exceeding 10 mm are secondary. Student-t and mixture CRPS formulas are tested against numerical CDF integration. All predictions and fitted parameters are hashed before the scorer consumes future internal-node outcomes. Reading a pickle loads its full array; the information exclusion is enforced at the prediction interface, not by claiming that future bytes were physically inaccessible.

Equal-trajectory scores are bootstrapped 10,000 times, stratified within the two fixed objects. These are trajectory-level intervals, not evidence for a population of unseen objects. The primary hypothesis requires upper 95% paired-difference limits below zero for BOTH NLL and CRPS against the plug-in model AND all three empirical covariance/bootstrap baselines. The two Gaussian posterior-moment comparisons diagnose what part of posterior use matters. Every source size and comparator is reported, whether positive or negative. A green workflow means execution and validation succeeded, not that this scientific gate passed.

## Execution

Only a request-file change on `science/deform-conditional-query-posterior-v1` triggers `.github/workflows/deform-conditional-query-posterior-v1.yml`. The request binds the exact source revision and script/test blobs. The job uses `[self-hosted, gpuserver4090]`, an isolated numerical environment, one CPU thread, and no dataset duplication. Runtime output is retained as a workflow artifact, including input manifests, models, prediction seal, predictions, per-trajectory/panel scores, aggregate contrasts, and a readable summary. No existing scientific claim or manuscript is automatically promoted.
