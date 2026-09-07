# Fixed-forecast heavy-tail controls

## Purpose

Continue the completed conditional posterior query experiment from PR #940 / Actions run `33985128557`. Its 32-fit primary superiority condition was not met, while a prespecified 8-fit secondary condition favored the posterior against the original controls. The missing comparison is a source-calibrated heavy-tailed model that does not integrate over a coefficient/covariance posterior.

This follow-up is designed **after those outcomes were observed**. It is a retrospective falsification/control analysis, not a new prospective confirmation. The original Bayesian model, predictions, scoring definitions, source memberships, and primary-versus-secondary source-size designations stay immutable. A new comparison cannot retroactively change the original primary decision.

## Data and frozen input

The parent artifact `9974939619` has ZIP SHA-256 `e7e182e009ac234fb611afb6ca03eb1006ee65da2688ec2a084105667fa4e591`. The evaluator verifies it, all four original prediction-seal hashes, and the exact parent evaluator Git blob before importing code. Dataset reads use only the original twelve calibration plus twelve source-test records per DLO from the user-authorized central `DLO4/train` and `DLO5/train` directories. Every file hash is checked against the parent manifest. No directory enumeration selects additional cases; no official `eval`, reserved data, other dataset, active probing, or new acquisition is used.

Parent fitted models already contain the source-fit residual vectors. They are reused without refitting. Forecasts continue to condition on recorded future clamped-node positions, not newly chosen robot commands. Pickles contain full trajectories; exclusion of future internal outcomes is enforced at the prediction-function interface, not claimed as physical byte inaccessibility.

## Four additional matched-mean controls

Let `V=(Xw^T Xw + lambda I)^-1` be the ridge inverse precision and `s2(c)` the original source-fitted heteroscedastic variance scale. The first three controls use a zero-centered empirical second moment of whole residual vectors with source-calibrated diagonal shrinkage. They preserve spatial dependence and can project into every new query.

- `empirical_student_global`: Student-t with constant empirical residual covariance.
- `empirical_student_conditional`: Student-t with normalized-residual covariance multiplied by `s2(c)`.
- `sandwich_student_conditional`: the same empirical Student-t plus the frequentist sampling-covariance term `x V Xw^T Xw V x^T`. It does not use the Bayesian coefficient-covariance term `x V x^T` and does not integrate a posterior.
- `matched_plugin_student`: the original plug-in noise-covariance estimate with independently calibrated Student-t shape and scale, without coefficient integration. This reuses the parent's noise-covariance estimate, so it is a distribution-family diagnostic, not an independently estimated empirical covariance baseline.

All four distributions are centered at the EXACT original forecast. They fit degrees of freedom, variance temperature, and (for empirical arms) shrinkage using only equal-recording NLL over the original six development queries. The df grid is `[3,4,5,8,11,16,19,35,64,128,infinity]`, including the parent dfs 11/19/35 and a Gaussian option. Temperature uses the unchanged parent grid. Shrinkage uses `[0,.25,.5,.75,1]` and preserves coordinate variances. No target metric chooses between these arms. All eleven arms, including the seven unchanged parent arms, are reported at 8, 16, and 32 fitting trajectories, each with twelve additional calibration trajectories per object.

For a desired variance `v`, a finite-df Student-t uses `scale^2=v*(df-2)/df`. The implementation tests this distinction against SciPy's variance routine. Calibration NLL is vectorized but checked independently against SciPy log densities for every grid entry. The sandwich term is checked against the algebraic difference `lambda x V^2 x^T` from the Bayesian coefficient term.

## Evaluation and interpretation

New source-fitted controls are saved and hashed before source-test prediction. All new predictions are saved and hashed before scoring. The original Bayesian score is replayed against the same recorded outcomes and must agree within `1e-10`; forecast reconstruction must agree within `1e-12 m`, and the actual stored means are copied exactly.

The formal follow-up criterion keeps **32 fitting trajectories primary** and checks both NLL and CRPS against the conditional empirical and sandwich Student-t controls. The 8-fit question is an explicitly secondary follow-up of the original signal. Every contrast, including adverse ones, is retained. Pointwise 95% intervals use 10,000 complete-trajectory bootstrap replicates stratified within the two fixed objects. They are descriptive post-result intervals, not multiplicity-adjusted fresh confirmation or population-of-objects uncertainty.

These controls test whether the observed Bayesian gain survives credible heavy-tailed alternatives. They cannot establish that Bayesian inference is uniquely required: equivalent predictive distributions can have multiple derivations. None of the hyperparameters or the parent preprocessing is fully integrated. Algebraic leave-one-row residuals condition on the original source-fitted preprocessing rather than rerunning nested PCA. No full DEFORM simulator, Prob4D provider, or Causal4D intervention pipeline is evaluated.

## Execution and output

The new workflow `.github/workflows/deform-query-heavy-tail-controls-v1.yml` runs on `[self-hosted,gpuserver4090]` only when `.github/requests/deform-query-heavy-tail-controls-v1.json` changes on the existing PR branch. The request-only commit binds the preceding implementation revision and all executable/workflow blobs. The job downloads the 9 MB immutable parent artifact rather than copying datasets and uses an isolated NumPy/SciPy runtime with one CPU thread.

Outputs include source-fitted control settings, sealed predictions, protocol, hash seal, all arm/context scores, aggregate and per-object scores, paired differences, a readable summary, and the verified parent artifact contents. Passing workflow checks means technical execution, not scientific superiority. No manuscript or existing claim is promoted automatically.
