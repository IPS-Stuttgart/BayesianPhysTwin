# Recorded residual regimes v1

Owner: IPS-Stuttgart/BayesianPhysTwin#951. Classification: retrospective research
experiment, not a stable API and not a change to a frozen estimator or result.

## Question

Does a library of persistent residual-dynamics experts improve forecasts beyond
the existing DEFORM hybrid, its BayesianPhysTwin local correction, and a matched
last-observed-residual control? Does a weak-limit sticky-HDP prior improve over
validation-selected finite switching experts?

No robot, probing, action selection, or new acquisition is involved. The only
inputs are existing DLO1 training recordings and the original frozen source
checkpoint. Official evaluation partitions and DLO2--DLO5 are excluded.

## Data and information

Use the original manifest in
`results/sota/deform_dlo_source_v1/source_manifest.json`: 40 fit, 8 validation,
and 8 previously opened source-test recordings. The latter are retrospective
scoring data, not fresh confirmation. No new physical object is represented.

The exporter verifies hashes and exactly reuses the 6400-update DEFORM hybrid
and the original local residual (ridge 1, shrinkage 0.5). DEFORM itself includes
a learned GCN correction; this is not a comparison against bare rod physics.
It verifies the historical validation replay before exporting separate fit,
validation, and source-test NPZ files.

The frozen hybrid trajectory starts from the original two observed states and
uses the benchmark's known future clamped-node trajectories. At each registered
forecast origin the correction methods may additionally observe the last eight
recorded states. They never receive future free-node states. This is passive
readout-residual forecasting on an unchanged rollout, not physical-state
injection, adaptive sensing, or the original full-trajectory benchmark metric.

## Models

All learned experts operate on rank-6 source-only PCA coefficients of the
existing local predictor's residual. Four source-only PCA features summarize
permitted state/action/baseline information. Each expert predicts the next
coefficient vector using an intercept, these features, and the previous
coefficient vector. The observed residual orthogonal complement persists.

The expert prior is matrix-normal/inverse-Wishart: M0 has zero intercept and
exogenous coefficients and a 0.98 I autoregressive block; row precision is I;
Q has inverse-Wishart prior with df D+2 and scale 0.01 I. This prior does not
impose a hard stability constraint. Nonfinite forecasts raise rather than being
silently clipped.

Finite K=1,2,4,8 use uniform global weights and sticky Dirichlet transition rows.
The HDP arm uses a K=8 weak-limit approximation, auxiliary restaurant-table
updates for global weights, and the same expert family. Alpha=2, kappa=20,
gamma=1. This is not exact infinite-dimensional inference. Seed=42; 40 Gibbs
sweeps; 20 burn-in; retain every fifth remaining draw (four parameter draws).
One exploratory chain does not establish convergence or posterior calibration.

The future-mode first moment is marginalized analytically conditional on each
parameter draw; it is not a hard mode assignment or a plug-in mean-mode rollout.
The forecast API accepts only prefix responses and permitted future features.

## Selection and evaluation

Origins are offsets 16,116,216,316 in the 498-frame exported prediction arrays.
Horizons are 10,50,100 frames. The primary selection metric is coordinate L1
averaged over the first 50 forecast frames, then windows within each recording,
then recordings. All clamped coordinates remain exactly unchanged.

Controls are hybrid, existing local residual, hybrid plus last observed
residual, and local plus last observed residual. Validation alone selects the
best control and finite K. All settings and posterior artifacts are sealed in
`selection.json` before the new model scorer loads the source-test NPZ. Every
fixed candidate is reported; there is no test-based reselection.

Paired exploratory bootstrap intervals resample the eight whole recordings,
not frames, points, coordinates, or overlapping windows. They do not establish
object-level generalization or account for unknown dependence across recordings.

## Run

The self-hosted source exporter uses the existing frozen DEFORM environment:

```bash
python experiments/recorded_residual_regimes_v1/export_dlo1.py --output /new/export
```

The portable model benchmark requires NumPy and SciPy, not a GPU:

```bash
python -m pytest -q experiments/recorded_residual_regimes_v1/test_model.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  experiments/recorded_residual_regimes_v1/model.py \
  --input /new/export \
  --request .github/requests/recorded-residual-development.json \
  --output /new/model-results
```

Output paths must be new. Preserve exported input hashes, model source hash,
protocol, posterior draws, selection, per-recording scores, and software versions.
The file-triggered `recorded-residual-development.yml` currently executes the
export stage; invoking it does not imply the model benchmark has completed.

## Boundaries

The fit residuals are in-sample with respect to the source-trained hybrid/local
model. This is an initial, single-DLO, previously opened development study.
A favorable result is neither fresh confirmation nor a paper-wide accuracy,
calibration, causal-contact, physical-regime-count, safety, or state-of-the-art
claim. A negative result is not a universal impossibility result for DPs.

Eight implementation tests cover FFBS, sequence boundaries, exact future mode
moments, autoregression, probability weights, SPD draws, and reproducibility.
Synthetic fixtures are implementation tests only and must not be reported as
recorded-data evidence.
