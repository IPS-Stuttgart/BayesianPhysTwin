# DP residual-expert source screen v1

Owner: issue #954. Diagnostic/prospective-method screen, not a change to an installed API or frozen campaign.

## Question

Does a truncated Dirichlet-process mixture improve held-source point forecasts beyond an unchanged action-conditioned ridge surrogate, a matched finite mixture, and a nonlinear RBF residual control? The first screen is entirely passive. It does not identify physical regimes, choose robot actions, use Prob4D, or run the official physical DEFORM checkpoint.

The unchanged `reference.py` is Git blob `57f9aff414f39f52a4ef346e525f40342554e75e`, the reference capsule from PR #940 / run 33985128557. It is retained verbatim for exact baseline reproduction; only its functions are imported. Its historical main program is not the present experiment. This source screen is NOT a comparison against the DLO2 7.8606-mm physical-plus-local-residual result.

## Data and information boundary

Only DLO4/DLO5 `train/*.pkl` may be read from `/mnt/seagate10tb/florianpfaff/datasets/deform/data_set`. The old 32-fit / 12-validation / 12-source-test split and every input hash are preserved; manifest SHA-256 is `9a1ef8d257f4150df842feecc2d45b579eb49a1276e0b023d670221084c0646f`. These source recordings are historically exposed, so held-source is retrospective, not independent confirmation. No official eval directory, other object, new acquisition, or dataset mutation is involved.

Every arm receives the same three observed states and recorded future clamped-node positions as exogenous boundary inputs. Future internal-node positions enter fitting or scoring only, never test gates or predictions. Loading trusted pickles makes all their bytes present in memory; exclusion is enforced at the prediction interface, not by claiming future bytes were inaccessible. Replacement tests verify the input boundary. All models and prediction bytes are saved and hashed before scoring starts.

## Method

For each object and horizon (5, 20, 50), preserve the five origin-specific reference ridge predictors (origins 25, 100, 200, 300, 400). Pool their source-only input features and in-sample remaining residuals, with equal windows per training recording. Source-only PCA reduces input and residual representations to at most six coordinates each. Experts fit a joint Gaussian mixture over both representations; conditioning on input coordinates produces affine residual specialists and input-only mixture weights.

`BayesianGaussianMixture` uses either a truncated stick-breaking DP prior or a finite Dirichlet prior. Both receive caps/counts 2/4/8, total concentration values 0.1/1/10, three source-only initializations, identical Gaussian priors, full covariance, and the same validation shrinkage grid 0/.25/.5/1. Finite per-component concentration is total concentration divided by K; DP concentration is the stick-breaking concentration. These are necessarily different weight priors, not identical parametrizations. Prediction conditions plug-in variational moments; it is not fully integrated parameter uncertainty or a sticky HDP temporal model. Marginal input likelihoods, not future residuals or oracle component choices, set weights.

RBF residual regression uses the same compressed inputs and residuals, nine kernel/ridge settings, and the same shrinkage grid. A single joint-Gaussian expert, persistence, and boundary interpolation complete the controls. Shrinkage zero returns the exact ridge forecast without evaluating the expert. Selecting by validation L1 never sees source-test outcomes. Nonconverged fits are recorded and not silently promoted. All grids, floors and seed values are declared in `CONFIG` before scoring.

## Decision and limitations

Primary: mean coordinate L1 across all 24 complete held-source recordings, five origins and three horizons. DP must improve at least 1% over reference ridge AND have a paired trajectory-bootstrap 95% upper difference limit below zero against ridge, finite mixture and RBF. The 10,000 bootstrap resamples are stratified within the two fixed objects. RMSE and object-wise results are descriptive. A green workflow denotes successful execution, not a positive hypothesis result.

This tests a compact residual-regression candidate, not the full proposed physical-twin extension or its uncertainty-aware observation model. The experts fit pooled correlated windows without fully modeling temporal dependence; equal per-recording window counts and trajectory-level intervals prevent interpreting windows as independent evaluation replications. The PCA representation and in-sample residual fitting may limit attainable gains. No component count is interpreted as a number of physical mechanisms.

## Reproduce

Use Python 3.12 with NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.8.0 and pytest 8.4.2. Set BLAS/OpenMP thread counts to one.

```bash
python -m pytest -q experiments/deform_dp_residual_v1/test_run.py
python experiments/deform_dp_residual_v1/run.py --dataset-root /path/to/deform/data_set --output /new/output/directory
```

The maintained source-capsule workflow accepts a request-only push on the residual-model-screen branch family and pins all capsule file identities. It uses an isolated environment and read-only dataset access. Results, model states, validation records, predictions, tests and runtime identities are retained as an Actions artifact.

References: scikit-learn `BayesianGaussianMixture` documentation (finite and truncated DP variational mixtures); unchanged source capsule PR #940. DP mixtures and switching dynamical models are established methods; this experiment makes no novelty claim for their existence.
