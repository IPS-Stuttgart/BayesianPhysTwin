# Material-aware GP residual development pilot

## Purpose and status

Test whether replacing the existing outer per-node ridge residual with a nonlinear Gaussian process improves predictions from the **same frozen DEFORM hybrid checkpoint**. This is a development experiment, not a new official evaluation, independent confirmation, calibrated-physical-posterior result, or state-of-the-art claim. It requires recorded trajectories only; no robot or additional action acquisition is involved.

The implementation is isolated in `scripts/experiments/material_gp_development_v1.py`; the native residual implementation and production selection policies are unchanged. The workflow is restricted to the experiment branch. Execution results are retained as GitHub Actions artifacts, including failures.

## Data and frozen backbone

Use only the original DLO2 v5 `fit` (40 files) and historical `validation` (8 files) in the checksum-bound source manifest. The eight source-test files and every official evaluation partition are excluded. Python audit hooks reject unapproved trajectory reads. Data remain on the existing self-hosted runner.

The checkpoint is the original 6400-update DEFORM **physics-plus-GCN hybrid**, not pure rod physics. SHA-256: `b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65`. The exporter verifies the source-training record, manifest, checkpoint, upstream revision, and reproduction of the historical validation coordinate L1 (`7.912029745057225 mm`, tolerance `0.0001 mm`). No backbone retraining occurs.

The cached arrays contain two observed initial states, the recorded future clamped-node action, the frozen baseline forecast, trajectory identities, and reference targets. Only initial state, recorded boundary motion and baseline forecast enter prediction features. Recorded future boundary motion is permitted by this operator; this is not an unknown-action forecast.

Exact causal-query duplicates are grouped before residual fitting and evaluation. A cross-split exact-query overlap causes an error rather than silently treating duplicates as independent. The existing per-node ridge implementation is reused directly.

## Models

- Unchanged DEFORM hybrid.
- Existing outer ridge: ridge 1.0, shrinkage 0.25, fitted to the permitted fitting trajectories.
- Retuned ridge: ridge in `{0.1, 1, 10}` and the same shrinkage choices as the GPs.
- Independent per-node GP: linear plus Matérn-3/2 deformation kernel.
- Material-sharing GP: the same deformation kernel multiplied by a Matérn-3/2 material-coordinate kernel with length 0.4 on the normalized internal-node index.

The spatial coordinate is a **uniform material-node coordinate**, consistent with the existing feature builder; it is not a newly measured deformed-space distance or exact unequal-edge physical arc length. Both GP arms receive the same pooled, fitting-only feature normalization. The material kernel prevents proximity in deformed world coordinates alone from determining sharing.

Sparse inference uses a whitened fully independent training conditional (FITC) construction. All three canonical output coordinates use the same kernel and separate training-only RMS scales. There are 768 inducing locations in total: 96 per internal node for the independent arm, versus 768 jointly for the shared arm. Inducing locations are selected from fitting inputs only using seed 20260907. GP fitting uses 24 uniformly spaced time samples per trajectory; final scoring uses the complete forecast horizon. The native ridge is allowed to use all its normal fitting frames, so it is not artificially weakened to match the sparse GP training density.

## Selection and scoring

An inner, whole-trajectory split of the original fitting groups selects hyperparameters. The first quarter of a fixed seeded group permutation is the inner tuning set; the other groups fit candidate models. No frame-wise random split is used.

Both GP arms search deformation length in `{0.5, 1.5, 4}` and normalized observation-noise variance in `{0.05, 0.25, 1}`. All selected residual methods use shrinkage in `{0, 0.125, 0.25, 0.5, 1}`. Inner coordinate L1 selects each arm; each selected arm is refitted on all fitting groups before scoring historical validation. Zero shrinkage is allowed and means no mean correction.

The workflow retains the complete candidate bank, selected settings, inner split identities, all final mean predictions and marginal variances, a prediction-content seal, and trajectory-level results. Boundaries remain exactly equal to the baseline. Primary metrics are full-trajectory coordinate L1 and coordinate RMSE in millimetres, together with paired trajectory wins, worst case ratios, and early/middle/late errors. Descriptive paired bootstrap intervals resample entire validation groups, not frames or coordinates.

The historical validation split already influenced checkpoint selection and past development. Keeping it out of this pilot's residual fitting does **not** turn it into a fresh confirmation set. The inner split also holds out residual fitting, not backbone training: the frozen backbone previously trained on the original fit partition.

## Uncertainty limitations

Coordinate Gaussian NLL, nominal-90% marginal coverage, normalized coordinate NEES, and interval width are diagnostic. A second report applies one scalar covariance multiplier fitted from inner tuning residuals only. The GP likelihood treats the thinned within-trajectory errors as conditionally independent; it does not implement Prob4D shared observation covariance. No full spatiotemporal covariance artifact is exported in this first pilot.

Shrinkage and variance scaling are operational adjustments, not a proof that the exported covariance is an exact calibrated physical posterior. GP kernels, noise misspecification, sparse approximation, uncertain inputs and the very small validation panel can all affect results. Mean improvements, if any, do not by themselves establish Bayesian-posterior value.

## Tests and reproduction

`tests/test_material_gp_development_v1.py` checks kernel positive semidefiniteness, exact-GP recovery when all training inputs are inducing locations, finite positive marginal variance, deterministic fits, invalid-input rejection, material-distance behavior, exact boundary preservation and fitting-only normalization. These tests are software evidence only.

On the configured runner, run the frozen-runtime exporter followed by:

```bash
python scripts/experiments/material_gp_development_v1.py \
  --data /path/to/development.npz \
  --output-root /path/to/new-comparison-directory
```

The script requires NumPy and SciPy in addition to the checked-out BayesianPhysTwin sources. The adjacent `export.json` must match the input archive. Existing result directories are never overwritten. The workflow uses the existing pinned Python 3.10 / Torch 2.0.1 runtime for faithful DEFORM replay and does not modify that environment.
