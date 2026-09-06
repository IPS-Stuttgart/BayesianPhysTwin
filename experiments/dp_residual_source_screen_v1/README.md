# DP residual source screen v1

## Result

A completed local five-fold, whole-trajectory source-data screen found useful
**mixture-of-residual-experts** value but no convincing **DP-specific** value.
The matching GitHub Actions workflow reproduces the same source-only experiment;
its run/artifact is the authority for whether CI reproduction has completed.

| Predictor | Normalized RMSE [% of endpoint span] | Normalized coordinate L1 [% of endpoint span] |
| --- | ---: | ---: |
| Damped endpoint-blend kinematic baseline | 7.059817 | 4.272813 |
| Global ridge residual | 2.476748 | 1.510462 |
| RBF kernel-ridge residual | 2.338500 | 1.448495 |
| Matched finite-mixture full-feature experts | 2.270506 | 1.388371 |
| DP-mixture full-feature experts | 2.268763 | 1.386975 |

The DP arm improves normalized RMSE by 8.40% against global ridge and 2.98%
against RBF kernel ridge, but by only **0.0768% against the matched finite
mixture**. DP minus finite error is -0.001743 percentage points of span; the
descriptive stratified trajectory-bootstrap 95% interval is
[-0.005094, +0.001721]. DP wins 66/112 paired trajectories against finite,
97/112 against ridge, and 67/112 against RBF kernel ridge. This is not a
DP-specific positive result. All fitted mixture candidates converged in the
local five-fold run.

**These are not millimetres and not the existing DEFORM hybrid benchmark.**
The archived targets are residuals divided by each query's endpoint separation.
That per-query metric length is not retained in this archive, so multiplying
these numbers by 1000 would incorrectly label normalized units as millimetres.
The baseline is the kinematic predictor from the decision-identifiability
experiment, not the trained DEFORM rod-plus-GCN hybrid and not the existing
per-node local-residual module. No improvement over those native methods is
claimed here.

## Data and information boundary

The source is GitHub Actions artifact `9787311310` from this repository, with
ZIP SHA-256:

```
fdb35f680e4cf685303f841b8974c16eb9301ad52cbdc8f5af0d87a9dfc358ee
```

The script separately verifies `source_model.npz` and `source_result.json`.
It uses only their raw source features/residuals and trajectory roster, ignoring
pre-fitted class labels, feature normalizers, and the parent selected model.
The bundle contains DLO4/train and DLO5/train: 56 trajectories each, 19 windows
per trajectory, 2,128 windows total. Each window predicts 25 frames of eight
internal 3-D nodes from a five-frame prefix and prescribed future endpoint
motion. The 81 normalized input features and 600 normalized response coordinates
are inherited from the parent source representation.

See the parent implementations:

- `experiments/deform_dlo45_decision_identifiability_v1/_common.py`, especially
  `observation_from_parts` and `source_window`;
- `experiments/deform_dlo45_decision_identifiability_v1/_model.py`, especially
  `build_pool` and `fit_model`.

Each DLO is fitted separately. Five outer trajectory folds each reserve nine
other trajectories for hyperparameter validation; all remaining trajectories
fit the model. Every trajectory is scored once out of fold. Preprocessing,
PCA, priors and experts use fit trajectories only. Validation selects models.
Prediction APIs accept features only; future responses cannot be supplied.
Selections and predictions are written before scoring each outer test fold.
This is semantic data separation: the historical source archive contains all
source responses, but test responses are not used in prediction or selection.
No official evaluation directory or new physical execution is accessed.

## Model and controls

A truncated variational Bayesian Gaussian mixture models the joint distribution
of 12 training-only input principal components and 12 residual principal
components. The prediction-time gate uses the **feature marginal** likelihood,
not joint likelihood evaluated at the unknown future response.

The main full-feature experts retain all 81 inputs and 600 outputs. Each expert
fits a weighted ridge correction to the global ridge model, with coefficients
shrunk toward that global model. Source joint responsibilities are permitted
for expert fitting, while forecast responsibilities only use permitted features.
The DP and finite arms have identical features, experts, initializations,
covariance family and validation budgets. Both search caps K={2,4,8}, total
concentration alpha={0.1,1,10} and expert ridge={10,100}. Finite per-component
concentration is alpha/K; DP stick concentration is alpha. Both therefore fit
18 full-expert candidates per fold, sharing nine gate fits.

This is a **DP-gated, plug-in mixture of ridge experts**, not a complete joint
Bayesian posterior over all regression parameters and not a sticky HDP-HMM.
No temporal-transition or regime-duration model is implemented here. The
predictive uncertainty calibration of this extension has not been evaluated.

Controls include no correction, validation-selected global ridge, weighted
nearest-source-window residuals, RBF kernel ridge, one conditional Gaussian,
and reduced-output finite/DP conditional Gaussian mixtures. Reduced-output
mixtures were much worse (DP normalized RMSE 3.875118 versus full-feature
2.268763); retaining the full regression features and outputs matters.

## Development provenance and statistical limits

An initial historical 39/9/8 split, scored on 16 source trajectories total,
showed the compressed conditional mixtures losing to ridge and kernel ridge.
Full-feature regularized experts were then added to exclude a PCA bottleneck.
On that historical split, full-feature DP and finite were virtually tied
(2.336639 versus 2.336683), and RBF kernel ridge was slightly better (2.311550).
The subsequent five-fold analysis remains **retrospective development**, not
fresh independent confirmation or a prospectively selected new result.

The local sandbox interrupted the first five-fold command between panels.
Completed panels were retained unchanged; remaining panels were completed with
identical code and settings. The GitHub workflow runs all panels in one job.
Bootstrap intervals resample complete trajectories within each DLO and are
descriptive: overlapping training folds, prior development and only two DLO
object types preclude interpreting them as independent confirmatory evidence.

The selected DP used the cap of eight components in eight of ten fits, with
seven or eight components above 1% weight in those fits. The remaining two
selected four-component models. This small screen does not establish that a
larger truncation, a temporal HDP model, or native-hybrid integration cannot help.
It does establish that the tested DP prior does not materially outperform the
matched finite-mixture implementation.

## Reproduction

Use Python 3.13, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0.
After extracting the verified source artifact:

```bash
python -m pytest -q tests/test_dp_residual_source_screen_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  -m experiments.dp_residual_source_screen_v1.run \
  --source-dir /path/to/extracted-source \
  --output /path/to/new-output-directory --folds 5
```

The output includes every validation score, fitted mixture diagnostics, exact
trajectory splits, prediction seals, per-trajectory metrics, descriptive paired
contrasts, runtime versions and script/source hashes. The workflow
`.github/workflows/dp-residual-source-screen-v1.yml` downloads only the verified
training-source artifact and retains the full run as an Actions artifact.

This experiment changes no frozen benchmark, public claim, physical checkpoint,
provider contract or manuscript status. The interpretation is: residual expert
sharing is promising in this source representation; the DP itself is not yet a
supported paper contribution.
