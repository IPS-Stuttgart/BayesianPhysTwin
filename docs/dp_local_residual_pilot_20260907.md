# Dirichlet-process local-residual development pilot

## Status

This is a development experiment, not a promoted method or a paper result.
The implementation and constructed-data software checks were completed in the
interactive session. The real DEFORM workflow was submitted, but its job was
still queued when this status was written. No measured DEFORM DP result exists
in this record. Queueing is neither a positive nor a negative scientific result.

- Experiment branch: `experiment/dp-local-residual-pilot-20260907`
- Execution revision: `6bb78cbe02adc307bd54a0aa9f45a2ee70024325`
- Workflow run: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34053321725
- Workflow job: `101540698844`
- Workflow: [dp-local-residual-pilot.yml](../.github/workflows/dp-local-residual-pilot.yml)
- Implementation: [pilot.py](../experiments/dp_local_residual_pilot/pilot.py)
- Exporter: [export_development.py](../experiments/dp_local_residual_pilot/export_development.py)

## Question

Does input-conditioned DP Gaussian-mixture regression improve the residual
correction of the existing DEFORM physics-plus-learned-GCN predictor, beyond
both the existing local ridge correction and matched finite mixtures?

No physical-data collection, active probing, robot access, new observation
provider, or Causal4D integration is required.

## Implemented comparison

The exporter reuses the existing local-residual feature builder and physical
rollout. The mixture fits a joint distribution of input features and 3-D
residuals independently for each free material node. At prediction time,
component weights depend only on the input marginal, and Gaussian conditioning
produces each expert's residual prediction. No future free-node residual enters
the predictor. Prescribed-node coordinates remain byte-exact.

Models: existing baseline; existing ridge correction; maximum-likelihood finite
Gaussian mixtures (K=1,2,4,8); finite Bayesian Gaussian mixtures (K=2,4,8); a
truncated DP mixture (Kmax=8, alpha=0.1,1,10); and an ExtraTrees nonlinear
residual comparator. Mixture input preprocessing is training-fitted scaling and
PCA with up to eight whitened components. Shrinkage is selected from
0, 0.25, 0.5, and 1. A fixed DP alpha=1, shrinkage=0.25 arm is also retained.

All mixture predictions use plug-in variational component moments, not exact
posterior-predictive integration. This pilot makes no calibration claim.

## Data and selection boundary

The required data are the pinned historical DLO2 fit40 and validation8 splits.
Exact duplicate causal queries in fit40 are grouped before an inner 80/20 split.
Hyperparameters and shrinkage are selected on this inner split; selected models
are refitted on fit40 before scoring the historical validation8 trajectories.
That validation set was previously used for checkpoint selection, so this is
not fresh confirmation. Frames and nodes are not independent test cases.

The source-test8 split and official evaluation are excluded. The exporter
requires exact training-record, source-manifest and checkpoint SHA-256 matches
from the existing v6 protocol, verifies the upstream revision, installs a
pickle-read allowlist, and checks baseline validation-error reproduction before
writing model inputs. Initial cache inventory returned metadata inconsistent
with the pinned protocol; those files must not silently be treated as equivalent.
The workflow's complete identity check has not yet produced an outcome here.

## Checks actually completed

Local checks passed for the analytic conditional mean of a Gaussian, normalized
finite input-only component gates, a constructed two-regime regression positive
control, and the complete fit/select/refit/score software path on synthetic
48-sequence arrays. The latter exercised all model families and prescribed-node
preservation. These checks establish implementation behavior only; their toy
accuracy is not DEFORM evidence.

The workflow retains `identity_audit.json`, any `failure.json`, fitted-model
diagnostics, inner scores, frozen selections, per-case validation predictions,
and a final `result.json` when execution reaches those stages. A cache or runtime
failure must remain a technical failure, not a rejection of the DP hypothesis.

## Interpretation

A DP-specific claim requires improvement over the matched finite-mixture
comparators, not merely over one linear correction. Any successful comparison
here remains historical development evidence. No existing canonical result,
main-branch predictor, or manuscript claim was replaced by this pilot.
