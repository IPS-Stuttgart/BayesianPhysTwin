# DEFORM GP residual development pilot

Draft implementation: PR #944, branch `research/deform-gp-residual-dev-v1`.
Entry point: `experiments/deform_gp_residual_dev_v1/gp_residual_pilot.py`.
Workflow: `.github/workflows/deform-gp-residual-dev-v1.yml`.

## Question and method

Does a nonlinear residual predictor improve on the existing DEFORM local ridge
residual using exactly the same recorded development inputs and cached hybrid
simulator predictions?

The current pilot fits independent node-wise, finite-rank Gaussian processes
using a whitened Nystrom Matern-3/2 basis, 128 fit-only inducing locations, and
three coordinate outputs. It reuses the existing action-local causal feature
builder, duplicate-query collapse, and exact clamped-node baseline values.
This is a GP posterior-mean experiment, equivalently kernel ridge regression
for the specified approximate kernel. It does not establish a uniquely Bayesian
accuracy benefit, covariance calibration, graph coupling, or physical-parameter
identification.

The comparison contains the unchanged hybrid, the existing ridge=1 residual
with shrinkage 0.1/0.25/0.5/1.0, and GP length scales 0.5/1/2, likelihood noise
parameters 0.1/1, and the same four shrinkages (24 GP settings). The incumbent
is ridge=1 with shrinkage=0.25. Report trajectory-level coordinate L1, individual
case errors, wins/losses, and all settings, not only the best GP score. The larger
GP selection budget must remain visible.

## Information boundary

Only explicitly named `fit.npz` and `validation.npz` development caches are
loaded. Cache identity order must match the retained manifest and is checked
against the separate source-test roster. The current loader expects the
retained 32-fit/12-validation split and fails rather than silently resplitting
when it differs. No source-test, official evaluation, sealed confirmation,
new robot actions, or new hardware collection is authorized by this pilot.
Selection on these already-open validation trajectories is exploratory and
cannot be described as fresh confirmation. Artifact hashes are retained;
artifact retention by itself is not predictive evidence.

## Numerical checks and execution status

An independent local reproduction of the pilot's algebra passed dense-GP mean
parity, exact repeatability, and constant-response checks on synthetic arrays.
The maximum absolute difference from dense GP conditioning was
`1.1605619310017801e-07` (tolerance `2e-5`; NumPy 2.3.5, SciPy 1.17.0).
These are numerical checks, not a real-data performance result.

At this documentation update, the real-data Actions comparison had been queued
on `gpuserver4090`; no real-data score had been obtained in this session.
Do not infer GP superiority or infer a negative scientific result from a queued
or technically failed run. A completed `GP_PILOT_RESULT` / `result.json` is needed
before making a performance statement.

The unused raw-rollout adapter was removed; the retained experiment uses the
existing development caches instead of retraining or recomputing the simulator.
