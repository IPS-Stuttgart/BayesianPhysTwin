# Gaussian residual development v1

## Question and ownership

Does a nonlinear Matern-3/2 GP residual improve deformation forecasts over the
existing finite-feature ridge correction, using exactly the same initial states,
prescribed boundary motion, and frozen physical-neural backbone?

This is the user-authorized source-development experiment owned by
`protocols/gp_residual_development_v1.json`. It adds a diagnostic comparison,
not a stable backend, a promoted scientific claim, or a replacement for any
historical experiment. DEFORM source, checkpoint, feature construction, and
recorded evidence are unchanged. No robot or new physical experiment is needed.

## Implemented first step

`bayesian_phystwin_experiments.gaussian_residual_v1` supplies a zero-mean GP,
fit-only feature normalization, exact Gaussian conditioning on deterministic
recording-balanced support, predictive covariance, and a free-node readout
correction. The noise variance is explicit; no automatic jitter or eigenvalue
clipping is used. Zero correction returns the exact baseline array. Clamped
nodes are unchanged at every strength. This is array/readout behavior, not an
assertion of complete physical-belief compatibility.

The first comparison uses independent models per node and independent coordinate
outputs with shared kernel parameters. It is not yet the proposed graph-coupled
model, a force correction inside the simulator, or a joint ridge-plus-GP model.
It isolates nonlinear regression before adding those mechanisms.

To bound computation, the GP conditions on eight deterministic, evenly spaced
time samples per fit recording. This is exact conditioning on a subset, not a
sparse variational approximation to all observations. The MLP uses the identical
support rows. The existing ridge comparator uses all permitted fit rows. All
three receive the unchanged existing causal feature representation.

## Data and comparison

The retained DLO2 training result, source manifest and checkpoint are checked
against the identities in the earlier v6 protocol before loading any trajectory.
Only the original 40 fit and eight validation recordings may be opened. The
eight source-test recordings, official evaluation directories, and all other
DLOs are excluded. An audit hook denies non-allowlisted pickle payloads and
opens inside evaluation directories.

For each fixed seed (7, 19, 41), eight of the original 40 fit recordings form
inner validation. Nested budgets of 8, 16 and 32 recordings from the remainder
fit residual models. The original eight validation recordings form the outer
scoring panel. Hyperparameters and correction strength are selected only by
inner-validation L1. Selected forecasts and selections are saved before outer
scoring. Repeated seeds reuse the same outer recordings and are not independent
new test cases.

Reported arms are:

- frozen DEFORM physical-neural backbone;
- matched-budget existing ridge-1 correction with fixed strength 0.25;
- matched-budget existing ridge-1 correction with inner-selected strength;
- GP residual with inner-selected length scale, noise and strength;
- matched-support one-hidden-layer MLP with selected width and strength; and
- the existing ridge-1, strength-0.25 correction fitted on all 40 original fit
  recordings, explicitly a stronger-data reference.

The backbone includes DEFORM's own trained GCN correction; it is not bare rod
physics. At the two-observed-state prediction boundary, no later measured
free-node innovation is available for a nontrivial last-residual control. This
experiment does not introduce future observations to manufacture that control.

Primary reporting is equal-recording full-coordinate L1 in millimetres. Free-node
L1, per-recording wins and horizon-quartile errors are also retained. A later
uncertainty study must evaluate covariance and mean separately; GP mean parity
with matched kernel ridge is an implementation check, not a competitor to beat.

## Execution

From a source checkout with the retained upstream runtime and source files:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts/remote:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/florianpfaff/source-only/deform-bayesian-v1/venv/bin/python \
  scripts/remote/run_gp_residual_development_v1.py \
  --protocol protocols/gp_residual_development_v1.json \
  --output-root /path/to/new/gp-development-output \
  --device cuda:0
```

`residual-development.yml` is the maintained Actions entry point. It runs exact
revision numerical tests before the source-only job on `gpuserver4090`. It does
not install into or modify the frozen DEFORM environment. Raw source trajectories
remain in the runner-local output; the workflow uploads compact JSON records and
logs, not the trajectory archive.

Focused implementation checks:

```bash
PYTHONPATH=src:scripts/remote python -m pytest -q tests/test_gaussian_residual_v1.py
```

The scikit-learn parity test requires scikit-learn in the test environment.
It is optional for ordinary source tests and mandatory in the hosted workflow.

## Interpretation

The original checkpoint and validation recordings were already selected or
examined historically. This is retrospective method development even though the
new hyperparameter selection is nested. A positive result does not convert old
validation data into fresh confirmation. Training budgets measure residual
adaptation data, not all data used to construct the physical-neural backbone.

Implementation tests do not establish predictive improvement. GP covariance is
not asserted to be calibrated on the real data. Forecast correction does not
identify true stiffness, damping or contact parameters. No source-test or official
outcome may be opened to repair a negative development result. Every execution
retains either results or an explicit technical-failure record.

Paper-facing status and interpretation belong in
`FlorianPfaff/BayesianPhysTwin-Paper`; none of its existing manuscript claims are
changed by this implementation.
