# DP-gated residual experts: DLO1 development v1

This is an executable, retrospective development experiment, not a new official
DEFORM evaluation, a calibration result, or an approved paper claim. It requires
no new hardware data, active probing, or robot experiment.

## Question

Can input-predictable residual regimes improve the existing local residual
correction, and does a Dirichlet-process (DP) mixture add value beyond a finite
mixture or a single nonlinear regressor?

## Methods

`experiments/dp_residual_development_v1/model.py` implements a two-stage prototype:

1. Fit source-only feature and residual embeddings (PCA ranks 6 and 4).
2. Fit a joint variational DP Gaussian mixture over those embeddings, truncated
   at eight components. A single gate is shared across nodes at each time.
3. Fit responsibility-weighted per-node ridge experts in the existing canonical
   feature representation, with ridge 1.
4. At prediction time, marginalize unknown future residuals out of the joint
   mixture. Expert weights may depend only on allowed query features.

This is **DP-gated ridge regression**, not a fully integrated Bayesian DP
regression posterior. It does not implement sticky-HDP temporal switching,
correlated observation-error inference, or latent physical-state corrections.
The DP component count is a predictive representation, not an identified number
of physical modes. The eight-component cap is a numerical truncation.

Matched controls use the same features, weighted ridge experts, PCA ranks,
random seed, likelihood regularization, and shrinkage grid:

- the source-closed DEFORM rod-plus-GCN hybrid without an extra correction;
- the current ridge mean with fixed shrinkage 0.5;
- the same ridge with inner-selected shrinkage;
- a single nonlinear ridge with 64 random Fourier features plus linear features;
- finite Gaussian-mixture gates with 2, 4, or 8 components;
- finite Bayesian mixture gates with 2, 4, or 8 components and total Dirichlet
  concentration 1;
- DP gates with concentration 0.1, 1, or 10 and truncation eight.

Shrinkage is chosen from 0, 0.25, 0.5, and 1. A zero correction retains exact
backbone values. Both finite mixture families are included so a DP improvement
cannot be attributed solely to having multiple experts or using variational
rather than maximum-likelihood estimation.

## Data and selection

Only the existing DLO1 `train` partition's 40 `fit` and eight `validation`
trajectories are opened. Within the 40 fitting trajectories, every fifth entry
of the frozen manifest is used as an inner validation set (32/8 split).
Hyperparameters and shrinkage are selected there, not on the eight outer
validation outcomes. Each selected family is refitted on all 40 fitting
trajectories and its eight validation forecasts are sealed before scoring.

The backbone remains the exact selected update-6400 checkpoint. It was
historically selected on this development validation panel; the inner split
also overlaps backbone training. Consequently this is development evidence,
not independent prospective confirmation, even though the new expert selection
is nested. No source-test or official target is repurposed as fresh evidence.

The information contract is unchanged: two observed states, known future
clamped inputs, and the frozen backbone rollout. Future free-node truth is
permitted for fitting or scoring, never for query features or expert selection.

The runner verifies the source manifest, checkpoint source record, long-run
protocol, upstream code, and checkpoint hashes. Audit hooks block DLO1
`source_test`, DLO1 official `eval`, and DLO2--DLO5 payloads. Duplicate causal
queries fail closed rather than being split across fitting and validation.

## Checks and outputs

The runner checks a controlled observable-regime case, normalized expert
weights, deterministic query prediction, response marginalization, and finite
input rejection. It checks that replacing future free-node truth cannot alter
query features, that clamped nodes remain exact, that the existing production
ridge mean is reproduced within 1e-7 m, and that the frozen backbone validation
mean is reproduced within the registered tolerance.

Primary metric: trajectory-balanced mean coordinate L1, in millimetres, over the
same full-coordinate forecast operator. Secondary diagnostics include coordinate
RMSE, horizon thirds, paired trajectory errors, learned component weights, fit
time, and convergence warnings. Eight-trajectory bootstrap intervals are
explicitly descriptive. No covariance calibration claim is evaluated.

Artifacts contain the protocol, controlled checks, development arrays, all inner
candidate scores, selection seal, fitted parameter arrays without pickle,
sealed predictions, prediction hashes, and final `result.json`. The paper
repository owns final result interpretation; this repository owns implementation
and executable boundaries.

## Execution

The isolated workflow `.github/workflows/dp-residual-development-v1.yml` runs on
`gpuserver4090` using the existing frozen DEFORM Python/Torch environment. It
writes new artifacts only and does not alter datasets, installed packages, or
existing scientific records.

```bash
PYTHONPATH="$PWD/src:$PWD/scripts/remote:$PWD" \
  /home/florianpfaff/source-only/deform-bayesian-v1/venv/bin/python \
  experiments/dp_residual_development_v1/run.py \
  --output outputs/dp-residual-development-v1
```

The synthetic mechanism test can also run without DEFORM data:

```bash
python experiments/dp_residual_development_v1/model.py
```

Local development found that an overly aggressive three-dimensional feature
embedding erased the synthetic regime cue. The retained controlled test uses
all six feature dimensions and verifies that both finite and DP mixtures can
exploit an observable cue. This is an implementation check, not a DP-specific
novelty or real-data result.
