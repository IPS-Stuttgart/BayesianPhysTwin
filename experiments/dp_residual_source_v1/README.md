# Sticky HDP residual dynamics: source-development feasibility

This isolated experiment tests a sticky weak-limit HDP autoregressive residual
model around the frozen DEFORM hybrid and the existing BayesianPhysTwin local
residual correction. It does not change the library, historical protocols,
main branch, paper claims, or data-access authorizations.

## Data and information boundary

Read-only reuse of the exact DLO1 development artifact from run `34052781162`,
preparation commit `9b49a0bac80e698d508c597794d70d007b6f6b87`, bundle SHA256
`70894781f89351bb25fc3b53f7b1453d9e6b36aeebc03caa7cb341ac4023ad3d`.
The bundle contains 40 historical fit and 8 historical validation recordings.
No DLO1 source-test, official evaluation, other DLO, or Deform360 payload is
opened. The preparation job belongs to a separate development branch and is
reused, not rerun or modified by this experiment.

This is retrospective development, not independent confirmation. The fixed
hybrid checkpoint was previously trained/selected on these source partitions;
inner splitting newly refits only the local residual, representation and AR
models. Previously inspected validation outcomes do not become fresh evidence.

The new task exposes an observed prefix and forecasts its recorded suffix.
It is NOT the original two-observed-state, 498-frame DEFORM benchmark and its
numbers must not be substituted into that benchmark's result tables.

## Fixed comparison

`run.py:PROTOCOL` binds all choices before loading the bundle. An inner 32/8
source-fit split selects finite K from 2, 4 and 8. Final models refit on 40 fit
recordings and score the historical 8-recording validation panel. Horizons are
10 and 50 frames, with 50 primary, at residual-index origins 29, 99, 199, 299.
The metric is recording-balanced coordinate L1 on nine free nodes. Clamped
nodes are preserved exactly. All assimilation arms use the same prefix and
known baseline/control future. Six source-fitted residual PCA dimensions model
the dynamics; the orthogonal observed residual is persisted identically.

Controls: original hybrid, existing local residual (ridge 1, shrinkage .5),
last-residual persistence around each, damped residual velocity, pooled Bayesian
AR(2), validation-selected finite sticky AR-HMM, and a hard-regime ablation.
The hard-regime ablation still averages coefficient draws; it is not a full
point-parameter ablation.

## Inference

The HDP implementation uses a finite weak-limit approximation with cap 8,
shared random global weights, sticky transition rows, auxiliary restaurant
tables with diagonal sticky-table thinning, and matrix-normal/inverse-Wishart
regression updates. State sequences are sampled with FFBS without transitions
across recordings. Finite controls keep global weights uniform and otherwise
use matched emissions, priors and compute. Initial-state probabilities have a
separate symmetric Dirichlet prior in all arms.

Two fixed seeds, 60 Gibbs iterations, burn 30, thin 5 yield 12 retained draws.
This is a feasibility sampler, not a convergence-certified infinite-HDP
posterior. Saturating the cap is reported explicitly. Forecasts average source
parameter-posterior draws equally while conditioning regime probabilities on
the prefix; the prefix does not importance-reweight the parameter draws.
Conditional switching-linear first moments are propagated exactly for each
draw. No covariance-calibration claim is made. Coherent predictive sampling is
implemented and tested but the primary experiment scores point forecasts.

Fourteen unit tests cover exact HMM filtering against enumeration, AR recursion,
Monte Carlo versus analytic predictive means, label invariance, positive
covariances, reproducibility, sticky-table correction, record boundaries,
future-suffix replacement and clamped-node preservation. Passing tests alone
is not evidence of improved physical prediction.

## Run

```bash
python -m pip install numpy==1.26.4 scipy==1.15.3
export PYTHONPATH="$PWD/src:$PWD/experiments/dp_residual_source_v1"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s experiments/dp_residual_source_v1 -p 'test_*.py' -v
python experiments/dp_residual_source_v1/run.py \
  --bundle /path/to/development.npz \
  --preparation /path/to/preparation.json \
  --output outputs/dp-residual-source-v1
```

Output must be new. Results retain protocol, source identity, selected finite K,
per-recording and paired-bootstrap scores, posterior draws, sampler traces,
forecasts, fitted representation and artifact hashes. Positive and negative
outcomes are retained. No result authorizes a sealed target or a paper claim.
