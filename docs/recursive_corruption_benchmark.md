# Recursive corruption benchmark

## Purpose

The released PhysTwin comparison shows that `last_residual` is an excellent
point predictor. This benchmark asks the complementary question: **when does a
recursive uncertainty-bearing belief add value over residual persistence?**

It evaluates one-step forecasts on a deterministic physical rollout with a
latent autoregressive discrepancy. The observation stream is then subjected to
controlled corruption:

- a complete missing-observation burst;
- independent large outliers;
- gradually increasing coherent drift;
- substitution by a distractor identity;
- observations carrying stale source timestamps; and
- an 80% observation-density reduction.

Run the benchmark with:

```bash
bpt benchmark recursive-corruption \
  --seeds 0:50 \
  --output-json outputs/recursive-corruption/result.json \
  --output-csv outputs/recursive-corruption/records.csv
```

The mandatory runtime remains NumPy-only.

## Compared methods

The benchmark deliberately uses small controlled abstractions rather than
claiming to reproduce every production BayesianPhysTwin component:

1. `physical_baseline` ignores observations;
2. `last_residual` persists the latest measured residual;
3. `exponential_residual` smooths residuals with fixed exponential decay;
4. `recursive_gaussian` performs an unguarded scalar Gaussian update; and
5. `guarded_recursive` combines the same recursive belief with target-free
   reliability, source-timestamp validation, an innovation gate, a trust region,
   and exact prior fallback.

The guarded method does not use the generating truth to accept an update. Truth
is used only after forecasting to score the methods and count materially harmful
accepted updates.

## Metrics

Each condition and random seed reports:

- overall, pre-corruption, corruption-window, and recovery RMSE;
- maximum absolute error;
- recovery half-life in time steps;
- accepted-update and exact-fallback counts;
- materially harmful accepted updates relative to the same method's pre-update
  forecast;
- exact-fallback invariant violations;
- Gaussian NLL, nominal-90% coverage, and interval width for recursive methods;
  and
- fallback reasons, including missing, low-reliability, stale-lineage,
  innovation-gate, and trust-region rejection.

A materially harmful accepted update must worsen absolute one-step error by more
than the registered margin. This avoids counting immaterial sign changes caused
by observation noise.

## Diagnostic innovation-threshold curve

A repository-only diagnostic sweeps a predeclared set of maximum normalized
innovation-squared thresholds while leaving every non-guard method unchanged:

```bash
python scripts/science/analyze_recursive_corruption_selectivity_v1.py \
  --seeds 0:50 \
  --conditions \
    missing_burst,outlier_burst,coherent_drift,identity_switch,delayed_observation,density_drop \
  --maximum-nis-grid 1,2,4,9,16,36,1000000 \
  --output outputs/recursive-corruption/selectivity.json
```

Each curve point reports:

- acceptance and fallback fractions;
- deployed RMSE relative to `last_residual` and unguarded recursion;
- materially harmful accepted updates per accepted update;
- Gaussian NLL and cumulative per-sequence NLL regret relative to unguarded
  recursion;
- nominal-90% coverage and interval width;
- fallback reasons; and
- exact-fallback invariant violations.

The reference-method metrics are required to remain exactly identical across the
threshold grid. This catches accidental coupling between a guard parameter and a
nominal comparator. The report is content-addressed, finite JSON and refuses to
overwrite an existing result unless `--force` is supplied.

This sweep deliberately uses generated truth and does **not** select a threshold.
It is a retrospective mechanism diagnostic for understanding the
acceptance-versus-risk trade-off. A real or sealed study must freeze one guard
from permitted source/calibration evidence before target outcomes are opened.

## Exact fallback

For every rejected guarded update, the posterior mean and variance are assigned
from the exact pre-update recursive belief. The benchmark checks this invariant
at every time step and reports any violation. The invariant establishes routing
behavior only; it is not a deployment-safety theorem.

## Scientific boundary

This is controlled mechanism evidence. It does not exercise a real PhysTwin
backend, a Prob4D provider, a sealed physical object/session cohort, or a
Causal4D intervention. Passing it does not establish real-provider competence,
physical-object transfer, covariance calibration, intervention benefit,
deployment safety, or state of the art. Its role is to expose regimes in which
recursive uncertainty and selective fallback can be distinguished from simple
residual persistence.
