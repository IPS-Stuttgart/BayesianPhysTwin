# Synthetic Fixed-Graph Benchmark

This benchmark supplies the ground truth required for parameter-recovery and
calibration claims. It uses a fixed path spring graph and jointly varies global
stiffness, graph damping, and known-action control scale:

```text
theta = [stiffness, damping, control_scale]
```

The simulator is deterministic given `theta` and the action. Inference uses the
first part of each trajectory and evaluates both fitted state and held-out
future state against the known trajectory.

## Corruption Conditions

- `clean`: Gaussian observation noise only
- `iid`: independent gross outliers with imperfect observable confidence
- `correlated`: contiguous occlusion, slowly accumulating track drift,
  boundary-localized noise, and a temporally coherent flow inconsistency

The correlated condition is the primary test of the structured-reliability
claim. Drift begins below the single-frame detection threshold and accumulates,
so pointwise robust losses cannot use later evidence to revise earlier inlier
decisions.

## Methods

| Method | Role |
|---|---|
| `unweighted_gaussian` | Deterministic equal-trust baseline |
| `huber` | Standard pointwise robust-loss baseline |
| `cue_weighted` | Heuristic confidence/covariance weighting |
| `iid_mixture` | Reliability-conditioned pointwise mixture |
| `markov_mixture` | Proposed temporally structured reliability |
| `oracle_covariates` | True corruption labels used as reliability priors |
| `oracle_inliers` | Upper bound that removes corrupted measurements |

The benchmark also compares `dynamic` and `quasi_static` actions. This ablation
measures the stiffness/damping/control-scale identifiability gained from
informative excitation.

## Metrics

- absolute error, 90% credible-interval coverage, and CRPS for each parameter
- state RMSE on fitted and held-out future frames
- state NEES and Gaussian 90% predictive coverage
- Brier score, log loss, ECE, and AUROC for cue prior, i.i.d. posterior, and
  Markov-smoothed inlier probability

## Registered Run

Run all conditions, both action modes, and 20 deterministic seeds:

```bash
bpt-synthetic-benchmark \
  --seeds 0:20 \
  --conditions clean,iid,correlated \
  --action-modes dynamic,quasi_static \
  --output-json runs/synthetic_v1/results.json \
  --output-csv runs/synthetic_v1/aggregate.csv
```

Use `scripts/remote/run_synthetic_benchmark.sh` to execute the same registered
run on a configured compute host. Record the exact code commit and host with the
curated paper artifact.
