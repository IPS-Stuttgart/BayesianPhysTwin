# Residual Replay

Residual replay is the first integration boundary with PhysTwin. It evaluates
completed reconstructions without changing the original optimizer and produces
row-level scores suitable for plots and aggregate JSON suitable for result
tables.

## Statistical Model

For pseudo-measurement `i`, cue-derived reliability `r_i` is a prior inlier
probability:

```text
p(y_i | x, r_i) =
    r_i       Normal(y_i; h_i(x), Sigma_obs_i + Sigma_model)
  + (1 - r_i) Normal(y_i; h_i(x), k (Sigma_obs_i + Sigma_model))
```

The second component is a broad outlier model, with `k = 100` by default. The
posterior inlier probability additionally conditions on the residual. Residuals
are deliberately excluded from prior reliability unless `--residual-scale` is
set for the residual-gated baseline. Additive model-discrepancy variance is a
separate CLI/config value so physics mismatch is not folded into perception
confidence.

When `track_id` and `frame` are present, replay also places a binary Markov
model over each track's inlier state. Cue reliability remains the observable
unary evidence, while `--inlier-persistence` and `--outlier-persistence`
control temporal coupling. The implementation normalizes the cue-conditioned
Markov prior, so the resulting sequence evidence can be used for parameter
inference rather than only as a smoothing heuristic.

## Canonical CSV

Each row represents one pseudo-measurement. Required vector columns use either
of these matched naming conventions:

```text
observed_x, observed_y, observed_z, predicted_x, predicted_y, predicted_z
obs_x, obs_y, obs_z, pred_x, pred_y, pred_z
```

Numeric suffixes such as `observed_0`/`predicted_0` are also accepted. Optional
columns are:

| Column | Meaning | Missing default |
|---|---|---|
| `variance` | Scalar variance for all coordinates in that row | CLI `--default-variance` |
| `variance_x`, ... or `var_x`, ... | Per-coordinate variance | CLI `--default-variance` |
| `confidence` | Learned cue confidence in `[0, 1]` | `1` |
| `occluded` | Boolean visibility flag | `false` |
| `boundary_distance` | Distance to segmentation boundary | infinity |
| `flow_inconsistency` | Nonnegative 4D-flow inconsistency | `0` |
| `is_inlier` | Optional ground-truth calibration label | no calibration metrics |
| `is_corrupted` | Inverse alias for `is_inlier` | no calibration metrics |
| `track_id` | Optional persistent sequence identifier | no Markov smoothing |
| `frame` | Optional frame identifier | no Markov smoothing or per-frame summary |

All other provenance columns, for example `track_id`, `source`, object, camera,
or run identifiers, are preserved in the scored CSV.

## Run

```bash
bpt residual replay examples/residuals_demo.csv \
  --summary-json outputs/residuals_demo/summary.json \
  --scored-csv outputs/residuals_demo/scored.csv
```

Without an editable install, use:

```bash
PYTHONPATH=src python3 -m bayesian_phystwin.cli.residual_replay \
  examples/residuals_demo.csv
```

The summary contains residual, prior-reliability, and posterior-inlier
distributions; i.i.d. and Markov-smoothed inlier probabilities; effective
sample sizes; unweighted, covariance-weighted, and robust objectives; per-frame
summaries; and Brier score, log loss, expected calibration error, and AUROC when
labels are present.
