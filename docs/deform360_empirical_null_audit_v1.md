# Deform360 empirical-error null and matched-acceptance audit

## Question and execution

This is a retrospective analysis of the original 92-object Deform360 tactile-response panel, with the same five registered scalar queries and unchanged source/target episodes. It is not a fresh confirmation and does not open other sealed datasets. No physical experiments, active probing, camera pixels or geometry are required.

Run trigger: `.github/requests/deform360-empirical-null-audit-20260906.json`.
Workflow: `.github/workflows/deform360-empirical-null-audit-v1.yml`.
Runner: `[self-hosted, Linux, X64, gpuserver4090]`.
Initial trigger commit: `f03075b9179f9048e760e0c54350bd01b764e9bd`.
Initial Actions run: `33984701676`.

The unchanged historical predictor and reader run from pinned revisions. The original parent artifact and v6 artifact are verified by ZIP SHA256. Every original carrier fingerprint, source/target identity and point prediction must reproduce. The observer saves query-level source residuals and exact target probabilities for independent scoring without modifying historical model inputs or return values.

## Why the empirical control is essential

The original v6 scalar-query calibration computes

```
source_mse = mean((centered_source_errors @ query_weight)**2)
scale = max(source_mse / full_raw_query_variance, 1e-8)
full_calibrated_query_variance = full_raw_query_variance * scale
```

When the floor is inactive, the calibrated full variance equals `source_mse`, regardless of the original covariance. A direct source empirical Gaussian can then yield the same query probabilities, proper scores and decisions. Independently calibrating the diagonal control has the same consequence. This is tested numerically, not presumed from the method label. Floors and floating-point effects are checked through retained prediction arrays.

This null applies to the specific fixed scalar-query scoring scheme. It does not imply that all covariance representations are equivalent, that joint events are equivalent, or that query-independent transfer is impossible.

## Comparator separation

The rank-matched empirical covariance is a source-only PCA residual model with 10% diagonal shrinkage and factor rank matched to the original model. Its rows are rescaled to reproduce every original coordinate marginal. It uses the same shared scalar-query scale as the three historical covariance arms. This comparison isolates changes in correlation structure at matched means, coordinate marginals, factor rank and calibration policy. The fixed shrinkage is not tuned on target results and is not claimed to be an optimal conventional baseline.

The direct empirical-query Gaussian is a different, stronger prediction-level null. It shares the exact mean and source residual data, but is not claimed to preserve the original full-field coordinate marginals. It estimates each of the five required scalar variances directly. It is a comparator for these query outputs, not an alternative full-field distribution with all the same constraints.

## Endpoints and independent units

Brier score, query NLL, event log loss, 90% interval coverage/width, normalized query NEES, fixed-cost decision loss and acceptance are reported. Objects receive equal weight; five queries receive equal weight within objects. Frames and sensor cells are not treated as independent replicates.

For matched acceptance, each arm ranks the same pooled five-query/windows within each object by predicted adverse-event probability, with deterministic index-based tie-breaking. Every arm accepts exactly the same count at each requested fraction. This uses prediction ranks only, never target outcomes. It is an offline ranking evaluation, not an online threshold-validation claim.

The predeclared primary fraction is 40%; 10%, 20%, 60%, 80% and 100% are secondary. Paired bootstrap intervals resample 92 physical objects 10,000 times with seed 260906. Intervals for secondary endpoints are descriptive and not multiplicity-adjusted. Exact numerical ties are not interpreted as inferiority or superiority through roundoff.

The fixed decision loss uses an adverse accepted-event cost of 1 and a flag/fallback cost of 0.1. Always flagging therefore costs exactly 0.1 and must be acknowledged when interpreting absolute utility. It is not an equal-acceptance comparator.

## Evidence and limitations

The job uploads both complete and incomplete evidence, including console output, the exact executed source, request, environment and per-object NPZ prediction arrays. A green run establishes execution and parity checks; scientific conclusions must come from the result and comparator contrasts. No production, safety, fresh-generalization, calibrated-full-field, actual Prob4D-provider or unique-Bayesian claim is authorized merely by CI passing.
