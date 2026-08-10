# Proper scoring for predictive physical queries

`bpt evidence score` converts registered predictive distributions into the
existing decisive-evidence contract. The resulting losses can therefore be
passed directly to `bpt evidence summarize` and to the Bayesian-value
decomposition without introducing a second fallback, grouping, or risk-coverage
implementation.

## Input boundary

The input contract is
`bayesian-phystwin-proper-scoring-input-v1`. Every record binds one method to one
statistical unit, physical-query identity, horizon, observed query value,
predictive distribution, common physical fallback, and frozen deployment
decision.

A forecast may contain one or more of these representations:

- finite empirical samples for the multivariate energy score;
- a positive-definite Gaussian mean and covariance for the logarithmic score;
- nested central intervals and a median for scalar weighted interval score.

A variogram score is emitted only when empirical samples and an explicit set of
coordinate pairs are both present. Pair indices, powers, and weights are part of
the registered input rather than selected after outcomes are opened.

The converter fails closed unless every method on a unit uses the same truth,
query dimension, horizon, statistical group, variogram registration, and exact
fallback prediction. It also requires every unit of one query to expose the same
score families and every registered method.

## Scores

For empirical draws \(X_1,\ldots,X_S\) and outcome \(y\), the energy score is

\[
  \frac{1}{S}\sum_s \lVert X_s-y\rVert
  - \frac{1}{2S^2}\sum_{s,t}\lVert X_s-X_t\rVert.
\]

The implementation evaluates the pairwise term in bounded blocks and records
the finite empirical distribution exactly. It does not subsample draws.

For registered coordinate pairs \(P\), power \(p\), and positive weights
\(w_{ij}\), the variogram score is

\[
  \sum_{(i,j)\in P} w_{ij}
  \left(
    |y_i-y_j|^p - \frac{1}{S}\sum_s |X_{s,i}-X_{s,j}|^p
  \right)^2.
\]

The Gaussian score is exact negative log predictive density, evaluated with a
Cholesky factorization and no jitter, clipping, or pseudoinverse fallback. The
decisive-evidence contract requires nonnegative losses, while a continuous log
score can be negative in a fixed physical unit. The input therefore declares one
common `gaussian_log_score_offset`. The converter stores both the raw score and
the additive offset and rejects any record that remains negative. A common,
outcome-independent offset preserves score comparisons, but it must be frozen on
source or calibration data before target outcomes are opened.

For a scalar median and \(K\) nested central intervals, weighted interval score
uses the standard median weight `0.5` and interval weight `alpha / 2`, divided by
`K + 0.5`. The emitted decisive-evidence record also retains deployed interval
coverage and width observations.

## Exact fallback

Raw candidate and fallback scores are both calculated from their registered
predictive distributions. When `accepted=false`, `deployed_loss` and deployed
interval observations are constructed from the fallback. Callers do not supply a
second deployed value that could disagree with the fallback artifact.

## Command

```bash
bpt evidence score \
  runs/prospective/predictive-distributions.json \
  runs/prospective/proper-score-evidence.json

bpt evidence summarize \
  runs/prospective/proper-score-evidence.json \
  runs/prospective/proper-score-summary.json \
  --reference-method last_residual
```

Input JSON is read with duplicate-key and non-finite-number rejection, an
ordinary-file stability check, and a configurable byte budget. Output is
published atomically and without replacement unless `--overwrite` is explicit.
Sample counts, dimensions, array elements, energy pairs, and variogram
evaluations have independent resource limits.

## Scientific boundary

Proper scoring jointly exposes accuracy, sharpness, and distributional
calibration under a registered query and statistical unit. The implementation is
analysis infrastructure. It does not establish calibrated uncertainty,
independent-object transfer, physical-query improvement, intervention benefit,
deployment safety, or state of the art. Those claims require separately frozen
source/calibration choices and unopened target evidence.
