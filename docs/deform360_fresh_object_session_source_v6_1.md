# Deform360 v6.1 nested source contract

## Status

This source-independent repair supersedes the v6.0 source evaluator before any
v6 challenger suffix outcome or v6 target payload was opened. The original
prediction archives remain immutable provenance, but their embedded acceptance
and interval summaries are diagnostic only and cannot advance the study.

The repair is frozen at
`protocols/amendments/deform360_official_hub_fresh_object_session_v6_nested_source_contract_repair.json`.
It does not change the ten source object-sessions, six registered variants,
source suffixes, target cohort, point predictions, target gate, or claim scope.

## Why the repair is necessary

The base v6 policy requires candidate-specific guard thresholds, covariance
selection, and source-only interval calibration inside each nine-object outer
training fold. The legacy adapter instead consumed
precomputed acceptance decisions and calibrated interval summaries from its
prediction records. That could not produce the registered nested estimate.

V6.1 separates the stages:

1. Before suffix scoring, seal 100 raw records: every outer-held-out object by
   every scored source object. Each challenger record binds its exact fit roster,
   point-prediction identity, raw covariance identity, and residual-independent
   risk score. No acceptance, threshold, coverage, width, or proper score is
   permitted.
2. After the atomic 100-record barrier, attach only raw scoring sufficient
   statistics for the exact bound predictions.
3. In each outer fold, fit guard thresholds and covariance calibration from the
   other nine objects, jointly select a candidate/covariance variant there, and
   score the untouched tenth object once.
4. Rejections and unavailable covariance variants use the exact physical point
   prediction and the separately calibrated physical interval.

## Calibration arithmetic

For raw covariance `Sigma`, v6.1 fits a positive scalar `s` by the equal-object
three-dimensional Gaussian-NLL minimizer,

```text
s = clip(mean Mahalanobis^2 / 3, 1e-6, 1e6).
```

The object-session nonconformity score is the maximum query Mahalanobis norm.
At nominal 90% coverage, each nine-object outer training fold uses rank 9 of 9.
The final ten-object source fit uses rank 10 of 10 leave-one-object-out
residuals. Dense queries never increase the calibration sample size.

This is **cross-fitted grouped residual calibration**, not an exact finite-sample
split-conformal guarantee. The same outer-training residual set estimates the
scalar and its grouped quantile, and the fitted model size differs between the
nested residuals and the final full-source model. The registered held-out source
coverage check is therefore empirical; no marginal exchangeability guarantee is
claimed for fresh objects. This source-independent repair explicitly supersedes
the base policy's `group-clustered-split-conformal` label and the covariance
amendment's `source_only_variance_scale_and_split_conformal_required` field;
retaining either label would overstate what this arithmetic guarantees.

## Guard and selection

Each inclusive risk threshold must accept at least eight of the nine outer-fold
training units and may include at most one update worse than the physical
fallback by more than 2%. Complete tied-risk blocks are accepted together.
Thresholds are ranked by deployed point loss, harmful updates, acceptance, then
the lower threshold.

Candidate/covariance variants must improve deployed point loss over last causal
residual by at least 2%, not regress calibrated Gaussian NLL, and pass the guard.
Following the later covariance amendment, eligible variants are ranked by
Gaussian NLL, deployed point loss, full interval width, then frozen complexity.
The same exact variant must win at least eight outer folds for the source gate to
pass.

## Command boundary

The only progressing command is:

```bash
python scripts/science/run_deform360_fresh_object_session_source_v6_1.py --help
```

Every subcommand requires the exact base policy, covariance amendment, nested
repair, and ten-unit source selection. `seal-batch` requires exactly 100 raw
prediction files. `assemble` requires exactly 100 outcome files and can run only
after that batch exists. `evaluate` can authorize a full-source fit, but always
leaves fresh-target selection, target payload access, and claims false.

The v6.0 CLI remains readable for provenance, but its public evaluator now fails
closed with a retirement error.

## Scientific boundary

This repair uses no v6 challenger outcome, v5 confirmation payload or outcome,
v6 target payload or outcome, human selection, or replacement. Prob4D is used
only through the already sealed decoded-uniform observation artifacts in the
upstream source batch; no new Prob4D or MotionCrafter inference is introduced by
this source-contract repair. A passing source gate would justify only the next
pre-registered source-to-fresh confirmation stage, not a state-of-the-art claim.
