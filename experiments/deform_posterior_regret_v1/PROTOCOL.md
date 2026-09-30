# Posterior Regret Admission V1

Classification: prospective method experiment on historically exposed recorded
public data. Owning admission basis: the user-requested two-gate test of whether
posterior uncertainty improves admission of an unchanged forecast update.
No production default, prior workflow, GP artifact, or reserved cohort changes.

## Frozen Forecasts And Mathematics

The finite-rank Matérn-3/2 implementation is copied from
`research/deform-gp-residual-trial-v1` at
`1b0622c823d6d927ab9d4863911caab67e98ebaf`, with formatting and a module-level
immutable default only. Feature construction and the native recorded DEFORM
replays are reused through the checksum-bound query cache, not recomputed using
future free-node observations. Configuration: ridge 1, amplitude 3, length
multiplier 1, 128 recording-balanced anchors, seed 20260907, shrinkage 0.25.
Standardization, anchors and weights are refitted in every training partition.

For every context there are exactly two forecasts: B is open-loop GP; C is
its prefix-conditioned mean. There are no alternative point forecasts, camera
tracks, actions, interventions, acquisition choices or simulated robot claims.
Forecast-index endpoints are 50, 150, 250 (zero-based); observed indices are
endpoint-46, endpoint-41, ..., endpoint-1. Horizon 100 is primary; 25 and 200
are secondary. Score eight free nodes and three coordinates only. Four known
future clamped-node trajectories are permitted inputs and remain byte-exact.
Metre-valued arrays are used throughout; regret is reported in mm^2.

For d scored coordinates and Delta=C-B, squared-loss regret is

    R = (||C||^2 - ||B||^2 - 2 Delta'Y)/d.

Under Y~N(mu,K), its mean is the same expression with mu and its variance is
4 Delta'K Delta/d^2. Harm means R>1 mm^2. Expected regret depends only on mu;
with mu=C it equals -||Delta||^2/d. These are established identities, not a
new theorem. The tested Bayesian benefit is harm probability/tail risk, not
an automatic improvement of squared-loss expectation.

Recover full temporal Schur complements, not only conditional diagonals.
The GP explicitly assumes independent node/coordinate blocks in the fixed
orthonormal local frame. Covariance family is the original follow-up's
q_coordinate*(a*shape/mean_diagonal+b*I), fitted by joint likelihood at 20
uniform time indices. The existing q floor (0.01 mm)^2 and bounds
1e-6<=a,b<=1e3 are retained. No covariance clipping, extra jitter, pseudoinverse
or silent repair. Invalid covariance retains a technical-failure context and
forces the unchanged B fallback for every arm. Gate passage also requires
no primary technical failures, preventing failure-driven selective omission.

## Source Custody

Only DLO2's 40 fit recordings in `fit_queries.npz` may be opened. Bind:

- Cache manifest SHA256:
  ff24ae708753febd0282c77c3f310a7ead21e981f2dd8464ee35ed0bb8d8c2dc
- Fit cache SHA256:
  b99f34a577f8616c4873df846866a01edd44dc9afd37cda68eeee3330a18cfa8
- Export run 34258802512, artifact 10069073833.

The archive also contains historical validation bytes. Extract only the two
named files; never deserialize validation queries/predictions, original source
test recordings, or official evaluation for this study.

Outer folds: seeded permutation (20260907) of the manifest's 40-recording order,
five equal groups of eight. Each outer fit has four eight-recording inner folds,
with permutation seed 20260907+100+outer_fold. Each inner training set of 24
has three further eight-recording cross-fit groups, seeded
20260907+1000+10*outer_fold+inner_fold. Thus an inner held recording contributes
neither GP, normalization, anchors, covariance nor empirical-risk fitting.
OOF errors calibrate covariances. Inner panels choose control settings and fit
positive-slope probability calibration. Every outer held recording is excluded
from its entire learner. A recording can train other outer folds, as usual.
The frozen physical simulator historically saw source recordings; cross-fitting
does NOT remove that upstream training exposure.

## Controls

All arms receive identical B/C, ten recorded free-node prefix observations,
known future clamp trajectories, preprocessing and permitted source outcomes.
Covariance controls are centered on the same C. Empirical predictive laws may
learn a conditional bias from source outcomes without changing B or C.

- Posterior risk: full calibrated conditional GP covariance. Also retain a raw
  uncalibrated-covariance diagnostic and raw/calibrated probability scores.
- Expected regret: identical mean, no covariance; rank by continuous expected R.
- Constant diagonal: same source-OOF coordinate noise, independent time.
- Trajectory-cluster sandwich: full source sandwich covariance, separately
  calibrated by the same covariance family, conditioned on the same prefix.
- Whole-trajectory conditional residual bootstrap: select whole aligned source
  recording residual trajectories by prefix/forecast/boundary neighborhood.
  Transport residuals, then evaluate fixed B/C regret on every draw. Never
  resample coordinates or time steps independently.
- Direct empirical regret: neighborhood distribution of actual source regrets.
- Ridge/kernel regret predictors: train on actual source regret. Recording-OOF
  prediction errors give their empirical predictive regret CDF.
- Matched predictive distribution: copied mean and covariance through the same
  interface; must produce numerically identical risks and selections. This is
  a parity control, not an independent superiority comparator.

Empirical input vector: ten-frame local innovations; per-node mean and RMS
forecast difference; per-node mean and RMS B; total clamp displacement and RMS
clamp velocity; normalized endpoint and horizon. No future free-node outcome.
Ridge settings .1,1,10,100,1000; kernel RBF lengths .5,1,2,4 with ridge 1;
neighbors 4,8,16,32. Inner recording validation selects lowest Brier, with
setting-value ties deterministic. Objectives average the three endpoints and
three registered horizons equally. Empirical finite-sample CDFs use Jeffreys
continuity correction. All arms get the same positive-slope logistic probability
calibration, penalized 0.01, log-slope bounds [-5,5], intercept [-10,10]; raw
scores remain separately reported. Calibration consumes inner-OOF probabilities,
not control resubstitution. Ridge/kernel CDF errors are four-recording-fold OOF.

Primary acceptance is floor(0.5*N) contexts using predictions alone. Lexical
context IDs break ties. Invalid rows cannot be selected and do not shrink N.
Seal pairs, risks, fit lineage and selected context IDs before held scoring.
Report each comparator at coverages .1,.25,.5,.75,1; harmful counts, Brier,
deployed coordinate RMSE, exact fallback and coordinate coverage/NEES/width.
Report complete-recording outcomes; do not claim coordinate/point independence.

## Source Gate And Stop

At horizon 100, require >=10 beneficial (R<0) and >=10 materially harmful
(R>1) contexts; >=20% fewer harmful accepted corrections than the strongest
independent source comparator at 50%; and >=.005 Brier improvement. The
strongest control minimizes harm, then Brier, then lexical arm name. A zero-harm
control has no remaining harm headroom and the superiority gate fails.
These are discovery thresholds, not significance tests. Failure closes this
Bayesian-risk direction. No retuning, alternative horizon rescue or replication.

## Conditional Retrospective Replication

Only after source passage, lock the source result, code digest, comparator
family and primary comparator before DLO4/DLO5 processing. Reuse the existing
32/12/12 train-record split with domain
`conditional-query-posterior-v1-20260906`, ordering ascending SHA256 of
domain+'/'+object+'/'+basename, from `deform_dp_residual_v1/reference.py`.
Bind all recording and native DEFORM checkpoint hashes before execution.
Use retained official DEFORM checkpoints; no physical retraining or official
14-recording eval access. This is a locked RETROSPECTIVE replication, not
fresh-object confirmation. Historical native-simulator exposure remains.

For each object, fit the GP on 32 fitting recordings; source-only nested
cross-fitting fits risk controls; the 12 calibration recordings fit covariance,
probability calibration and a deployable threshold. Test 12 recordings only
after joint pair/risk/selection seals for both objects. Rank exactly 50% within
each object. Source-calibration threshold is the 50% order statistic; probability
ties follow the same lexical key rule and actual test acceptance is reported.
Retain technical failures, no replacement. Every comparator is reported.

Resample 12 complete test recordings within each of two fixed objects, 10,000
times with seed 20260907; retain all endpoints/horizons and paired arms together.
Require >=20% harm reduction, paired one-sided 97.5% upper harm-difference bound
<0; one-sided 97.5% upper deployed RMSE ratio <=1.01; favorable harm differences
on both objects; improved Brier. Do not claim superiority over a stronger
alternative beating the posterior. No follow-up target-tuned calibration.

## Verification And Interpretation

Test suffix mutation, identical forecasts, clamps, covariance-only risk changes,
analytic moments/Monte Carlo, equivalent-distribution parity, deterministic ties,
invalid uncertainty and exact fallback. Publish a compact paper-side evidence
bundle with runtime, input/code/selection seals, per-recording outcomes, raw
and calibrated risk, calibration diagnostics, curves and both gate decisions.
The main manuscript direction is admitted only if both gates pass. Otherwise
retain the negative and the separate bounded residual-adaptation evidence.
No camera competence, robot safety, acquisition or intervention selection,
population-wide object generalization, Prob4D or held-v8 claim follows.

Run after a clean immutable commit:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m \
  experiments.deform_posterior_regret_v1.study \
  --cache /path/to/fit-only-cache --output /new/run/path \
  --freeze-commit EXACT_PRE_OUTCOME_COMMIT
```
