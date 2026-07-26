# Selective guard evaluation

A Bayesian-PhysTwin guard must be evaluated against the exact physical baseline
returned on rejection. Average candidate accuracy alone does not establish that
the deployed system accepts updates safely.

For evaluation unit `i`, define paired excess loss

```text
Delta_i = candidate_loss_i - baseline_loss_i.
```

Negative values improve on the fallback. Positive values are harmful unless they
remain within a predeclared numerical or practical tolerance.

`bayesian_phystwin.selective_risk` is a NumPy-only evidence module. It reports:

- a fixed guard decision and its exact fallback behavior;
- a tie-preserving risk-coverage curve including the zero-coverage endpoint;
- mean, high-quantile, and worst accepted regression;
- harmful accepted-update frequency;
- cluster-bootstrap uncertainty at the declared statistical unit;
- conditional guard performance by frozen horizon, reliability, or identifiable
  rank strata;
- matched guarded comparisons against simple methods using the same fallback;
- predictive interval coverage and width by prediction horizon.

## Fixed guard metrics

`evaluate_guard(...)` reports coverage, fallback rate, baseline loss, unguarded
candidate loss, guarded selected-system loss, and candidate-minus-baseline excess.
Accepted-only diagnostics include the mean excess, a configurable upper quantile,
the harmful-update rate, and the worst excess.

The guarded selected loss is

```text
selected_i = candidate_i  when accepted_i
             baseline_i   otherwise.
```

Consequently, zero acceptance must reproduce the baseline mean exactly and yield
zero selected-system excess loss. Accepted-only metrics are undefined rather than
silently set to zero when no row is accepted.

The default upper-tail statistic is the 95th percentile of accepted excess loss.
The quantile and harmful tolerance are part of the experiment contract and must
not be chosen after target outcomes are opened.

## Risk-coverage curves

`selective_risk_curve(...)` evaluates every unique score threshold. Equal scores
enter together, so the curve does not depend on arbitrary row ordering. The first
point rejects every candidate and represents the exact physical fallback.
`higher_is_safer=False` supports uncertainty or risk scores for which smaller
values imply greater confidence.

A diagnostic curve may inspect all thresholds. A confirmatory deployment threshold
must be fixed from source or calibration data before target outcomes are opened.
The best target threshold on the curve is not a valid deployment claim.

## Matched Bayesian and simple comparators

`evaluate_matched_guards(...)` evaluates every method with the same baseline loss,
scientific endpoint, harmful tolerance, and upper-tail quantile. Each method may
have its own predeclared guard, but every rejected row receives the same exact
fallback.

This is the required comparison for Bayesian anchoring versus, for example,
last-residual persistence. Do not compare a guarded Bayesian system against an
unguarded simple candidate: that changes both the intervention policy and the
fallback treatment.

```python
comparison = evaluate_matched_guards(
    baseline_loss=physical_baseline_loss,
    candidate_losses={
        "bayesian": bayesian_loss,
        "last_residual": last_residual_loss,
    },
    accepted_by_method={
        "bayesian": bayesian_guard,
        "last_residual": last_residual_guard,
    },
    reference_method="bayesian",
)
```

## Conditional decision quality

`evaluate_guard_by_stratum(...)` exposes failure modes hidden by one cohort mean.
Use it separately with strata frozen from source or calibration data, including:

- prediction-horizon bands;
- reliability-score bins;
- identifiable-rank categories;
- sensor, object, or acquisition-session categories named in the protocol.

Do not choose bin edges from target outcomes. Report the row count, coverage,
harmful-update frequency, upper-tail regression, and fallback rate for every
stratum, including strata in which the method accepts nothing.

## Prediction-interval calibration by horizon

`evaluate_prediction_intervals(...)` reports empirical coverage, coverage error,
mean and median width, 90th-percentile width, and lower/upper miss rates for a
scalar predictive quantity.

`evaluate_prediction_intervals_by_horizon(...)` reports the same quantities at
each declared horizon. Horizons must be nonnegative and should be supplied in the
physical unit stated by the protocol. For continuous horizons, freeze bins on
source or calibration data and pass the corresponding bin values.

Coverage without width can be made vacuous by very broad intervals; width without
coverage can reward overconfidence. Both must be shown together. For a selective
system, report candidate calibration on accepted rows and state the acceptance
count at every horizon. Do not reinterpret exact fallback as a Bayesian interval
unless the fallback itself has a separately specified uncertainty model.

## Statistical units and bootstrap uncertainty

Rows from one interaction, physical object, acquisition session, overlapping
window family, or shared backbone realization are generally dependent.
`bootstrap_guard_evaluation(...)` samples the supplied `group_ids` with
replacement and retains every row from each sampled group. Repeated groups are
duplicated as complete clusters.

Replicates containing no accepted row remain valid for coverage, fallback, and
selected-system loss. Their accepted-only statistics are missing and excluded
from the corresponding percentile interval; every interval reports the number of
finite replicates explicitly.

Use the paper's declared statistical unit as `group_ids`. Do not substitute
coordinate-level or pixel-level groups merely to obtain narrower intervals.

## Prospective Prob4D to Bayesian-PhysTwin experiment

The intended prospective comparison has a source/calibration/target split at the
object or acquisition-session level. Freeze all choices before opening target
outcomes. The minimum arms are:

| Arm | Observation treatment |
|---|---|
| B0 | unchanged physical baseline |
| B1 | simple visual or last-residual interface |
| P1 | Prob4D fused observations with gauge marginalized |
| P2 | Prob4D unfused factors with explicit gauge nuisance |
| P3 | P2 plus an independently calibrated metric anchor |

Every candidate arm receives a source-frozen guard and the same B0 fallback.
Primary reporting consists of paired held-out physical-prediction loss and the
maximum or high-quantile regression relative to B0. Secondary reporting includes
coverage, harmful-update rate, exact fallback frequency, interval calibration and
width by horizon, identifiable rank, retained gauge-covariance trace, and
reliability strata.

The target split must not be used merely to verify that the pipeline runs. If a
source competence gate fails, improve the source-side method or register a new
physical cohort rather than tuning on a sealed target phase.

## Minimal example

```python
from bayesian_phystwin.selective_risk import (
    bootstrap_guard_evaluation,
    evaluate_guard_by_stratum,
    evaluate_prediction_intervals_by_horizon,
    selective_risk_curve,
)

curve = selective_risk_curve(
    baseline_loss=baseline_track_error,
    candidate_loss=candidate_track_error,
    acceptance_score=source_calibrated_safety_score,
)

summary = bootstrap_guard_evaluation(
    baseline_loss=baseline_track_error,
    candidate_loss=candidate_track_error,
    accepted=locked_guard_decision,
    group_ids=interaction_ids,
    bootstrap_repeats=5000,
    confidence_level=0.95,
    seed=20260726,
)

by_rank = evaluate_guard_by_stratum(
    baseline_track_error,
    candidate_track_error,
    locked_guard_decision,
    identifiable_rank,
)

by_horizon = evaluate_prediction_intervals_by_horizon(
    target=future_error,
    lower=predictive_lower,
    upper=predictive_upper,
    horizon=horizon_seconds,
)
```

The module computes paired diagnostics only. It does not choose the scientific
loss, score, threshold, harmful tolerance, quantile, strata, statistical unit,
nominal coverage, or bootstrap protocol. Those choices remain part of the frozen
experiment contract.
