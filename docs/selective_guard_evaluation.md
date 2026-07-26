# Selective guard evaluation

A Bayesian-PhysTwin guard must be evaluated against the exact physical baseline
returned on rejection. Average candidate accuracy alone does not establish that
the system accepts updates safely.

For observation or interaction unit `i`, define paired excess loss

```text
Delta_i = candidate_loss_i - baseline_loss_i.
```

Negative values improve on the fallback. Positive values are harmful unless they
are within a predeclared numerical or practical tolerance.

`bayesian_phystwin.selective_risk` provides three NumPy-only operations:

- `evaluate_guard(...)` evaluates one fixed accept/fallback decision;
- `selective_risk_curve(...)` evaluates every distinct acceptance-score
  threshold without splitting tied scores;
- `bootstrap_guard_evaluation(...)` resamples declared interaction, object, or
  session groups and reports percentile intervals.

## Reported quantities

For a fixed acceptance mask the point evaluation reports:

- coverage and exact fallback rate;
- mean baseline, all-candidate, and guarded selected-system loss;
- unconditional selected-system excess loss relative to baseline;
- mean excess loss among accepted updates;
- fraction of accepted updates whose excess exceeds the declared harmful
  tolerance; and
- worst accepted excess loss.

The guarded selected loss is

```text
selected_i = candidate_i  when accepted_i
             baseline_i   otherwise.
```

Consequently, zero acceptance must reproduce the baseline mean exactly and yield
zero selected-system excess loss.

## Score thresholds and ties

A risk-coverage curve evaluates each unique score value. All rows with an equal
score enter together, so the reported curve does not depend on arbitrary row
ordering. `higher_is_safer=False` supports scores for which smaller values imply
greater confidence.

A diagnostic curve may inspect all thresholds. A confirmatory deployed threshold
must be fixed from source/calibration data before target outcomes are opened.

## Statistical units

Rows from one interaction, physical object, acquisition session, overlapping
window family, or shared backbone realization are generally dependent. The
cluster bootstrap therefore samples the supplied `group_ids` with replacement
and retains every row from each selected group. Repeated groups are duplicated as
complete clusters.

Replicates containing no accepted row remain valid for coverage, fallback, and
selected-system loss. Their accepted-only statistics are missing and excluded
from the corresponding percentile interval; every interval reports the number of
finite replicates explicitly.

Use the paper's declared statistical unit as `group_ids`. Do not substitute
coordinate-level or pixel-level groups merely to obtain narrower intervals.

## Example

```python
from bayesian_phystwin.selective_risk import (
    bootstrap_guard_evaluation,
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
```

The module computes paired evaluation statistics only. It does not choose the
scientific loss, acceptance score, harmful tolerance, statistical unit, threshold,
or resampling protocol; those remain part of the frozen experiment contract.
