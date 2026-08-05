# Finite-group calibration design

BayesianPhysTwin calibrates uncertainty over independent physical objects or
acquisition sessions. Frames, views, tracks, points, and tactile taxels are
repeated observations within one group; they do not increase the finite-sample
calibration count.

## Rank and finite-coverage boundary

For `n` independent calibration groups and nominal coverage `c`, the ordinary
split-conformal rank is

```text
ceil((n + 1) * c).
```

The quantile is finite only when this rank is at most `n`. Equivalently, the
largest finite nominal coverage is `n / (n + 1)`.

The public helpers in `bayesian_phystwin.calibration` use decimal-exact
arithmetic for these planning calculations:

```python
from bayesian_phystwin.calibration import (
    finite_group_conformal_rank,
    maximum_finite_group_coverage,
    minimum_groups_for_finite_conformal,
    plan_finite_group_calibration,
)

assert finite_group_conformal_rank(10, 0.90) == 10
assert finite_group_conformal_rank(10, 0.95) == 11
assert minimum_groups_for_finite_conformal(0.90) == 9
assert minimum_groups_for_finite_conformal(0.95) == 19
```

`plan_finite_group_calibration` fails before target access when the requested
coverage would require an infinite quantile.

## Information-order boundary

The split-conformal interpretation applies to the complete deployed policy, not
only to its final numeric scale. Before interval-calibration outcomes are
inspected, the following must already be frozen:

- the observation producer and physical predictor;
- the nonconformity score;
- the acceptance or regret guard;
- the grouping rule;
- the registered endpoint set; and
- every policy-selection decision that changes the deployed predictor.

The same calibration outcomes must not both select the deployed policy and
calibrate its split-conformal interval. A CV+ or jackknife+ construction can use
a different information order, but it requires a separately versioned contract
and guarantee.

## Deform360 amendment

The additive amendment at
`protocols/amendments/deform360_official_hub_visuotactile_v1_calibration_separation.json`
locks the following design before selected calibration payload access:

- the primary interval pools all 10 independent calibration objects;
- nominal 90% coverage has finite rank 10;
- nominal 95% coverage is not claimable from those 10 groups by ordinary
  split conformal;
- each five-object stratum has maximum finite nominal coverage `5 / 6`, so
  sheet- and volumetric-specific 90% intervals are forbidden;
- stratum results remain diagnostic and participate in the preregistered
  no-regression rule;
- policy selection must be external or source-only before interval scores are
  inspected; and
- calibration, confirmation, and target payloads remain unopened at the
  amendment boundary.

This amendment does not change the frozen Stage-0 cohort and does not establish
provider competence, physical-query improvement, calibrated deployment
uncertainty, or Causal4D intervention benefit.
