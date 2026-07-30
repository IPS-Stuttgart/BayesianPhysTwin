# Group-balanced conformal calibration

## Purpose

Fresh Bayesian-PhysTwin validation is grouped by physical object or independent
acquisition session. Calibration must therefore use the same independent unit.
Treating every frame or tracked coordinate as exchangeable would let long
sequences dominate the calibration quantile and would overstate the effective
sample size.

`bayesian_phystwin.grouped_conformal` provides a NumPy-only split-conformal
primitive that assigns exactly one score to each registered calibration group.
It is intended for frozen prospective evaluations, not for target-informed
method selection.

## Construction

For calibration group `g`, target losses `y_gj`, and positive predicted scales
`s_gj`, the scaled group score is

```text
r_g = max_j y_gj / s_gj.
```

For additive bounds with predictions `p_gj`, the score is

```text
r_g = max_j (y_gj - p_gj).
```

The maximum reduction means that one accepted quantile covers every registered
endpoint in a future group simultaneously. Each group contributes one value,
regardless of its frame or coordinate count.

For `G` calibration groups and nominal coverage `1 - alpha`, the finite-sample
rank is

```text
k = ceil((G + 1) * (1 - alpha)).
```

If `k > G`, no finite distribution-free split-conformal quantile exists. The API
returns an infinite quantile and infinite upper bounds instead of silently using
an anti-conservative order statistic. In particular, nominal 90% coverage needs
at least nine independent calibration groups to produce a finite bound.

## Python API

```python
import numpy as np

from bayesian_phystwin import grouped_conformal_upper_bounds

result = grouped_conformal_upper_bounds(
    calibration_targets=(
        np.asarray([0.010, 0.014]),
        np.asarray([0.012, 0.018]),
        np.asarray([0.009, 0.015]),
    ),
    calibration_predictions=(
        np.asarray([0.008, 0.010]),
        np.asarray([0.009, 0.011]),
        np.asarray([0.007, 0.010]),
    ),
    future_prediction=np.asarray([0.009, 0.012]),
    coverage=0.5,
    score="scaled",
)

print(result.finite_sample_rank)
print(result.quantile)
print(result.upper_bound)
```

The returned upper bounds and group-score vector are immutable arrays. The
result records the nominal coverage, finite-sample rank, number of calibration
groups, score family, and selected quantile.

## Required experimental boundary

A simultaneous group-level coverage statement requires all of the following:

- calibration groups and the future group are exchangeable at the registered
  object/session level;
- the predictor, score family, endpoint set, horizon grid, grouping rule, and
  missing-data policy are frozen before target outcomes are opened;
- development, calibration, and target groups are disjoint;
- every technical failure and preregistered exclusion is retained rather than
  silently replaced;
- the future group is evaluated once, without target-informed retuning.

Varying group size does not create frame weighting, but it can affect the
distribution of a maximum score. Group size and missingness must therefore be
part of the registered exchangeability argument.

## Claim boundary

This primitive provides finite-sample split-conformal accounting under the
stated group-exchangeability assumptions. It does **not** establish that raw
Bayesian covariance is calibrated, that an update is physically admissible, that
exact fallback is a universal safety guarantee, or that the method transfers to
new objects. Those are separate empirical gates in the prospective protocol.
