# Group-robust posterior covariance

## Purpose

BayesianPhysTwin robust updates currently expose a local IRLS/Gauss--Newton
working covariance. That matrix is useful for numerical conditioning and local
belief propagation, but repeated rows from one object, window, contact episode,
or sensor region must not be counted as independent evidence.

`bayesian_phystwin.group_sandwich_covariance` adds an explicit group-score
sandwich estimator. It does not change a point estimate and does not claim that
the resulting covariance is calibrated.

## Estimator

Let `H` be the positive-definite local information matrix (the bread), let
`u_i` be a data-score contribution, and let `g(i)` be a declared independent
correlation group. The implementation first forms

```text
s_g = sum_{i: g(i)=g} u_i
```

and then computes

```text
V = c_G H^{-1} (sum_g s_g s_g^T) H^{-1}
```

The default finite-group correction is

```text
c_G = G / (G - 1).
```

`minimum_group_count` defaults to three. A caller may raise that threshold for a
claim-bearing protocol. Two groups are accepted only when the caller explicitly
selects a lower threshold; a single group always fails closed.

## Group semantics

The group must be the unit that is defensibly independent for the experiment,
for example:

- one physical object;
- one acquisition session;
- one MotionCrafter window when windows are independent;
- one contact episode; or
- one declared sensor correlation region.

Points, pixels, taxels, repeated frames, duplicated rows, and persistent
observations from the same physical unit are not new groups merely because they
appear as separate likelihood rows. The required `grouping_semantics` string,
ordered canonical group IDs, row counts, grouped scores, information matrix,
finite-group correction, and numerical covariance are all bound into the result
content identity.

## Prior and score boundary

When `prior_included=True`, `H` includes prior curvature. `score_rows` still
contains data-score contributions only. The prior is not repeated once per
group in the meat matrix. This is the intended generalized-Bayes interpretation
for the current BayesianPhysTwin updates.

The implementation validates that:

- the bread is finite, symmetric, and positive definite;
- score width matches the parameter dimension;
- identifiers are canonical strings and match the row count;
- group IDs are aggregated in a deterministic sorted order;
- the supplied covariance is exactly the covariance implied by the recorded
  bread, grouped scores, and correction;
- only numerical negative eigenvalues may be projected to zero; and
- the attached `PosteriorCovarianceSemanticsV1` is an uncalibrated
  `group_sandwich` interpretation with group-score correction enabled.

## Example

```python
import numpy as np

from bayesian_phystwin.group_sandwich_covariance import (
    estimate_group_sandwich_covariance,
)

result = estimate_group_sandwich_covariance(
    bread=np.array([[2.0]]),
    score_rows=np.array([[1.0], [2.0], [-1.0], [3.0]]),
    group_ids=["object-a", "object-a", "object-b", "object-c"],
    grouping_semantics="independent-physical-object-v1",
)

print(result.covariance)
print(result.covariance_semantics.method)  # group_sandwich
print(result.covariance_semantics.calibrated)  # False
```

The two rows from `object-a` are summed before the meat matrix is formed. They
do not increase the independent group count.

## Calibration boundary

A group-sandwich covariance estimates score variability under the declared
grouping. It is not, by itself, a finite-sample coverage guarantee. A
claim-bearing experiment should still report empirical coverage and full
interval width on independent calibration or confirmation groups. Where a
conformal multiplier is used, its artifact and exchangeability assumptions must
remain separate from the sandwich covariance artifact.

This implementation does not retroactively relabel frozen IRLS covariance. A
protocol must opt into it and must preserve one covariance policy throughout a
recursive run.
