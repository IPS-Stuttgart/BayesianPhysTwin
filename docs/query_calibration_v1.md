# Query calibration V1

`QueryCalibrationV1` converts a frozen Bayesian query covariance into a
finite-group split-conformal covariance for deployment and evaluation. The
calibration unit is one independent physical object or acquisition session,
not one frame, view, track, point, or tactile sample.

## Construction

For endpoint `j` in independent group `g`, let `e[g, j]` be the vector query
residual and let `Sigma[g, j]` be the raw predictive covariance. Two optional
covariance corrections must be frozen before calibration outcomes are opened:

```text
Sigma_base = covariance_scale * Sigma + isotropic_variance * I.
```

The group contributes one score:

```text
s[g] = max_j sqrt(e[g, j]^T Sigma_base[g, j]^-1 e[g, j]).
```

For `n` independent calibration groups and nominal coverage `c`, the artifact
uses rank

```text
k = ceil((n + 1) * c).
```

The fit fails before residuals or covariances are inspected when `k > n`. For a
finite rank, `q` is the `k`-th ordered group score and future covariances become

```text
Sigma_deployed = q^2 * Sigma_base.
```

Taking the maximum within each group targets simultaneous coverage of every
registered endpoint in one future group. Long sequences therefore do not gain
extra weight merely because they contain more frames.

A zero conformal radius is retained exactly when every relevant order statistic
is zero. In that edge case the deployed region is deterministically degenerate:
only a zero residual is covered. The implementation does not silently inject a
nugget after calibration; a positive `isotropic_variance` must have been frozen
before the calibration outcomes were opened.

## Provenance and information order

Every artifact binds lower-case SHA-256 identifiers for:

- the complete deployed predictor;
- the registered query set or horizon/observability partition;
- the grouping rule;
- the acceptance or regret guard; and
- the exact calibration-evidence manifest.

The artifact also records all independent group IDs, their maximum scores, the
finite-sample rank, the covariance transform, and the derived conformal
quantile. Group order is canonical, JSON loading rejects duplicate keys, and
the artifact ID is recomputed on every load.

`covariance_scale`, `isotropic_variance`, the predictor, the query set, and the
guard must be selected without using the same calibration outcomes that produce
`q`. The contract rejects a fit when the predictor was not frozen before scores
or when calibration outcomes were used for policy selection.

## Retained-artifact behavior

Calibration scores and deployed covariance arrays are copied into bytes-backed
NumPy storage. Their writeability cannot be restored with
`array.setflags(write=True)`, so the content address and returned covariance do
not depend on later caller mutation.

`save_query_calibration(...)` publishes a complete JSON artifact atomically and
is idempotent for the same content address. It refuses symbolic-link targets,
corrupt existing files, non-regular destinations, and attempts to replace a
different calibration artifact. Concurrent publication cannot overwrite an
existing path. Loading normalizes unreadable and malformed-JSON failures and
revalidates the closed schema, finite-group rank, derived order statistic, and
artifact ID.

The numerical boundary is also fail-closed: transformed covariances,
Mahalanobis scores, the squared conformal multiplier, and deployed covariance
entries must remain finite. Overflow is reported instead of being retained as
`inf` or `nan`.

## Python example

```python
import numpy as np

from bayesian_phystwin.query_calibration import (
    calibrate_query_covariance,
    fit_query_calibration,
    save_query_calibration,
)

calibration = fit_query_calibration(
    calibration_group_ids=[f"object-{index:02d}" for index in range(10)],
    residual_groups=residuals_by_object,
    covariance_groups=covariances_by_object,
    nominal_coverage=0.90,
    predictor_id=predictor_artifact_id,
    query_set_id=query_set_artifact_id,
    grouping_rule_id=grouping_rule_artifact_id,
    guard_id=guard_artifact_id,
    calibration_evidence_id=calibration_manifest_id,
    covariance_scale=1.0,
    isotropic_variance=0.0,
    predictor_frozen_before_scores=True,
    calibration_outcomes_used_for_selection=False,
)

save_query_calibration(calibration, "query_calibration.json")
future_covariance = calibrate_query_covariance(
    np.asarray(raw_future_covariance),
    calibration,
)
```

Use one artifact for the exact registered query set. Separate horizon or
observability strata require separately frozen query-set IDs and enough
independent groups for each requested finite-sample coverage. With 10 groups,
90% pooled coverage has finite rank 10, whereas ordinary 95% split-conformal
coverage is impossible without at least 19 independent groups.

## Claim boundary

Under exchangeability of calibration groups and the future group, and with the
complete deployed policy frozen before calibration outcomes are inspected, the
maximum-score construction supplies the ordinary marginal finite-group
split-conformal guarantee for the registered endpoints. It does not establish:

- conditional coverage for every object type, horizon, or observability level;
- calibration after choosing a policy on the same outcomes;
- physical-state identifiability or a unique causal explanation;
- provider competence, downstream physical-query improvement, or deployment
  safety; or
- independence when several sequences come from the same physical object or
  acquisition session.

Report empirical coverage, interval width, and worst-group behavior on a
separate confirmation cohort. Preserve the exact physical fallback whenever a
prospective guard rejects the updated belief.
