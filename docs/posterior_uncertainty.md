# Posterior query uncertainty

`PosteriorQueryUncertaintyV1` is the claim-bearing join between a numerical
posterior result, a registered physical-query set, one covariance interpretation,
an optional covariance-estimator artifact, and an optional independent-group
query calibration.

It does not change a point estimate or an existing inference algorithm. Its
purpose is to prevent a working IRLS covariance, exact local
observed-information covariance, group-score sandwich covariance, and calibrated
query covariance from being interchanged without preserving their different
semantics.

## Information flow

```text
claim-bearing inference result
        |
        +--> registered query set and projected source covariance
        |          |
        |          +--> covariance semantics
        |          +--> observed-information or sandwich artifact, when used
        |          |
        |          +--> predictor_id frozen before calibration outcomes
        |                         |
        |                         +--> QueryCalibrationV1
        |                                  |
        v                                  v
      source covariance             calibrated covariance
      explicitly uncalibrated       finite-group coverage claim
```

The predictor identity binds:

- the complete numerical inference-result identity;
- the exact query-set identity;
- shape, dtype, and bytes of the source query covariance;
- the covariance-semantics artifact; and
- the covariance-estimator artifact when the source is not the solver's working
  covariance.

Free-form metadata and the later calibration are deliberately excluded from the
predictor identity. A `QueryCalibrationV1` is admitted only when its
`predictor_id` and `query_set_id` match exactly.

## Covariance semantics

The source covariance must carry an uncalibrated
`PosteriorCovarianceSemanticsV1`:

- `irls_working` is a local working covariance;
- `laplace_observed_information` is the local exact-mixture-curvature
  approximation; and
- `group_sandwich` uses declared independent group-score sums.

Observed-information and group-sandwich sources require an explicit estimator
artifact identity. Calibration does not rewrite that source interpretation. The
reported calibrated semantics retain the source method and additionally bind the
query-calibration artifact.

`reported_query_covariance_m2` returns the calibrated covariance only when a
matching calibration is present. Otherwise it returns the source covariance and
`calibrated` remains false. Absence of calibration is never represented as a
nominal coverage claim.

## Finite-group feasibility

`finite_group_coverage_status()` reports whether a requested split-conformal
coverage has a finite order statistic before any calibration outcomes are read.
The independent group is the physical object or acquisition session declared by
the protocol, not a frame, view, point, track, or taxel.

For ten independent calibration objects:

```python
from bayesian_phystwin.uncertainty import finite_group_coverage_status

status = finite_group_coverage_status(10, 0.95)
assert not status.finite
assert status.maximum_finite_coverage == 10 / 11
assert status.minimum_required_group_count == 19
```

A report should therefore say that ordinary finite-group 95% split-conformal
coverage is unavailable with ten calibration objects. It may use a preregistered
attainable level at or below `10 / 11`; it must not relabel an infinite or
unavailable quantile as a finite 95% interval.

## Supported namespace

Stable uncertainty-facing interfaces are collected under
`bayesian_phystwin.uncertainty`. The package root is intentionally not expanded
with the new contract. This keeps downstream Prob4D and Causal4D integrations on
an explicit surface while research modules remain separately versioned.

## Claim boundary

A content-addressed uncertainty object proves numerical and provenance
consistency. It does not prove exchangeability, provider competence, physical
state identifiability, downstream accuracy, safe deployment, or state of the
art. Those claims still require the registered independent-object/session
calibration and confirmation protocols with exact fallback and complete failure
accounting.
