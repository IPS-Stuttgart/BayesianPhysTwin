# Posterior query covariance portfolios

## Purpose

BayesianPhysTwin exposes several raw covariance interpretations for the same
point estimate:

- the solver-native IRLS/Gauss--Newton working covariance;
- an exact local observed-information covariance when the grouped-mixture
  Hessian is positive definite; and
- a group-score sandwich covariance when enough defensibly independent groups
  are available.

A single matrix cannot communicate which interpretation produced it. Reporting
only one available matrix can also hide that another estimator failed or was
silently omitted. `bayesian_phystwin.posterior_covariance_portfolio` binds the
alternatives to one inference result and one registered linear query without
choosing a winner.

## Source contracts and adapters

`PosteriorCovarianceSourceV1` binds:

- the exact inference-result identity;
- the complete uncalibrated covariance matrix;
- its `PosteriorCovarianceSemanticsV1` interpretation;
- the source or estimator artifact identity; and
- optional finite JSON metadata.

The uncertainty namespace provides explicit adapters:

```python
from bayesian_phystwin.uncertainty import (
    group_sandwich_covariance_source,
    observed_information_covariance_source,
    working_covariance_source,
)

working_source = working_covariance_source(
    inference_result_id,
    solver_result.posterior_covariance,
    source_artifact_id=claim_bearing_candidate_id,
)
observed_source = observed_information_covariance_source(
    inference_result_id,
    observed_information_result,
)
sandwich_source = group_sandwich_covariance_source(
    inference_result_id,
    sandwich_result,
)
```

All source matrices must have the same parameter dimension. Calibrated
covariances are rejected because calibration is a later, query-specific stage.
The adapters retain the original estimator artifact and covariance-semantics
identities instead of relabelling their arrays.

## Common query projection

Use one query matrix for every covariance source:

```python
from bayesian_phystwin.uncertainty import (
    build_posterior_query_covariance_portfolio,
)

portfolio = build_posterior_query_covariance_portfolio(
    inference_result_id,
    query_set_id,
    query_matrix,
    [working_source, observed_source, sandwich_source],
    inference_admissible=True,
    reason="inference-admissible",
)
```

The builder applies `Q P Q.T` to every source and creates one
`PosteriorQueryUncertaintyV1` per method. It binds both the caller-owned
`query_set_id` and a content digest of the actual query matrix. Every projected
semantics record points back to its full-parameter source.

The final portfolio retains the query matrix and every source. Construction
therefore recomputes each projected covariance and the complete projected
semantics, then independently verifies the query, source, semantics, and
estimator identities. A manually assembled entry cannot substitute a different
covariance or interpretation with matching labels.

## Complete accounting

For an accepted result, the portfolio always uses `irls_working` as the
reference covariance. The other two raw methods must either be present or carry
an explicit unavailability reason:

```python
portfolio = build_posterior_query_covariance_portfolio(
    inference_result_id,
    query_set_id,
    query_matrix,
    [working_source],
    inference_admissible=True,
    reason="inference-admissible",
    unavailable_methods={
        "laplace_observed_information": (
            "observed-information-not-positive-definite"
        ),
        "group_sandwich": "fewer-than-three-independent-groups",
    },
)
```

This prevents an unavailable or inconvenient covariance interpretation from
silently disappearing. The portfolio records
`no-implicit-covariance-winner-v1`; it does not rank methods by trace, width, or
coverage.

A rejected inference result has a different contract. It contains exactly one
`exact_prior_fallback` covariance, requires the portfolio reason to match the
fallback covariance's recorded reason, and cannot list accepted-update
alternatives. Use
`exact_prior_fallback_covariance_source(...)` to bind this case explicitly.

## Calibration boundary

Portfolio members are deliberately raw. Each projected
`PosteriorQueryUncertaintyV1` can later receive its own source-frozen
`QueryCalibrationV1`, but calibration outcomes do not alter the raw portfolio.
A paper or deployment report should compare, for every retained method:

- empirical coverage;
- full interval width;
- proper score;
- acceptance and fallback frequency; and
- object- or session-level worst-group behavior.

The portfolio itself establishes identity, common-query projection, complete
method accounting, and interpretation. It does not establish calibration,
provider competence, physical-query benefit, intervention benefit, deployment
safety, or state of the art.
