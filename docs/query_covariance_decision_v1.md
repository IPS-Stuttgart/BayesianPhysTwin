# Query covariance treatment decision v1

`query_covariance_decision_v1` is the narrow composition layer between three
already existing artifacts:

1. BayesianPhysTwin's target-blind `PhysicalQueryV1`;
2. Prob4D's neutral `QueryCovarianceProjectionV1.summary()` output; and
3. BayesianPhysTwin's independently evaluated
   `CovarianceOnlyValueCertificateV1`.

It does not fit covariance, select thresholds from target outcomes, recompute
proper-score statistics, or authorize a physical update by itself.

## Purpose

Prob4D can report how much covariance in a registered query is caused by shared
low-rank modes. BayesianPhysTwin owns the query and the rule deciding whether
that amount justifies marginal-gauge or complete explicit-joint-gauge inference.
The value certificate separately determines whether the already frozen
covariance-only policy has bounded score regret, interval width, and harmful
accepted-group frequency while preserving byte-identical point means.

The composition decision joins those responsibilities without moving them:

```text
Prob4D neutral projection summary
        +
BayesianPhysTwin PhysicalQueryV1
        +
BayesianPhysTwin CovarianceOnlyValueCertificateV1
        |
        v
QueryCovarianceTreatmentDecisionV1
```

## Structural bindings

`compose_query_covariance_treatment` fails closed unless:

- the Prob4D summary has the exact version-1 schema and claim boundary;
- its query dimension equals the frozen `PhysicalQueryV1` dimension;
- trace, rank, directional-fraction, and shared-fraction fields are internally
  consistent;
- the value certificate's `query_set_id` and `policy_freeze_artifact_id` equal
  the exact `PhysicalQueryV1.query_id`;
- its statistical unit equals the frozen complete object/session definition;
- its proper score equals the query's registered proper score;
- its score, harmful-increase, width, and familywise-confidence thresholds equal
  the corresponding frozen query settings; and
- the query uses the supported trace-fraction diagnostic and selection rule.

These are lineage and policy mismatches, not ordinary provider rejections, so no
decision artifact is created when they fail.

## Treatment rule

For shared query-covariance trace fraction `r` and frozen threshold `tau`, the
selection is:

```text
r is undefined  -> no treatment; exact fallback remains required
r < tau         -> marginal-gauge covariance
r >= tau        -> complete explicit-joint-gauge covariance
```

The selected treatment must equal the source-frozen
`principal_covariance_treatment` in `PhysicalQueryV1`. A mismatch produces a
content-addressed negative decision rather than changing the frozen query after
seeing target outcomes.

Authorization additionally requires the exact-mean covariance value certificate
to be certified. The decision binds:

- the physical-query identity;
- source observation artifact identity;
- a content address over the validated Prob4D summary, query Jacobian provider,
  and provider manifest;
- value-certificate and candidate/reference policy identities;
- selected and principal covariance treatments;
- shared-covariance relevance and threshold; and
- exact fallback identity.

## Example

```python
from bayesian_phystwin.query_covariance_decision_v1 import (
    compose_query_covariance_treatment,
)

summary = prob4d_projection.summary()
decision = compose_query_covariance_treatment(
    physical_query,
    summary,
    covariance_value_certificate,
    source_observation_artifact_id=observation_belief.artifact_id,
)

if decision.authorized:
    treatment = decision.selected_covariance_treatment
else:
    fallback_id = decision.exact_fallback_id
```

The decision has atomic, no-clobber JSON load/write helpers. Revalidation checks
its content address and recomputes the authorization reasons from its recorded
gates.

## Scientific boundary

Passing this decision is software composition evidence only. It does not
establish:

- real Prob4D provider competence;
- fresh-object BayesianPhysTwin benefit;
- calibrated deployment uncertainty;
- Causal4D interventional benefit;
- deployment safety; or
- state of the art.

Those claims still require the preregistered independent object/session study,
provider-competence gate, accepted-query scoring, coverage and width reporting,
exact fallback accounting, and target-access attestations.
