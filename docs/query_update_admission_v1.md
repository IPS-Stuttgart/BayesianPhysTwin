# Query-specific guarded update admission v1

## Purpose

BayesianPhysTwin already distinguishes several decisions that must not be
collapsed:

- a Prob4D provider may or may not be competent on its own observation task;
- a covariance treatment may or may not be appropriate for one physical query;
- query uncertainty may or may not be calibrated on independent source groups;
- a candidate belief may or may not improve the downstream query relative to
  the unchanged physical baseline.

`QueryUpdateAdmissionCertificateV1` composes those boundaries into one final,
query-specific accept-or-fallback decision. It does not define a physical query,
fit a provider, estimate calibration, or calculate a regret bound. Those inputs
must be produced and frozen by their owning protocols before this certificate is
constructed.

The existing `QueryCovarianceTreatmentDecisionV1` remains responsible only for
selecting a covariance interpretation. It explicitly does not authorize a
belief update. The new admission certificate is the downstream decision layer.

## Decision inputs

`QueryUpdateEvidenceV1` binds content identities for:

- the `PhysicalQueryV1` definition;
- the complete baseline belief;
- the complete candidate belief;
- the exact fallback belief;
- the provider-competence decision;
- the query-calibration artifact;
- the query-identifiability diagnostic;
- the source-frozen regret evidence; and
- the expected-information-gain evidence.

It also records the scalar summaries consumed by the frozen policy:

- whether provider competence passed;
- whether query calibration passed;
- the fraction of the query supported by identifiable state or nuisance modes;
- the baseline-relative regret upper bound; and
- the expected information gain for the named query.

The evidence contract requires the candidate identity to differ from the
baseline and the fallback identity to equal the baseline identity exactly.
This prevents a rejection from silently selecting a reconstructed or partially
modified approximation of the baseline.

## Frozen policy

`QueryUpdateAdmissionPolicyV1` declares:

- `minimum_identifiable_fraction`;
- `maximum_regret_upper_bound`;
- `minimum_expected_information_gain`;
- whether provider competence is mandatory;
- whether query calibration is mandatory; and
- one numerical tolerance.

The update is authorized only when every required gate passes. Rejections retain
all failing reasons rather than returning only the first failure. Registered
reason identifiers are:

```text
provider-competence-not-passed
query-calibration-not-passed
identifiable-query-fraction-below-threshold
query-regret-upper-bound-exceeds-threshold
query-information-gain-below-threshold
```

A passing decision has the single reason `query-update-authorized`.

## Exact fallback

The resulting certificate is internally self-validating:

- authorization must agree with the policy and evidence;
- `exact_fallback` must be the logical opposite of authorization;
- an accepted certificate must select the candidate belief identity; and
- a rejected certificate must select the exact baseline belief identity.

Changing any decision field without changing the bound evidence invalidates the
content-addressed certificate.

## Python example

```python
from bayesian_phystwin.query_update_admission_v1 import (
    QueryUpdateAdmissionPolicyV1,
    QueryUpdateEvidenceV1,
    evaluate_query_update_admission,
)

policy = QueryUpdateAdmissionPolicyV1(
    minimum_identifiable_fraction=0.5,
    maximum_regret_upper_bound=0.0,
    minimum_expected_information_gain=0.01,
)

evidence = QueryUpdateEvidenceV1(
    physical_query_id=physical_query.artifact_id,
    baseline_belief_id=baseline_belief.artifact_id,
    candidate_belief_id=candidate_belief.artifact_id,
    fallback_belief_id=baseline_belief.artifact_id,
    provider_decision_id=provider_decision.artifact_id,
    query_calibration_id=query_calibration.artifact_id,
    identifiability_diagnostic_id=identifiability.artifact_id,
    regret_evidence_id=query_regret.artifact_id,
    information_gain_evidence_id=information_gain.artifact_id,
    provider_competence_passed=provider_decision.authorized,
    query_calibration_passed=query_calibration_supported,
    identifiable_fraction=identifiability.query_fraction,
    regret_upper_bound=query_regret.upper_bound,
    expected_information_gain=information_gain.expected_value,
)

decision = evaluate_query_update_admission(evidence, policy=policy)
selected_belief_id = decision.selected_belief_id
assert decision.authorized or decision.exact_fallback
```

The example uses illustrative producer attributes. The admission module consumes
only immutable identities, Booleans, and finite summaries; it does not import
Causal4D or Prob4D runtime internals.

## Statistical and ownership boundary

Provider, regret, calibration, identifiability, and information-gain evidence
must use the independent physical object or acquisition session as the
statistical unit whenever that is the true independent unit. Frames, points,
tracks, views, and taxels must not be promoted to independent groups merely to
increase sample size.

Prob4D owns observation-provider competence and observation lineage.
BayesianPhysTwin owns the candidate-versus-exact-baseline decision. Causal4D
owns the interventional query and downstream counterfactual interpretation.
The certificate binds these decisions without transferring ownership between
repositories.

A passing software certificate is not empirical evidence by itself. It does not
establish fresh-object benefit, calibrated deployment uncertainty, intervention
benefit, safety, or state of the art. The bound protocols and outcome artifacts
remain authoritative for those claims.
