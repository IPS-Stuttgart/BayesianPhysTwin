# Query-specific guarded update admission v1

## Purpose

BayesianPhysTwin keeps several decisions separate:

- a Prob4D or other observation provider may or may not be competent;
- a covariance treatment may or may not be suitable for one physical query;
- the query may or may not be identifiable and calibrated on independent source
  groups; and
- a complete candidate belief may or may not improve the named query relative to
  the unchanged physical baseline.

`QueryUpdateAdmissionCertificateV1` composes those decisions into one
query-specific accept-or-exact-fallback result. It does not define a query, fit a
provider, estimate calibration, or calculate regret or information gain. Those
quantities remain owned by separately frozen protocols.

`QueryCovarianceTreatmentDecisionV1` still selects only a covariance
interpretation. It does not authorize a physical-twin update.

## Evidence and identity binding

`QueryUpdateEvidenceV1` binds the content identities of:

- the `PhysicalQueryV1` definition;
- the complete baseline, candidate, and exact fallback beliefs;
- the provider-competence decision;
- the query-calibration artifact;
- the query-identifiability diagnostic;
- the source-frozen regret evidence; and
- the expected-information-gain evidence.

The candidate must differ from the baseline. The fallback must equal the
baseline exactly.

For claim-bearing use, evidence should additionally bind:

- `source_protocol_id`;
- `grouping_rule_id`; and
- `independent_group_count`.

These three fields are all-or-none. They make the physical object or acquisition
session grouping visible in the evidence and final certificate instead of
leaving it in free-form metadata.

### Record-derived construction

Use `build_query_update_evidence_from_records` for claim-bearing composition.
Each supplied record must:

1. contain an `artifact_id` equal to the content hash of the remainder of the
   record;
2. carry the same `physical_query_id`, baseline and candidate belief IDs,
   source protocol ID, grouping-rule ID, and independent-group count; and
3. expose its one admission summary:
   `provider_competence_passed`, `query_calibration_passed`,
   `identifiable_fraction`, `regret_upper_bound`, or
   `expected_information_gain`.

The builder rejects tampered records and cross-artifact context drift before it
constructs `QueryUpdateEvidenceV1`. The owning repositories may retain richer
native artifacts; a compact bound record can be published as their
content-addressed admission summary.

Direct `QueryUpdateEvidenceV1` construction remains available for
software-only and compatibility uses. A claim-bearing policy should set
`require_source_context=True`, which rejects direct evidence lacking the frozen
source context with exact baseline fallback.

## Frozen policy

`QueryUpdateAdmissionPolicyV1` declares:

- `minimum_identifiable_fraction`;
- `maximum_regret_upper_bound`;
- `minimum_expected_information_gain`;
- whether provider competence is mandatory;
- whether query calibration is mandatory;
- whether source context is mandatory; and
- separate tolerances for identifiable fraction, regret, and information gain.

The separate tolerance fields are:

```text
identifiable_fraction_tolerance
regret_tolerance
information_gain_tolerance
```

This avoids applying one dimensionless epsilon indiscriminately to quantities
with different scales and semantics. `numerical_tolerance` remains a
compatibility shorthand that assigns one value to all three fields. It cannot be
combined with explicitly split tolerances and is not part of the canonical
policy descriptor.

The update is authorized only when every required gate passes. Rejections retain
all failing reasons:

```text
provider-competence-not-passed
query-calibration-not-passed
source-context-not-bound
identifiable-query-fraction-below-threshold
query-regret-upper-bound-exceeds-threshold
query-information-gain-below-threshold
```

A passing decision has the single reason `query-update-authorized`.

## Exact fallback

The certificate validates itself:

- authorization must agree with the frozen policy and evidence;
- `exact_fallback` must be the logical opposite of authorization;
- acceptance must select the complete candidate belief identity; and
- rejection must select the exact baseline belief identity.

The source protocol, grouping rule, and independent-group count are copied into
the certificate when present. Changing any bound identity, summary, decision
field, or context invalidates the content address.

## Python example

```python
from bayesian_phystwin.query_update_admission_v1 import (
    QueryUpdateAdmissionPolicyV1,
    build_query_update_evidence_from_records,
    evaluate_query_update_admission,
)

policy = QueryUpdateAdmissionPolicyV1(
    minimum_identifiable_fraction=0.5,
    maximum_regret_upper_bound=0.0,
    minimum_expected_information_gain=0.01,
    require_source_context=True,
    identifiable_fraction_tolerance=1e-12,
    regret_tolerance=1e-6,
    information_gain_tolerance=1e-12,
)

evidence = build_query_update_evidence_from_records(
    physical_query_id=physical_query.artifact_id,
    baseline_belief_id=baseline_belief.artifact_id,
    candidate_belief_id=candidate_belief.artifact_id,
    fallback_belief_id=baseline_belief.artifact_id,
    source_protocol_id=source_protocol.artifact_id,
    grouping_rule_id=grouping_rule.artifact_id,
    independent_group_count=source_protocol.independent_group_count,
    provider_decision=provider_admission_record,
    query_calibration=query_calibration_record,
    identifiability_diagnostic=identifiability_record,
    regret_evidence=query_regret_record,
    information_gain_evidence=information_gain_record,
)

decision = evaluate_query_update_admission(evidence, policy=policy)
selected_belief_id = decision.selected_belief_id
assert decision.authorized or decision.exact_fallback
```

The five bound records in the example are content-addressed summaries, not
unverified dictionaries. The builder revalidates every record and checks their
shared query and source context.

## Statistical and ownership boundary

Provider, regret, calibration, identifiability, and information-gain evidence
must use the independent physical object or acquisition session as the
statistical unit whenever that is the true independent unit. Frames, points,
tracks, views, and taxels must not be promoted to independent groups merely to
increase sample size.

Prob4D owns observation-provider competence and observation lineage.
BayesianPhysTwin owns the candidate-versus-exact-baseline decision. Causal4D
owns the interventional query and downstream counterfactual interpretation.
The certificate binds these decisions without transferring ownership.

A passing software certificate is not empirical evidence by itself. It does not
establish fresh-object benefit, calibrated deployment uncertainty,
interventional benefit, safety, or state of the art. The bound protocols and
outcome artifacts remain authoritative.
