# Query-specific admission certificate v1

## Purpose

A provider can be scientifically competent without improving every downstream
physical query. Conversely, a globally small trajectory change can materially
alter one intervention-sensitive quantity. BayesianPhysTwin therefore needs an
explicit decision between a validated candidate belief and the unchanged
physical belief at the **query** boundary.

`QueryAdmissionCertificateV1` composes existing frozen artifacts rather than
introducing another estimator. Admission requires all of the following:

```text
provider competence passes
AND query covariance treatment is authorized
AND source-evaluated query non-harm passes
AND query information/identifiability passes
```

Any failed clause selects the physical baseline exactly.

## Inputs

The composer binds:

- a content-addressed `PhysicalQueryV1`;
- the provider `EvidenceDecisionV1` already frozen in the query;
- a `QueryCovarianceTreatmentDecisionV1` for the same query and exact fallback;
- immutable candidate/baseline query mean and covariance identities; and
- source-evaluated query evidence grouped by the query's declared independent
  physical unit.

The provider decision must match the exact decision ID recorded under the
policy's `provider_decision_key`, which defaults to `source-provider-gate`.
Changing the provider result after the physical query was frozen is rejected
rather than interpreted as a new admission.

## Source-evaluated query evidence

`QueryAdmissionEvidenceV1` records:

- the candidate belief and query mean/covariance identities;
- the baseline query mean/covariance identities;
- the proper score and physical width unit;
- the independent-group count;
- mean, upper-bound, maximum, and worst-group score regret;
- harmful-group frequency;
- accepted coverage and mean full interval width;
- overlap with the identifiable subspace;
- shared-covariance relevance;
- expected information gain; and
- the information-order declarations for policy freezing, candidate selection,
  and independent statistical units.

The evidence object is content addressed. Its identifier changes whenever a
metric, identity, information-order declaration, or metadata field changes.

## Admission policy

`QueryAdmissionPolicyV1` adds only thresholds that are not already part of
`PhysicalQueryDecisionMarginsV1`:

- required provider evidence level and claim authorization;
- minimum independent-group count;
- minimum identifiable-subspace overlap;
- minimum expected information gain; and
- maximum harmful-group fraction.

Proper-score regret, accepted coverage, interval width, worst-group regret, and
shared-covariance relevance remain governed by the margins frozen in
`PhysicalQueryV1`. The certificate copies those margins into its own identity so
an admission cannot silently use different thresholds.

## Exact fallback

A passing certificate selects `candidate_belief_id`. Every rejection selects
`baseline_physical_belief_id` and records `exact_fallback=true`. The separate
`exact_fallback_id` from `PhysicalQueryV1` remains bound so downstream Causal4D
or orchestration code can verify the expected fallback bytes or artifact
identity.

Reasons are deterministic and grouped into three visible decisions:

- `provider_competence_passed`;
- `query_nonharm_passed`; and
- `query_information_passed`.

The overall `admitted` flag is true only when all three pass and the covariance
treatment is authorized. Redundant booleans, selected belief, and reasons are
recomputed during construction and loading; contradictory records are rejected.

## Python use

```python
from bayesian_phystwin.query_admission_v1 import (
    QueryAdmissionEvidenceV1,
    QueryAdmissionPolicyV1,
    compose_query_admission,
)

certificate = compose_query_admission(
    physical_query,
    provider_evidence_decision,
    query_covariance_decision,
    query_evidence,
    policy=QueryAdmissionPolicyV1(
        minimum_group_count=10,
        minimum_identifiable_subspace_overlap=0.5,
        minimum_expected_information_gain=0.1,
    ),
)

selected_belief_id = certificate.selected_belief_id
assert certificate.admitted or certificate.exact_fallback
```

Use `write_query_admission_certificate` and
`load_query_admission_certificate` for strict, content-addressed JSON
serialization.

## Scientific boundary

A passing certificate is an admission record for already-frozen source evidence.
It is not by itself:

- proof that a provider is competent on fresh objects or sessions;
- calibrated deployment uncertainty;
- evidence that Causal4D improves held-out intervention prediction;
- deployment or safety authorization; or
- a state-of-the-art result.

Those claims require their own registered, object/session-disjoint experiments.
A negative query admission remains a complete result and must not be repaired by
retuning on the same evaluation groups.
