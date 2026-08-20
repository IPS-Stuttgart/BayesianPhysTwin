# Physical-cause selection v2

## Purpose

`bayesian_phystwin.physical_cause_selection_v2` hardens the source-to-target
boundary of physical-cause routing. Version 1 defines the complete-belief
candidate semantics and exact baseline/discrepancy fallback. Version 2 keeps
those routing semantics unchanged but refuses to use a candidate's upper regret
bound unless a typed source-evidence set binds the complete comparison.

The selected interpretations remain:

```text
baseline              unchanged caller-owned physical belief
observation_bias      nuisance explanation without a physical update
readout_discrepancy   predictive discrepancy without a latent-state claim
physical_parameter    supported parameter or controller correction
physical_state        reachable and query-identifiable state correction
```

Version 2 is additive. Existing V1 records retain their exact identities and
interpretation. New target-facing studies should prefer V2 when more than one
physical cause is compared.

## Why a second evidence boundary is required

A scalar upper regret bound is meaningful only relative to the experiment that
produced it. The same numerical value can refer to another query, another group
roster, another proper score, or a pointwise interval that did not account for
candidate selection. Accepting that scalar without its source certificate can
silently substitute evidence while leaving candidate construction unchanged.

V2 therefore binds every candidate to the same:

- physical domain;
- registered query and exact query Jacobian;
- independent-group definition and complete source roster;
- proper-score definition and group-level score table;
- interval method and simultaneous interval artifact;
- confidence level and source-group count; and
- preregistered candidate-cause family.

Every candidate must have been frozen before source scores were read. Target
outcomes are forbidden. Every candidate must be evaluated on the complete source
roster, including inadmissible or weak candidates. This prevents a favorable
candidate from being selected from several separately reported pointwise bounds
without accounting for that comparison.

## Contracts

`PhysicalCauseCandidateEvidenceV2` binds one existing
`PhysicalCauseCandidateV1` to:

```text
candidate_id
cause
belief_id
construction_id
candidate_score_id
upper_regret
inference_admissible
evaluated_group_count
simultaneous_bound
candidate_frozen_before_scores
target_outcomes_used
```

The candidate identity already includes its complete belief, construction,
cause-specific support evidence, upper regret, admissibility, reason, and
metadata. V2 requires the certificate to reproduce those values exactly.

`PhysicalCauseEvidenceSetV2` canonically groups all registered causes and binds
the common domain/query/roster/score/interval identities. Candidate order cannot
change its content identity. A missing cause, duplicate cause, partial source
roster, reused candidate score, non-simultaneous bound, post-score candidate
freeze, or target-outcome use fails before selection.

`PhysicalCauseDecisionPolicyV2` repeats the source identities deliberately. The
selector requires exact equality with the evidence set and binds its complete
content address. This makes policy inspection explicit while preventing a
matching readable label from authorizing different bytes.

## Example

```python
from bayesian_phystwin.physical_cause_selection_v1 import (
    PhysicalCause,
    PhysicalCauseAmbiguityFallback,
    PhysicalCauseCandidateV1,
)
from bayesian_phystwin.physical_cause_selection_v2 import (
    PhysicalCauseCandidateEvidenceV2,
    PhysicalCauseDecisionPolicyV2,
    PhysicalCauseEvidenceSetV2,
    select_physical_cause_v2,
)

state = PhysicalCauseCandidateV1(
    cause=PhysicalCause.PHYSICAL_STATE,
    belief_id=state_belief.artifact_id,
    construction_id=state_construction_id,
    upper_regret=-0.040,
    inference_admissible=True,
    reason="source-crossfit",
    physical_response_id=physical_response_id,
    identifiability_report_id=identifiability_report_id,
)

discrepancy = PhysicalCauseCandidateV1(
    cause=PhysicalCause.READOUT_DISCREPANCY,
    belief_id=discrepancy_belief.artifact_id,
    construction_id=discrepancy_construction_id,
    upper_regret=-0.038,
    inference_admissible=True,
    reason="source-crossfit",
    discrepancy_model_id=discrepancy_model_id,
)

state_evidence = PhysicalCauseCandidateEvidenceV2(
    candidate_id=state.candidate_id,
    cause=state.cause,
    belief_id=state.belief_id,
    construction_id=state.construction_id,
    candidate_score_id=state_score_id,
    upper_regret=state.upper_regret,
    inference_admissible=state.inference_admissible,
    evaluated_group_count=12,
    simultaneous_bound=True,
    candidate_frozen_before_scores=True,
    target_outcomes_used=False,
)

discrepancy_evidence = PhysicalCauseCandidateEvidenceV2(
    candidate_id=discrepancy.candidate_id,
    cause=discrepancy.cause,
    belief_id=discrepancy.belief_id,
    construction_id=discrepancy.construction_id,
    candidate_score_id=discrepancy_score_id,
    upper_regret=discrepancy.upper_regret,
    inference_admissible=discrepancy.inference_admissible,
    evaluated_group_count=12,
    simultaneous_bound=True,
    candidate_frozen_before_scores=True,
    target_outcomes_used=False,
)

evidence = PhysicalCauseEvidenceSetV2(
    common_domain_id=domain_id,
    registered_query_id=query_id,
    query_jacobian_id=query_jacobian_id,
    grouping_rule_id=grouping_rule_id,
    source_roster_id=source_roster_id,
    score_definition_id=proper_score_id,
    source_score_table_id=source_score_table_id,
    interval_method_id=max_t_bootstrap_method_id,
    simultaneous_interval_id=simultaneous_interval_id,
    confidence_level=0.95,
    source_group_count=12,
    registered_candidate_causes=(state.cause, discrepancy.cause),
    candidate_evidence=(state_evidence, discrepancy_evidence),
)

policy = PhysicalCauseDecisionPolicyV2(
    baseline_belief_id=baseline_belief.artifact_id,
    common_domain_id=evidence.common_domain_id,
    registered_query_id=evidence.registered_query_id,
    query_jacobian_id=evidence.query_jacobian_id,
    grouping_rule_id=evidence.grouping_rule_id,
    source_roster_id=evidence.source_roster_id,
    score_definition_id=evidence.score_definition_id,
    source_score_table_id=evidence.source_score_table_id,
    interval_method_id=evidence.interval_method_id,
    simultaneous_interval_id=evidence.simultaneous_interval_id,
    source_evidence_set_id=evidence.evidence_set_id,
    confidence_level=evidence.confidence_level,
    minimum_source_groups=10,
    registered_candidate_causes=evidence.registered_candidate_causes,
    minimum_improvement=0.01,
    tie_tolerance=0.005,
    ambiguity_fallback=PhysicalCauseAmbiguityFallback.READOUT_DISCREPANCY,
)

selected, decision = select_physical_cause_v2(
    baseline_belief,
    [(state_belief, state), (discrepancy_belief, discrepancy)],
    policy,
    evidence,
)
```

The selector first verifies all source bindings and only then projects the V2
policy to the exact V1 routing rule. The returned belief is the exact input
candidate or the exact caller-owned baseline object. It is never reconstructed
from approximately equal arrays.

## Simultaneous inference requirement

V2 does not prescribe one statistical procedure, but the registered
`interval_method_id` and `simultaneous_interval_id` must identify an interval
that accounts for the complete candidate family. Suitable examples include:

- a max-t or bootstrap-t simultaneous upper-bound procedure;
- a familywise randomization interval;
- nested source selection and evaluation splits; or
- a preregistered cross-fitting procedure with a simultaneous final bound.

Separate pointwise confidence intervals for several causes are not a
simultaneous candidate-family certificate merely because each interval has the
same nominal confidence level.

## Relationship to V1

V2 deliberately reuses:

- `PhysicalCauseCandidateV1` for cause-specific complete-belief semantics;
- `PhysicalCauseDecisionPolicyV1` as the internal routing projection; and
- `select_physical_cause` for exact selection and ambiguity behavior.

The V1 `source_evidence_id` is set to the V2 evidence-set identity. The V2
decision stores and validates the exact projected V1 decision. Thus, evidence
hardening does not fork selection semantics or change existing V1 artifacts.

## Scientific boundary

A passing V2 decision establishes that one target-facing routing decision used
the registered simultaneous source comparison without substituting domain,
query, Jacobian, group roster, score, interval, candidate, or target evidence.
It does not establish that the selected label is the unique physical cause.

Physical-state and physical-parameter claims still require prospective
intervention transport and independent physical evidence. A source-bound
selection does not by itself establish unseen-object transfer, calibrated
uncertainty, provider competence, deployment safety, Causal4D benefit, or state
of the art.
