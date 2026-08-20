# Physical-cause selection v1

## Purpose

A predictive residual does not identify its own physical cause. The same camera
innovation can be compatible with a latent state displacement, a material or
controller-parameter change, coherent observation bias, unresolved readout/model
discrepancy, or no admissible update at all.

`bayesian_phystwin.physical_cause_selection_v1` makes that decision explicit. It
routes among already-constructed **complete beliefs** and never edits individual
mean, covariance, parameter, nuisance, or provenance fields after selection.
The candidate-construction process remains separate from this routing contract.

The registered interpretations are:

```text
baseline              unchanged caller-owned physical belief
observation_bias      nuisance/bias explanation without a physical update
readout_discrepancy   bounded predictive discrepancy without latent-state claims
physical_parameter    physically supported parameter/controller correction
physical_state        reachable and query-identifiable state correction
```

## Source-only evidence contract

Each nonbaseline candidate binds:

- its complete belief identity;
- the exact candidate-construction identity;
- numerical admissibility;
- a source-calibrated upper regret bound relative to the baseline;
- cause-specific support evidence; and
- finite immutable metadata.

The regret quantity is

```text
candidate proper score - baseline proper score
```

so lower is better. A candidate advances only when its upper regret bound is
**strictly below** the negative registered improvement margin. Therefore a score
tie retains the baseline.

Cause-specific evidence is fail closed:

| Cause | Required binding |
| --- | --- |
| observation bias | bias-design identity |
| readout discrepancy | discrepancy-model identity |
| physical parameter | parameter-sensitivity and identifiability-report identities |
| physical state | causal physical-response and identifiability-report identities |

Optional bias-design identities may accompany discrepancy, parameter, and state
candidates. They document the nuisance model against which those candidates were
constructed; they do not themselves establish identifiability.

## Ambiguity policy

Among source-supported candidates, the decision compares upper regret bounds.
Candidates within the registered tie tolerance are treated as physically
ambiguous. There are only two permitted ambiguity outcomes:

1. reuse the exact caller-owned baseline belief; or
2. select a near-best readout-discrepancy belief when the policy explicitly
   registers that fallback.

An ambiguous decision never promotes a physical-state or physical-parameter
candidate merely because it has the numerically smallest point estimate.
Selection is object-preserving: `select_physical_cause` returns the exact input
baseline or candidate object by identity; it does not reconstruct a belief from
the decision record.

## Example

```python
from bayesian_phystwin.physical_cause_selection_v1 import (
    PhysicalCause,
    PhysicalCauseAmbiguityFallback,
    PhysicalCauseCandidateV1,
    PhysicalCauseDecisionPolicyV1,
    select_physical_cause,
)

policy = PhysicalCauseDecisionPolicyV1(
    baseline_belief_id=baseline_belief.artifact_id,
    common_domain_id=common_domain_id,
    registered_query_id=query_id,
    source_evidence_id=source_crossfit_id,
    minimum_improvement=0.01,
    tie_tolerance=0.005,
    ambiguity_fallback=(
        PhysicalCauseAmbiguityFallback.READOUT_DISCREPANCY
    ),
)

state = PhysicalCauseCandidateV1(
    cause=PhysicalCause.PHYSICAL_STATE,
    belief_id=state_belief.artifact_id,
    construction_id=state_construction_id,
    upper_regret=-0.04,
    inference_admissible=True,
    reason="source-cross-fit",
    physical_response_id=physical_response_id,
    identifiability_report_id=identifiability_report_id,
)

discrepancy = PhysicalCauseCandidateV1(
    cause=PhysicalCause.READOUT_DISCREPANCY,
    belief_id=discrepancy_belief.artifact_id,
    construction_id=discrepancy_construction_id,
    upper_regret=-0.038,
    inference_admissible=True,
    reason="source-cross-fit",
    discrepancy_model_id=discrepancy_model_id,
)

selected_belief, decision = select_physical_cause(
    baseline_belief,
    [(state_belief, state), (discrepancy_belief, discrepancy)],
    policy,
)

# The two candidates are within the tie tolerance. The registered ambiguity
# policy therefore chooses the exact discrepancy belief, not the state belief.
assert selected_belief is discrepancy_belief
assert decision.selected_cause is PhysicalCause.READOUT_DISCREPANCY
assert decision.ambiguity_detected
```

## Relationship to existing guarded inference

This contract is an additional semantic layer, not a replacement for:

- numerical inference admission;
- `CompleteBeliefGuardDecisionV1` and exact complete-belief fallback;
- separate mean/covariance admission in `inference.components_v1`;
- semantic arm validation in `inference.component_beliefs_v1`; or
- `IdentifiabilityReportV1`.

A typical pipeline first constructs and validates each complete belief, then
produces source-only regret evidence, and finally applies the cause decision.
Downstream consumers receive only the selected complete belief and the
content-addressed decision record.

## Scientific boundary

A valid decision proves deterministic, source-bound routing under the registered
policy. It does not prove that the selected label is the unique data-generating
cause. It also does not establish provider competence, calibrated uncertainty,
unseen-object transfer, deployment safety, Causal4D benefit, or state of the
art. Independent physical evidence and separately frozen target evaluation
remain necessary for those claims.
