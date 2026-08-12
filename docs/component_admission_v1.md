# Separate mean and covariance admission v1

## Purpose

`bayesian_phystwin.inference.components_v1` separates two decisions that answer
different statistical questions:

- whether a candidate point mean may replace the deterministic reference; and
- whether a frozen covariance treatment may be attached to a registered mean.

The module composes existing evidence. It does not fit a mean, calibrate a
covariance, choose a threshold, or inspect an outcome. Each routing result reuses
one exact caller-owned belief object so rejection cannot reconstruct or partially
modify the fallback.

The module is an explicit versioned integration surface. It is deliberately not
re-exported by `bayesian_phystwin.inference.v1`, whose existing exact symbol
inventory remains unchanged.

## Inputs

`BeliefComponentAdmissionPolicyV1` binds five complete belief arms:

1. exact physical fallback;
2. deterministic reference;
3. admitted mean with reference covariance;
4. deterministic reference mean with admitted covariance; and
5. the full admitted mean-and-covariance belief.

It also binds the common domain, exact-fallback policy, reference covariance
policy, candidate covariance policy, and whether mean-only or covariance-only
routing is permitted. All identities are lowercase SHA-256 digests and all five
belief-arm identities must be distinct.

`compose_belief_component_admission` consumes:

- one `CompleteBeliefGuardDecisionV1` for point-mean inference and regret;
- one `QueryCovarianceTreatmentDecisionV1` for query-space covariance value;
- a separately frozen per-candidate covariance-admissibility flag;
- a common-prerequisite decision; and
- deterministic-reference support.

The composer verifies that both upstream decisions bind the exact policy domain,
arms, covariance policies, and fallback policy. A downstream positive decision
cannot rescue an upstream mismatch or negative decision.

## Routing table

The default policy permits covariance-only routing and rejects mean-only routing.

| Common/reference gate | Mean gate | Covariance gate | Result |
| --- | --- | --- | --- |
| fail | any | any | exact physical fallback |
| pass | pass | pass | full mean-and-covariance belief |
| pass | fail | pass | covariance-only belief, when permitted |
| pass | pass | fail | mean-only if permitted; otherwise reference |
| pass | fail | fail | deterministic reference |

The covariance gate is the conjunction of the query covariance treatment
decision and the separately frozen candidate-specific admissibility decision.
The mean gate is the conjunction of numerical inference admission and the point
regret guard.

## Exact-object fallback

`route_belief_component_admission` receives all five complete belief objects. It
validates each object's `artifact_id` against the frozen policy and returns the
exact object registered for the selected mode. It never copies arrays, combines
fields from different beliefs, or creates a zero-correction approximation.

This matters because a complete belief may contain state, parameters,
discrepancy, nuisance variables, weights, covariance factors, and provenance.
Those fields must remain mutually consistent under both acceptance and fallback.

## Example

```python
from bayesian_phystwin.inference.components_v1 import (
    BeliefComponentAdmissionPolicyV1,
    compose_belief_component_admission,
    route_belief_component_admission,
)

policy = BeliefComponentAdmissionPolicyV1(
    common_domain_id=common_domain_sha256,
    exact_fallback_arm_id=physical_fallback.artifact_id,
    deterministic_reference_arm_id=last_residual.artifact_id,
    mean_candidate_arm_id=mean_candidate.artifact_id,
    covariance_candidate_arm_id=covariance_only.artifact_id,
    full_belief_arm_id=full_candidate.artifact_id,
    exact_fallback_policy_id=fallback_policy_sha256,
    reference_covariance_policy_id=reference_covariance_sha256,
    candidate_covariance_policy_id=candidate_covariance_sha256,
)

decision = compose_belief_component_admission(
    policy,
    point_guard_decision,
    query_covariance_decision,
    covariance_candidate_admissible=True,
    common_prerequisites_admissible=True,
    reference_supported=True,
)

result = route_belief_component_admission(
    decision,
    exact_fallback_belief=physical_fallback,
    deterministic_reference_belief=last_residual,
    mean_candidate_belief=mean_candidate,
    covariance_candidate_belief=covariance_only,
    full_belief=full_candidate,
)
selected = result.selected_belief
```

## Scientific boundary

A valid decision establishes content identity, cross-artifact consistency, and
exact-object routing only. It does not establish provider competence, calibrated
raw covariance, unseen-object transfer, safe deployment, Causal4D intervention
benefit, or state of the art. Those claims require the separately frozen fresh
physical study and its independent object/session-level evidence.
