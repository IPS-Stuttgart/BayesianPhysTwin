# Provider-neutral guarded inference API v2

## Purpose

`bayesian_phystwin.inference.v2` is the small provider-neutral orchestration
surface for BayesianPhysTwin. It composes three responsibilities without
collapsing their scientific boundaries:

1. a caller-supplied candidate factory constructs one complete candidate belief;
2. a caller-supplied guard policy decides whether that candidate is admissible;
3. the existing complete-belief router selects the candidate or returns the exact
   caller-owned baseline object.

The v2 session does not contain a default estimator, default provider, default
regret threshold, or default covariance interpretation. Prob4D-specific strict
candidate construction remains available through
`bayesian_phystwin.inference.v1`. Direct observations, replay providers, future
probabilistic perception systems, and controlled synthetic producers can use the
same v2 orchestration contract without importing Prob4D types.

## Minimal use

```python
from bayesian_phystwin.inference.v2 import (
    CandidateProposalV1,
    InferenceSession,
)


def build_candidate(prior, observation, *, context):
    inference, candidate_belief = provider.infer(
        prior=prior,
        observation=observation,
        context=context,
    )
    return CandidateProposalV1(
        inference=inference,
        candidate_belief=candidate_belief,
        metadata={"provider": provider.provider_id},
    )


def choose_guard(inference, baseline, candidate, *, context):
    return frozen_guard.decide(
        inference=inference,
        baseline_belief=baseline,
        candidate_belief=candidate,
        context=context,
    )


session = InferenceSession(
    session_id=frozen_protocol_id,
    candidate_factory=build_candidate,
    guard_policy=choose_guard,
    metadata={"protocol_id": frozen_protocol_id},
)
result = session.assimilate(
    prior=baseline_belief,
    observation=observation_belief,
    context={"case_id": case_id, "domain_id": domain_id},
)

assert result.selected_belief is (
    baseline_belief if result.exact_fallback else result.candidate_belief
)
```

`session_id` is an explicit lowercase SHA-256 identity supplied by the caller.
It should bind the frozen estimator, provider, guard, covariance semantics, and
other protocol choices used by the session. Python callable identity is not
stable scientific provenance and is therefore not guessed from function names
or object representations.

## Contracts

### `CandidateProposalV1`

A proposal binds:

- one `GuardedCandidateInference`, including its candidate identity and numerical
  admissibility;
- one complete candidate belief with a lowercase SHA-256 `artifact_id`; and
- finite JSON metadata.

Its `proposal_id` content-addresses those three elements. A proposal remains
undeployed: constructing it does not imply that a guard accepts it or that its
covariance is calibrated.

### `SessionCandidateFactory`

The candidate factory receives the baseline belief, an arbitrary observation
object, and immutable JSON context. It returns exactly one
`CandidateProposalV1`. The observation type is provider-owned; the session does
not require `ObservationBeliefV1` or any other provider-specific class.

### `SessionGuardPolicy`

The guard receives the inference record, baseline belief, candidate belief, and
the same immutable context. It must return a
`CompleteBeliefGuardDecisionV1` that binds the exact baseline and candidate
artifact identities and agrees with the inference admissibility flag.

The session supplies no application outcome or loss to either policy. A
prospective protocol must still ensure that context does not contain forbidden
target information.

### `InferenceSession`

`InferenceSession.assimilate` validates the prior, freezes the application
context, invokes the two caller-owned policies, and delegates routing to
`finalize_guarded_update`. The returned record binds:

- the explicit session identity;
- the content-addressed candidate proposal;
- session, proposal, and application metadata;
- the guard decision and complete-belief selection; and
- exact selected-object identity.

Every ordinary rejection therefore returns the exact input baseline object. The
session does not reconstruct a nominal baseline from zero correction
coefficients, copy the baseline, or select only part of a belief.

## Prob4D adapter pattern

The strict Prob4D path remains deliberately explicit:

```python
from bayesian_phystwin.inference.v1 import infer_prob4d_candidate
from bayesian_phystwin.inference.v2 import CandidateProposalV1


def build_prob4d_candidate(prior, observation, *, context):
    inference = infer_prob4d_candidate(
        observation,
        context["linearization"],
        physical_prediction_xyz_m=context["physical_prediction_xyz_m"],
        config=context["solver_config"],
    )
    candidate_belief = context["complete_belief_builder"](
        prior,
        inference,
    )
    return CandidateProposalV1(
        inference=inference,
        candidate_belief=candidate_belief,
        metadata={"provider": "prob4d"},
    )
```

Claim-bearing experiments should normally pass content identities in context
rather than large numerical arrays. The abbreviated example above illustrates
the adapter shape only; it is not a registered scientific protocol.

## Failure behavior

The API fails closed when:

- the session, inference, or belief identity is not a lowercase SHA-256 digest;
- a policy is not callable;
- proposal or context metadata is not finite JSON;
- the candidate factory returns another type;
- the guard returns another type;
- the guard does not bind the exact baseline and candidate beliefs;
- guard and inference admissibility disagree; or
- the selected object contradicts the complete-belief routing record.

Exceptions from provider or guard code are not converted into acceptance. A
higher-level application may catch such exceptions and retain its baseline, but
that application behavior must be explicit and separately recorded.

## Versioning boundary

`bayesian_phystwin.inference.v1` remains the frozen strict Prob4D candidate and
finalization surface for the 0.4 compatibility line. V2 is additive and does not
change any v1 symbol, estimator, protocol, artifact identity, or scientific
result.

The exact v2 export inventory is pinned in
`api/inference-session-public-api-v2.json`. New provider-specific estimator
arguments do not belong in this namespace; they should remain in a provider
adapter. Removing or changing a v2 contract requires a new versioned namespace.

## Scientific boundary

A valid v2 session record establishes orchestration, binding, and exact-fallback
behavior. It does not establish provider competence, physical-state validity,
covariance calibration, unseen-object transfer, bounded harm under an untested
domain shift, Causal4D intervention benefit, deployment safety, or state of the
art.
