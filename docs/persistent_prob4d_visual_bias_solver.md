# Persistent recursive Prob4D visual-bias inference

## Purpose

A recursive Prob4D stream may contain several causally disjoint observation
updates that share one source-calibrated camera or provider bias. The bias prior
must be introduced once and the same posterior state must be carried through all
later updates.

For update `k`, the linearized observation model is

```text
z_k - h_k(x_k) = H_k delta_x_k + B_k b + epsilon_k,
b ~ N(0, Sigma_b).
```

The physical block `delta_x_k` may include explicit Prob4D gauge variables and
any other variables admitted by the physical linearization. `b` is one
persistent visual-bias state. `epsilon_k` contains only conditional local point
noise; the gauge and visual-bias covariance must not be added to it again.

`bayesian_phystwin.persistent_prob4d_visual_bias` provides a content-addressed
linear-Gaussian reference solver for this boundary.

## Exact prior representation

The solver uses the symmetric positive-semidefinite root

```text
L L' = Sigma_b
```

and represents

```text
b = L u,   u ~ N(0, I).
```

This introduces the source-calibrated prior exactly once when the run starts.
The latent representation also supports singular `Sigma_b`: null directions in
provider coefficient space remain exactly zero rather than receiving an
artificial numerical variance.

The retained `PersistentVisualBiasBeliefV1` contains:

- the physical mean;
- the persistent bias-latent mean;
- one complete joint physical/bias covariance, including cross-covariance;
- the exact visual-bias covariance root;
- the stream-binding, bias-model, and physical-domain identities; and
- immutable content-addressed array descriptors.

Provider-space bias moments are available through `provider_bias_mean` and
`provider_bias_covariance`.

## Matrix-free measurement update

`propose_persistent_visual_bias_update(...)` consumes one stream member at a
time. It reads each row's local bias scope and Jacobian and accumulates the small
joint information matrix. It does not materialize the complete
`N x 3 x bias_dimension` global design.

The caller supplies:

- an innovation evaluated at the current prior mean;
- the physical Jacobian for the admitted physical block;
- the conditional local `3 x 3` covariance for each row; and
- the exact physical-linearization artifact ID.

For nonlinear or iterated use, recompute the innovation and Jacobian at the
current selected belief before every call. The solver is a linearized update;
it does not silently reuse a residual evaluated at an earlier posterior.

The candidate records the ordered stream, factor-stream, observation-binding,
physical-linearization, and input-array identities. It also reports the
conditional innovation quadratic per dimension and the joint Gaussian
information gain. These values are diagnostics and do not replace a frozen
BayesianPhysTwin regret or observability guard.

## Transactional accept and exact fallback

Candidate construction and deployment selection are deliberately separate:

```python
candidate = propose_persistent_visual_bias_update(
    run,
    innovation_xyz=innovation_at_current_prior,
    physical_jacobian=physical_jacobian,
    conditional_covariance=conditional_point_covariance,
    physical_linearization_id=physical_linearization.artifact_id,
)
run = select_persistent_visual_bias_candidate(
    run,
    candidate,
    innovation_xyz=innovation_at_current_prior,
    physical_jacobian=physical_jacobian,
    conditional_covariance=conditional_point_covariance,
    accepted=guard_decision.accepted,
    reason=guard_decision.reason,
)
```

Before applying either the accepted or rejected decision, the selection
boundary independently revalidates the candidate against the live run. It
requires:

- the exact active visual-bias stream update, factor-stream update, and
  observation-binding IDs;
- the current prior-belief ID and update index;
- the same stream binding, visual-bias model, physical state domain, physical
  dimension, bias dimension, and covariance root in the proposed posterior;
- posterior lineage that names the same update index and physical-linearization
  artifact as the candidate;
- a posterior covariance that is a positive-semidefinite measurement
  contraction of the prior covariance; and
- an information-gain diagnostic that exactly matches the prior and posterior
  covariance determinants within the declared numerical tolerance; and
- exact reproduction of the complete candidate from the supplied innovation,
  physical Jacobian, conditional covariance, active stream member, live prior,
  and physical-linearization identity.

Selection requires the same update arrays used during proposal. It reruns the
canonical solver and compares the complete candidate identity before either an
accept or fallback event is committed. This rejects changed posterior means,
alternative contracting covariances, forged innovation diagnostics, and arrays
that do not reproduce the proposed update.

When accepted, the complete joint posterior becomes the next belief. When
rejected, the selected belief is the exact prior belief object. Neither the
physical state nor the persistent bias state learns from a rejected update.
The stream member is nevertheless recorded as consumed, so it cannot be replayed
later under a different decision.

`apply_persistent_visual_bias_update(...)` is the corresponding convenience
wrapper that proposes and selects in one call.

## Physical prediction between observations

`predict_persistent_visual_bias_run(...)` applies an affine transition to the
physical block and adds declared physical process covariance. The persistent
bias marginal is retained, while the physical/bias cross-covariance is
propagated by the physical transition.

A prediction event does not consume a Prob4D stream update. Its transition,
process covariance, and offset are content-addressed separately from the later
measurement event.

## Starting and updating a run

Start from an independently validated
`Prob4DVisualBiasStreamConsumptionBindingV1`:

```python
import numpy as np

from bayesian_phystwin.persistent_prob4d_visual_bias import (
    apply_persistent_visual_bias_update,
    predict_persistent_visual_bias_run,
    start_persistent_visual_bias_run,
)

run = start_persistent_visual_bias_run(
    stream_binding,
    physical_state_domain_id=complete_physical_state_domain_id,
    physical_mean=physical_mean,
    physical_covariance=physical_covariance,
    metadata={"protocol": "registered-recursive-prefix-v1"},
)

run, candidate = apply_persistent_visual_bias_update(
    run,
    innovation_xyz=innovation_at_current_prior,
    physical_jacobian=physical_jacobian,
    conditional_covariance=conditional_point_covariance,
    physical_linearization_id=physical_linearization.artifact_id,
    accepted=guard_decision.accepted,
    reason=guard_decision.reason,
)

run = predict_persistent_visual_bias_run(
    run,
    physical_transition=transition,
    process_covariance=process_covariance,
    physical_offset=physical_offset,
    transition_id=transition_artifact_id,
)
```

The run rejects stale candidates, reordered or replayed stream members, a
changed stream binding, a changed visual-bias model, a changed physical state
domain, a changed covariance root, forged member bindings, inconsistent
posterior lineage, and a broken belief-event chain.

## Compatibility boundary

The existing one-shot V2 visual-bias update and its artifact identities remain
unchanged. The stream binding also remains fail-closed when callers invoke its
legacy `require_claim_bearing_execution()` method on a multi-update stream.
That method protects the one-shot path from accidentally duplicating the prior.

Multi-update execution is authorized only by constructing a
`PersistentVisualBiasRunV1` from the validated binding and using the persistent
solver API. The run records the solver schema and version in every candidate and
result identity.

Causal4D should receive only the selected physical belief and its lineage. Raw
Prob4D factors, gauge likelihoods, or visual-bias likelihoods must not be applied
again downstream.

## Contract evidence

Focused regressions cover:

- recursive partitioning versus one stacked linear-Gaussian update;
- singular source-calibrated bias covariance;
- exact object fallback and stale-candidate rejection;
- physical prediction with retained bias marginal and propagated
  cross-covariance;
- a negative control showing overconfidence when the bias prior is
  reinstantiated;
- matrix-free operation without constructing the complete global design;
- forged factor-update and observation-binding rejection;
- posterior state-domain, covariance-root, and lineage mismatch rejection;
- noncontracting covariance and misreported information-gain rejection;
- solver-replay rejection of changed posterior means, alternative contracting
  covariances, forged innovation diagnostics, and substituted update arrays;
- irreversible NumPy immutability for retained and derived belief arrays; and
- content-identity tamper detection.

The authoritative workflow is read-only, checks out the exact reviewed
BayesianPhysTwin head and merged Prob4D producer revision
`e37c3d50d4a07a2c3760389e79d59b0ac9402dc4`, installs both packages into a
fresh Python environment, and runs focused plus adjacent producer-consumer
contracts before verifying a clean repository tree.

These are solver and contract properties. They do not establish real Prob4D
provider competence, complete camera-bias coverage, calibrated target
uncertainty, physical-query improvement, deployment safety, Causal4D benefit,
or state of the art. Those claims still require a frozen independent physical
object or acquisition-session experiment with retained failures and exact
fallback.
