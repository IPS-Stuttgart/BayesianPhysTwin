# Explain, transport, probe, or abstain

A predictive residual does not identify its own physical cause. The same factual
error can be compatible with state, material, contact, observation-gauge, and
persistent-discrepancy explanations. This module composes three separate
certificates without replacing the remaining ambiguity by a point label.

## Operational order

```text
residual within the registered noise radius
    -> no detectable correction; exact fallback
residual outside the registered cause-family span
    -> none of the above; exact fallback
cause family adequate and target invariant over all explanations
    -> transport the target correction
cause family adequate, target ambiguous, identifying probe available
    -> request the minimum-cost target-identifying probe; fallback now
cause family adequate, only a target subspace identifiable
    -> report the invariant component; fallback for the complete target
cause family adequate, target unresolvable by the roster
    -> abstain; exact fallback
```

The top-level implementation is
`bayesian_phystwin_experiments.explain_transport_probe_abstain_v1`.

## Cause-family adequacy

For stacked registered signatures `S` and whitened residual `r`, the adequacy
certificate decomposes

\[
r = SS^\dagger r + (I-SS^\dagger)r.
\]

If the orthogonal component exceeds the frozen radius, the only permitted
semantic result is `none_of_the_above`. The intervention selector is deliberately
not called in this state: adding probes inside an already inadequate cause family
would only refine the wrong model.

When the family is adequate, the complete registered explanation set remains

\[
\mathcal B(r)=S^\dagger r+\ker(S).
\]

`explain_and_transport` means unique only inside this registered local linear
family. It is not a claim that the natural physical cause is known.

## Transport before cause identification

For held-intervention target map `T`, the complete target effect is invariant over
the explanation set exactly when

\[
T\ker(S)=\{0\}.
\]

In that case the state machine returns `transport_without_cause` whenever the
coefficient explanation remains set valued. This is a first-class result rather
than a degraded attribution outcome: the pending physical effect is known even
though its latent label is not.

For vector targets, the transport certificate also returns the identifiable
projection orthogonal to `range(T ker(S))`. The top-level state machine never
mistakes that partial component for a complete target correction.

## Target-directed probing

If the complete target is not currently invariant, a finite candidate roster is
searched exactly for

\[
\arg\min_U \sum_{u\in U}c(u)
\quad\text{subject to}\quad
\ker(S_U)\subseteq\ker(T).
\]

Full cause identification requires the stronger condition
`ker(S_U) = {0}`. The report stores both costs and therefore quantifies how much
physical interaction is saved by identifying only the pending target.

A `probe_then_reassess` result does not deploy a correction. The current call
returns the exact caller-owned fallback. After an observed probe response, the
caller must append the registered response rows and rebuild the complete
adequacy/transport certificate. The module never substitutes a predicted probe
outcome.

## Controlled strict-separation study

`scripts/science/run_explain_transport_probe_abstain_controlled_v1.py` constructs
all operational phases in one deterministic study:

- unique registered explanation and transport;
- cause ambiguity but target-invariant transport;
- target-specific state/gauge and material probes;
- omitted cause / none of the above;
- unresolvable target / abstention; and
- no detectable error.

For the three targets in the ambiguous-family phase, one target needs no probe
and the other two need one query-specific probe each. Full cause identification
needs both informative probes for every target. The target-directed mean cost is
therefore one third of the full-cause cost.

## Required real-data progression

A real attribution claim needs a separately frozen information order:

1. construct response signatures on one source partition;
2. calibrate the adequacy radius and select target/probe/placebo definitions on a
   disjoint source partition;
3. seal every target correction, none-of-the-above decision, probe choice, and
   relation-breaking control before held future outcomes are read; and
4. score complete physical groups once.

Physical promotion additionally requires nonlinear replay closure and separation
from wrong-action, wrong-object/material, temporal-shift, identity-permutation,
and omitted-cause controls. Natural dataset labels must not be treated as a
complete causal ground truth merely because they are available.

## Claim boundary

The state machine is exact only for the supplied local linear cause family,
adequacy radius, target maps, candidate intervention designs, costs, coordinates,
and tolerances. It does not validate those physical models, identify unrestricted
causal structure, prove nonlinear closure, establish real held-intervention
transport, authorize a probe, provide deployment safety, or establish state of
the art.
