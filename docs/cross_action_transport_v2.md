# Chronological sparse-pair cross-action transport v2

## Purpose

`CrossActionProtocolV2` is a separately versioned, target-closed extension of the
BayesianPhysTwin cross-action transport study. It is intended for causal physical
acquisitions in which one physical session exposes exactly one chronological
source action followed by one held-out target action.

Version 1 remains unchanged. V1 is the appropriate contract when every target
session contains a complete Cartesian action matrix. V2 must not be used to
reinterpret or weaken an existing V1 result.

The first intended binding is the frozen Causal4D sloth multi-action design:

- source design: `IPS-Stuttgart/Causal4D:configs/causal4d/sloth_multi_action_v1.json`;
- design SHA-256: `6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968`;
- 18 independent same-grasp physical sessions;
- 36 command executions;
- exactly one registered `pair_order=0 -> pair_order=1` source-to-target direction
  in each session; and
- no reverse same-session target-to-source reuse.

The checked-in sparse-pair roster is
`protocols/cross_action_transport/causal4d_sloth_multi_action_v1_sparse_pairs.json`.
It freezes the design mapping only. It is **not** the final runtime protocol
instance: the exact object registration, software/environment, query, scorer,
candidate family, admission policies, guards, and target-access identities still
have to be frozen before confirmatory execution 1.

## Publication-level question

The primary question is deliberately stronger than same-action residual
correction:

> Does information admitted from the first action in a physical grasp improve a
> different held-out action in the same grasp in a way that cannot be explained
> by carrying forward the latest residual or by discrepancy-only persistence?

The first claim-bearing comparison should remain small:

1. `physical_fallback` — the unchanged caller-owned physical prediction;
2. `last_residual` — the principal matched deterministic comparator;
3. `discrepancy_only` — source discrepancy persistence without a physical-state
   interpretation; and
4. `guarded_physical` — at most one preregistered physical candidate with exact
   fallback on rejection.

Additional state-only/state-parameter variants may be reported only as frozen
non-claim-bearing diagnostics. No target-side challenger tournament is permitted.

## Information order and evidence identity

Each `ChronologicalSessionPairV2` binds:

- physical session ID;
- source and target execution IDs;
- source and target action IDs;
- optional already-frozen contact/stratum labels; and
- a content-addressed `information_order_id`.

Every physical session must occur exactly once and every execution ID must occur
in exactly one pair. A scored prediction is valid only when its complete
source/target identity equals the registered pair byte-for-byte. Reversing the
same two executions therefore fails closed rather than creating an extra sample.

Every candidate prediction also binds the exact baseline, candidate and selected
belief identities, source/admission evidence, prediction batch and
BayesianPhysTwin revision. Predictions must be sealed before target access.
Rejected candidates must select the exact baseline belief.

## Statistical unit and primary test

The physical grasp session is the independent statistical unit. Frames, time
points, coordinates, views, contacts, action labels and repeated metric terms are
nested observations and must not increase the effective sample size.

For registered lower-is-better score `S`, the session-level gain for method `m`
is

```text
G[s,m] = S[s,physical_fallback] - S[s,m].
```

V2 performs one bootstrap over the complete scored session values. The guarded
physical arm must satisfy all frozen primary conditions, including:

- enough complete independent sessions;
- a positive lower confidence bound on gain over physical fallback;
- a positive lower confidence bound relative to `discrepancy_only`;
- a positive lower confidence bound relative to `last_residual`;
- enough actually accepted physical updates;
- the registered harmful-accepted-update risk bound; and
- exact fallback for every rejected physical update.

Scored, preregistered-excluded and technical-failure sessions are accounted for
separately and must partition the exact frozen roster. Missing sessions cannot be
silently removed.

## Harmful accepted-update certificate

A fallback is not an accepted physical update and therefore cannot dilute the
harm denominator. V2 computes harmful-update risk only over sessions where the
physical candidate was actually selected.

The registered implementation uses a one-sided exact Clopper-Pearson upper bound.
This has an important planning consequence at 95% confidence when zero harmful
accepted updates are observed:

- 13 accepted sessions: upper bound `0.2058167`, so a 20% cap cannot pass;
- 14 accepted sessions: upper bound `0.1926362`, so a 20% cap can pass;
- 18 accepted sessions: upper bound `0.1533176`; and
- 29 accepted sessions: upper bound `0.0981446`, the first point at which a 10%
  cap can pass.

Consequently the frozen 18-session Causal4D acquisition cannot support a 10%
harm claim at one-sided 95% confidence. A claim-bearing 20% cap requires at
least 14 accepted sessions with zero harmful accepted updates. Insufficiency is
a valid result and must not be repaired by treating nested measurements as
independent samples.

The protocol constructor rejects a harmful-update cap that is impossible even
under zero harms across the complete frozen session roster. This is a prospective
feasibility check, not an empirical result.

## Relationship to Causal4D

This analysis reuses the measurements from the preregistered Causal4D acquisition
without modifying that acquisition. It does not change:

- the 18-session / 36-execution schedule;
- command profiles, contacts or realization conditions;
- chronological information order;
- Causal4D's commanded/realized intervention analysis; or
- any Causal4D claim threshold.

Causal4D and BayesianPhysTwin therefore answer different questions on the same
physical acquisition. Causal4D tests intervention abduction and decision-relevant
prediction under its registered causal protocol. BayesianPhysTwin V2 tests
whether a source-admitted correction transports to a different action as
physical information rather than merely as residual persistence.

## Before confirmatory execution 1

The following must be completed before any confirmatory target outcome is opened:

1. merge and review the V2 contract/evaluator and adversarial tests;
2. verify the checked-in 18-pair design roster against the frozen Causal4D design;
3. complete the actual physical object/contact registration required by Causal4D;
4. freeze a content-addressed V2 runtime protocol containing the exact final
   BayesianPhysTwin/Causal4D software and environment identities;
5. bind one query/scorer, one candidate family, source support and identifiability
   policies, estimability policy and guard policy;
6. freeze feasible effect and harm thresholds, including the minimum accepted
   physical-session count; and
7. verify synthetically that missing, duplicated, reversed or target-informed
   evidence fails closed.

After target outcomes are opened, retain the registered positive, negative,
technical-failure or insufficient result without tuning the protocol on those
18 sessions.

## Claim boundary

A positive V2 result is evidence for bounded held-out action transport under the
exact frozen physical object/session roster and software stack. It is not evidence
for arbitrary actions or unseen objects, a unique physical cause, deployment
safety, Prob4D provider competence, Causal4D intervention benefit, or general
state of the art.
