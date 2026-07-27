# Dynamic TAPNext++ Provider V1

## Status

This protocol is locked before implementation, cohort selection, tracker
execution, or outcome opening. It defines a new experiment; it does not modify
the failed one-case TAPNext++ competence result or authorize reuse of any
held-v8, Prob4D, or MolmoMotion object.

The complete method contract is
`configs/sota/deform360_dynamic_tapnextpp_provider_v1.json`. The hash-only
exclusion union contains 82 physical objects and has canonical SHA-256
`5454f86c6b434c1f10b7762c3ed00e887b6184f03b0e61ce9b02651d2fed0e66`.

## Hypothesis

The first TAPNext++ control was accurate on admitted observations but supported
only 52 of 76 eligible point-frames. The new hypothesis is that support can be
raised without weakening geometric checks by changing when and where queries
are born:

1. PhysTwin and the measured action propose graph identities that should move
   and remain visible.
2. Three causal birth waves replace points that become occluded.
3. Eight cameras provide enough redundancy for a strict three-view update.
4. A D-optimal selector spreads observations across graph modes rather than
   clustering them near one contact.

This is active observation scheduling around a fixed tracker, not tracker
tuning.

## Causal Schedule

Deform360 updates remain fixed at frames 19, 38, and 57. For each update,
TAPNext++ queries are born at three earlier causal frames:

| Update | Query births | Scored future |
| ---: | --- | --- |
| 19 | 0, 6, 12 | `[20,38)` |
| 38 | 20, 26, 32 | `[39,57)` |
| 57 | 39, 45, 51 | `[58,76)` |

Each birth introduces eight previously unused material identities. Selection
uses only the physical rollout, action, frame-zero attachment, calibration, and
causal images available by that birth. Every measured identity from every
birth wave is permanently removed from future scoring.

Low-motion episodes are not removed from the final benchmark. The physical and
action support gates instead return the unchanged backbone exactly when an
update has no defensible headroom.

## Observation Contract

Only triangulations with at least three inlier cameras enter the state
likelihood. Two-view estimates may preserve tracker continuity and association
hypotheses, but cannot update the physical state.

Each accepted observation is exported as `ObservationBeliefV1` with:

- metric local triangulation covariance;
- assignment-mixture covariance;
- one coherent 3-D bias factor per update interval;
- correlation groups that cap each tracked trajectory at three effective
  samples;
- association probability separate from perception reliability; and
- residual-independent reliability from tracker visibility, masks, depth,
  reprojection, view redundancy, and association entropy.

The physical innovation is evaluated exactly once by the grouped robust
Student-t mixture likelihood.

## State And Safety

The updater estimates graph-mode position and velocity at the causal endpoint.
Measured robot motion and registered contact support anchor global modes, while
camera bias remains an explicit nuisance variable. Modes that cannot be
separated from bias retain their physical prior.

The candidate is compared with the unchanged physical/persistence backbone
using a source-cross-fitted upper confidence bound on regret. A nonnegative,
missing, or invalid bound returns the backbone byte-for-byte. Thus abstention is
part of the method, not a technical failure or a post-hoc choice.

## Prospective Evidence Order

The source and target object partitions are locked together from metadata and
frame-zero admission before source outcomes:

```text
82-object hash exclusion union
-> source-only admission
-> locked 8-source / 12-target object split
-> implementation and environment lock
-> source provider prediction seals
-> source provider competence outcome
-> source disjoint-identity assimilation seals
-> source assimilation outcome
-> freeze conformal and regret artifacts
-> target prediction seals
-> completeness barrier
-> one target outcome-opening operator
```

Either source gate failing stops the protocol before target outcomes. Technical
failures remain in the denominator and are never replaced.

## Claim Boundary

Provider competence does not establish a better digital twin. Source
assimilation establishes only source transfer. A state-of-the-art statement
requires all of:

1. untouched target objects;
2. authoritative Deform360 evaluator parity;
3. local reproduction of the strongest eligible baseline;
4. improvement on both primary metrics;
5. favourable object-cluster confidence intervals;
6. at least 8 of 12 joint target-object wins;
7. no object regression above 10%; and
8. complete provenance and exact-fallback validation.

Without evaluator parity, the strongest permitted statement is transfer under
the explicitly declared hidden-identity metrics.
