# Dynamic TAPNext++ Provider V1

## Status

This protocol was locked before implementation, cohort selection, tracker
execution, or outcome opening. It defines a new experiment; it does not modify
the failed one-case TAPNext++ competence result or authorize reuse of any
held-v8, Prob4D, or MolmoMotion object.

The complete method contract is
`configs/sota/deform360_dynamic_tapnextpp_provider_v1.json`. The hash-only
exclusion union contains 100 physical objects and has canonical SHA-256
`cf472da17400ce2191d0af9b0b25788fd27b5e5c9976293e2814d2604d8da684`.
The separately hashed source scorer is
`configs/sota/deform360_dynamic_tapnextpp_source_evaluation_v1.json`; the
cohort lock binds both files before any provider outcome.

Before cohort selection, a separately owned source campaign reported 12 newly
opened physical objects that were absent from the original 82-object union.
Exclusion amendment 1 adds those objects by hash only. No source or target
outcome from this protocol had been opened, and no method, threshold, or gate
changed.

A second pre-cohort audit found that the same campaign had opened or
technically dispositioned 18 queued objects, whereas amendment 1 covered only
the 12 retained cohort objects. Amendment 2 adds the six remaining objects.
Again, no provider outcome, cohort choice, method, threshold, or gate had been
opened or changed.

A third pre-cohort amendment pins the public Deform360 dataset revision to
`7fea8e20231a47641d1d2bc8791920ec4e62ec5e` and makes candidate selection
executable. After hash exclusion and metadata enum validation, the staging queue
takes the first 12 public identities in each of three name-only morphology
strata and interleaves them. The three strata are sheet-like names containing
`cloth`, non-`cloth` names below numeric prefix 138, and non-`cloth` names at or
above 138. This amendment was made before episode payload download or cohort
selection and does not change the tracker, updater, or gates.

The resulting metadata preflight contains 90 non-excluded public objects: 89
pass the metadata contract and one fails its exact enum check. Its canonical
SHA-256 is
`c80b3db5057c3c24bdd8f6e9dd7b2b8eeb529d980cdc8f4245fec3052d87a939`.
The 36-object staging queue contains 12 objects per stratum and is locked at
canonical SHA-256
`8afcfe64fe62af36a303e376a8c2f1fb78fc855446eadcd6687c13e12a650bdc`.
It is bound to implementation commit
`a41a581f27097ed04a2a0ec4d58cffcb6963a2b6`. At queue creation, no queued
episode media, processed geometry, future trajectory, or evaluation metric had
been read.

A fourth pre-cohort amendment narrows the candidate from a graph position and
velocity update to a recursive readout-discrepancy belief. Implementation review
showed that the former wording exceeded the source-supported method: the
available open evidence supports a guarded spatial discrepancy update, but not
yet a dynamically admissible Warp state update. This amendment was made before
cohort locking, provider execution, or any source or target outcome. The
physical and persistence trajectories remain immutable backbones in V1.

After the initial cohort lock, the first target-free physical-backbone smoke
stopped before simulation because that runner invoked the generic source
validator without the dynamic protocol's exact admission configuration.
Amendment 5 routes the physical runner through the same tested dynamic
admission normalizer as the provider. No rollout, tracker result, future object
payload, or metric existed, and neither the method, gates, nor deterministic
cohort membership changed. The cohort lock is regenerated solely to bind the
corrected code and protocol hashes.

The next target-free smoke passed admission and materialized only the permitted
frame-zero graph/state files, then stopped before an accepted twin summary
because its provenance snapshot still requested the obsolete
`state_coordinates` key. Amendment 6 binds the already frozen
`candidate_coordinates` field instead. No Warp rollout, tracker output, future
object payload, or metric existed. Cohort membership, method, and gates remain
unchanged, and the lock is regenerated again to bind this compatibility fix.

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

The pinned TAPNext++ implementation cannot append queries to an existing
recurrent state. The provider therefore runs each camera/birth-wave pair as an
independent causal rollout, initialized on the birth frame and stopped on the
associated update frame. Pixels before birth and after update remain explicitly
inactive. This is an execution constraint of the frozen tracker, not an
outcome-selected reset policy.

Low-motion episodes are not removed from the final benchmark. The physical and
action support gates instead return the unchanged backbone exactly when an
update has no defensible headroom.

## Observation Contract

Only triangulations with at least three inlier cameras enter the discrepancy
likelihood. Two-view estimates may preserve tracker continuity and association
hypotheses, but cannot update the physical prediction.

Each accepted observation is exported as `ObservationBeliefV1` with:

- metric local triangulation covariance;
- assignment-mixture covariance;
- one coherent 3-D bias factor per update interval;
- correlation groups that cap each tracked trajectory at three effective
  samples;
- association probability separate from perception reliability; and
- residual-independent reliability from tracker visibility, masks, depth,
  reprojection, view redundancy, and association entropy.

The physical innovation is evaluated exactly once by the covariance-aware
Student-t robust update. If no claim-bearing row survives, the observation
artifact is explicitly marked as prior-only and the candidate returns the
selected backbone exactly.

## Discrepancy And Safety

For identity \(i\), born at frame \(b\) and revisited at causal update \(t\),
the candidate constructs

\[
\hat y_{i,t}
= x^{\mathrm{phys}}_{i,b}
+ \left(\hat o_{i,t}-\hat o_{i,b}\right).
\]

This birth-relative displacement cancels a camera/world offset that is constant
within the birth wave. Its covariance is conservatively bounded as
\(2(\Sigma_{i,b}+\Sigma_{i,t})\), rather than treating the two observations as
independent. A pairwise-distance consensus gate first checks whether the
accepted identities describe one coherent deformable correspondence set. The
innovation relative to the selected physical or persistence backbone is then
processed once by a covariance-aware robust update and decoded through a
global-plus-local RBF discrepancy field.

The resulting mean is added in readout space over the following interval. It
does not modify graph positions, velocities, physical parameters, contact, or
the Warp rollout. The physical/action model determines where observations are
worth seeking, but it does not manufacture their perception reliability.
Unidentifiable or rejected updates retain the selected backbone exactly.

The candidate is compared with the unchanged physical/persistence backbone
using a source-cross-fitted upper confidence bound on regret. A nonnegative,
missing, or invalid bound returns the backbone byte-for-byte. Thus abstention is
part of the method, not a technical failure or a post-hoc choice.

## Prospective Evidence Order

The source and target object partitions are locked together from metadata and
frame-zero admission before source outcomes:

```text
100-object hash exclusion union
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

Provider competence is evaluated on identities with claim-bearing
triangulations at both birth and update. Its persistence comparator holds the
provider's own birth triangulation fixed, and covariance is calibrated
leave-one-object-out. Downstream assimilation is scored only on identities that
were never queried. Four deterministic leave-two-object folds fit a
baseline-relative regret certificate from target-free support, correspondence,
reliability, and correction-scale features. An interval is admitted only when
its upper regret bound is strictly negative; otherwise it is the selected
backbone byte-for-byte.

The executable post-admission lock requires exactly one source-only disposition
for every one of the 36 queued objects before it selects a cohort. It takes the
first admitted 7 sheet, 7 compact, and 6 complex objects in frozen queue order,
interleaves the strata, assigns the first 8 objects to source, and seals the
remaining 12 as target. This yields a 3/3/2 source split and a 4/4/4 target
split. If any stratum lacks its quota, this provider version stops; a technical
failure cannot be silently replaced by a later object.

Each ordinary prediction seal binds the query schedule, provider archive,
observation belief, assimilation archive, narrow hidden-identity prediction
input, runtime report, exact code revision, and environment digest. A tracker
or backend runtime failure is therefore a registered technical disposition,
not a model prediction.

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

## Implemented Boundaries

The opt-in implementation keeps four quantities distinct:

- birth association probability from local mask/depth geometry;
- residual-independent perception reliability;
- local metric covariance using independent camera clusters and an
  equal-weight covariance-intersection approximation; and
- one coherent 3-D camera-bias factor per update interval.

Exact or near-exact duplicate camera poses form one information cluster. Two
independent views can preserve a tracker proposal but cannot create a
claim-bearing likelihood row. Assignment ambiguity contributes metric
covariance, while the PhysTwin innovation is formed once downstream and enters
only the robust likelihood.

Pre-lock admission now rejects malformed metadata enums, incomplete streams,
short episodes, camera panels below eight eligible views, and physical
geometry below the 128-node backend minimum. Ordinary prediction seals and
retained technical failures are separate artifact kinds and are counted
separately by the source completeness barrier.
