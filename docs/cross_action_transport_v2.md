# Chronological cross-action transport v2

`CrossActionProtocolV2` is the target-closed BayesianPhysTwin evidence contract for
the frozen Causal4D same-grasp multi-action acquisition.

It exists because v1 and v2 answer the same scientific question under different
acquisition structures:

- v1 requires the complete Cartesian action matrix inside every target session;
- v2 requires exactly one preregistered chronological source-to-target pair per
  independent physical session.

The v1 contract is unchanged.

## Scientific question

The claim-bearing test asks whether source-prefix information that is admitted as
physical state/parameter information transports through the simulator to a
*different held-out action*.

For each complete session `s`, with the lower-is-better frozen trajectory score
`S`, v2 evaluates

```text
G[s, m] = S[s, physical_fallback] - S[s, m].
```

The independent statistical unit is the complete physical session. Frames,
points, coordinates, views, time steps, action labels, and posterior samples are
nested observations.

A positive result requires guarded physical transport to improve over all three
scientifically relevant references:

1. unchanged `physical_fallback`;
2. `discrepancy_only`, which carries source discrepancy without a physical-state
   interpretation; and
3. `last_residual`, the matched deterministic persistence comparator.

The four primary arms are fixed by the v2 contract. A target-side challenger
tournament is not representable.

## Frozen Causal4D binding

The first v2 contract is deliberately specific to

```text
configs/causal4d/sloth_multi_action_v1.json
SHA-256 6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968
```

The registered session roster binds, for every physical session:

- source and target execution IDs;
- source and target action IDs;
- one contact/stratum identity;
- one information-order identity; and
- the assertion that the source execution precedes the held-out target.

Each execution may appear exactly once. Reversing target-to-source order is not a
valid v2 prediction and fails closed.

The final protocol must additionally bind the exact source policy,
BayesianPhysTwin/Causal4D model stack, numerical environment, candidate family,
support admission, single- and multi-action query-identifiability evidence,
nonlinear-closure evidence, guard, score, target-access policy, and technical
failure policy before confirmatory execution 1.

## Prediction-first information order

For each registered session, every arm emits one
`SealedTransportPredictionV2`. The record content-binds:

- the exact registered chronological pair;
- baseline, rejected candidate where applicable, and selected complete belief;
- the source evidence;
- the complete prediction artifact and batch;
- the exact BayesianPhysTwin revision; and
- the fact that prediction was sealed before target access.

`EXACT_FALLBACK` must select the exact baseline belief. During scoring, a rejected
arm must also obtain exactly the physical-fallback score on the same target.
This catches partial-update or scorer drift that belief IDs alone would miss.

Post-access `TransportScoreRowV2` records forbid target-side model or threshold
selection and require one scorer and one target-access attestation over the
complete evaluation.

## Complete accounting

Every frozen session must be in exactly one of three states:

- scored;
- preregistered exclusion; or
- retained technical failure.

A scored session must contain every primary arm. Exclusions and technical
failures reduce the number of independent units and cannot silently be replaced.

## Decision rule

`physical_transport_supported` is returned only when all frozen conditions pass:

- enough independent scored sessions remain;
- the bootstrap lower bound for guarded-physical gain exceeds the registered
  fallback margin;
- the bootstrap lower bound for guarded physical minus `discrepancy_only`
  exceeds the registered physicality margin;
- the bootstrap lower bound for guarded physical minus `last_residual` exceeds
  the registered matched-comparator margin;
- the Wilson upper bound for harmful sessions is below its registered maximum;
- the Wilson upper bound for harmful *selected* physical updates is below its
  registered maximum; and
- at least one guarded physical candidate is actually selected.

Otherwise the result is a registered negative or, when too few independent
sessions remain, an insufficient-session result.

## What the pre-acquisition tests establish

The synthetic unit tests establish contract behavior only. In particular they
verify that:

- roster ordering cannot change content identity;
- reverse chronological reuse fails closed;
- missing arms fail closed;
- exclusions and technical failures are completely accounted;
- exact fallback must be identity- and score-exact;
- a physical candidate must beat both nonphysical references; and
- duplicated execution IDs cannot enter the frozen roster.

These tests do **not** constitute physical transport evidence.

## Claim boundary

A positive physical run supports bounded transport of the exact admitted
correction across the registered held-out actions on the frozen same-grasp
sessions. It does not establish unique physical causation, arbitrary-action or
unseen-object generalization, Prob4D provider competence, Causal4D intervention
recovery, calibrated raw covariance, deployment safety, or state of the art.

Issue `#785` is the governing research registration. The physical result remains
blocked until the separately registered Causal4D acquisition is executed.
