# Deform360 Event-Conditioned Online Belief V15

## Status

V15 is an implementation prelock for a genuinely new prospective study. No
V15 source object has been selected, no source or target outcome has been
opened, and no state-of-the-art claim is authorized.

V15 does not reopen an earlier arm:

- V12 is closed at `cf2b532` after its registered query-feasibility gate
  admitted only 2 of 8 cases.
- V13 passed adaptive camera-carrier feasibility but its fixed-identity
  tracker failed support.
- V14 sealed twelve predictions or exact fallbacks, admitted a causal-response
  event in only 1 of 12 objects, applied no update, and closed before source
  outcomes were opened.

The all-attempt held-v8 exclusion manifest can be used only as provenance for
a fresh protocol. It does not authorize a V12 rerun or a relaxed V12-V14
gate.

## New Question

The earlier Deform360 protocols selected an action-rich 81-frame window from
the robot trajectory before looking at object response. In many selected
windows the object then moved by less than one millimetre, making exact
persistence an exceptionally strong baseline.

V15 asks a different, prospectively declared question:

> After a deformable object has shown the earliest independently supported
> nonrigid response to measured contact and actuation, can a guarded
> Bayesian-PhysTwin update improve a fixed-horizon forecast without increasing
> regret relative to the unchanged baseline?

This is an event-conditioned prediction population. Every compared method
must receive the same branch frames and the same causal evidence. Event
prevalence, abstentions, and technical failures remain first-class results.
V15 results must not be substituted for unconditional Deform360 performance.

## Stage A: Model-Independent Event Selection

The implementation is
`deform360_event_conditioned_window_v15.py`. It scans a complete episode in
time order and stops at the earliest frame that passes a fixed event rule
while leaving a fixed future horizon.

The selector consumes two disjoint camera panels. Each panel supplies a small
set of gripper-excluded, cluster-level pairwise shape signatures. Pairwise
distances remove global translation and rotation, so actuator or camera
motion alone cannot define a deformable response event. Dense pixels do not
accumulate as independent evidence: camera support is a threshold, and
uncertainty is carried once per predeclared spatial cluster.

At a candidate branch frame, both panels must independently show:

1. sufficient camera support for enough spatial clusters;
2. no gripper overlap at either endpoint;
3. a nonrigid change above the metric floor;
4. signal above the registered covariance floor;
5. consistent direction and magnitude across panels; and
6. acceptable cross-panel discrepancy under a shared-bias variance.

The same causal interval must also contain tactile contact support and
measured actuator movement. Tactile and actuation establish causal support;
they supply neither metric object geometry nor observation reliability.

The population selector deliberately does not read:

- a physical prediction;
- a candidate belief update;
- future object observations after the selected branch;
- manual or hidden material identities;
- future point clouds or evaluation metrics.

Physical-model agreement belongs to update admission after the population and
branch frame have been fixed. This prevents the event definition from
selecting episodes merely because PhysTwin already predicts them well.

## Stage B: Guarded Belief Update

After Stage A selects a branch, Stage B may construct current material
associations and compare the physical and persistence backbones using only
evidence available through that branch. A candidate discrepancy update must
have:

- current independently anchored material identity or a set-valued
  association with assignment-mixture covariance;
- action-conditioned physical support;
- correlation-aware metric covariance and a shared-camera-bias nuisance;
- disjoint proposal and validation evidence;
- one robust innovation likelihood;
- a source-calibrated upper confidence bound on regret; and
- exact baseline fallback when any gate fails.

The privileged open-27 capacity audit is motivation only. Its dense
score-family identities are unavailable to a deployable method and cannot
enter V15. Stage B is not locked until an automatic current-identity carrier
passes an outcome-blind competence gate on fresh source objects.

## Causal Custody

For a selected branch frame `tau`, the selector hashes only arrays through
`tau`. Mutating any RGB-D-derived signature, tactile sample, or actuator sample
after `tau` cannot change the selection artifact. The forecast interval is
the half-open range

```text
[tau + 1, tau + 1 + H)
```

for a fixed registered horizon `H`.

If no event passes before the last branch that leaves `H` future frames, the
case is an explicit no-event abstention. Reserved tail values do not enter the
artifact. A later evaluation may either retain the unchanged baseline for
that case or report an event-conditioned denominator, but that choice must be
locked before outcomes and reported alongside the abstention count.

## Prospective Order

Before any V15 source selection:

1. build a new hash-only exclusion union containing every prior opened,
   reserved, selected, or technically dispositioned physical object;
2. freeze the cluster-signature preprocessor and camera-panel assignment;
3. pass synthetic deformable-response positives and rigid-motion,
   gripper-overlap, panel-disagreement, missing-contact, and coherent-bias
   placebos;
4. freeze an outcome-blind metadata, stream, camera, and backend preflight;
5. select a fresh source panel in immutable metadata order; and
6. seal all Stage A events or abstentions before any future identity or metric
   is opened.

The source feasibility gate should be evaluated before outcomes. Failure to
reach the registered event count closes V15 without threshold changes.
Passing feasibility permits evaluation of the already sealed predictions; it
is not itself positive evidence.

## Advancement Gates

The eventual source protocol should retain the existing falsification-first
requirements:

- complete prediction, exact fallback, abstention, and technical-failure
  accounting;
- minimum event prevalence fixed before source outcomes;
- object-balanced hidden-identity and Chamfer improvement over the unchanged
  selected baseline;
- joint object wins and a bounded worst-object regression;
- early, middle, and late horizon reports;
- NEES, coverage, interval width, and false-safe admissions;
- source-calibrated upper-regret coverage;
- bit-exact fallback; and
- a fresh-object target protocol locked only after every source gate passes.

All published baselines, including persistence and the physical model without
an update, must be re-evaluated on the same event-conditioned population.

## Current Implementation Evidence

The focused unit tests establish:

- earliest causal stopping;
- fixed future-horizon reservation;
- invariance to mutations after the selected branch;
- invariance to mutations in the reserved tail after a no-event scan;
- explicit no-event abstention;
- rejection of rigid-only shape signatures;
- rejection of cross-panel disagreement and gripper overlap;
- tactile support without tactile metric leakage; and
- no confidence increase from duplicating an identical camera count.

These are software and information-boundary tests, not real-data evidence.

