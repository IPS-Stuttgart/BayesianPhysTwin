# Fresh Deform360 Pairwise-Belief Evaluation

## Purpose

This protocol tests whether the frozen open-development online-belief method
transfers to 12 previously unopened physical objects. The candidate combines:

1. the exact 384-node PhysTwin action-support backbone used by the open-27
   development study;
2. causal AllTracker observations from RGB prefixes ending at frames 19, 38,
   and 57;
3. current-observation selection between the physical and persistence
   backbones; and
4. the frozen pairwise-consensus gate and recursive full-blend RBF correction.

The method is frozen in
`configs/sota/deform360_fresh_pairwise_belief_v1.json`. The cohort is frozen in
`results/sota/deform360_fresh_source_lock_v1/deform360_fresh_object_cohort_lock_v1.json`.

## Causal Boundary

The physical backbone reads frame-zero object geometry and the known robot
action. The action staging archive has 81 frames; the frozen prediction uses
frames `[0,76)` and skips the five-frame tracking tail. At update frame `u`,
AllTracker reads exactly RGB frames `[0,u]`.
The eligible camera panel is the lexically sorted intersection of calibrated
cameras with materialized RGB, frame-zero mask, and frame-zero depth files;
the frozen selector still chooses exactly eight cameras from that panel.
Selection and correspondence gating use only the current sparse observation,
the two sealed backbones, and prior belief state.

No prediction process accepts a future target, outcome manifest, future dense
geometry, future tactile data, or post-update RGB. All 12 belief predictions
must be checksummed and pass the completeness barrier before a separate
operator may materialize any outcome.

An automatic-twin source-admission rejection has one preregistered behavior:
the physical arm becomes exact persistence. Any runtime exception, malformed
artifact, missing case, or unauthorized input is a technical failure and
blocks the barrier. Cases are never replaced.

## Calibration Boundary

The pairwise and RBF thresholds are exactly those selected on the open
development cohort. They are not recalibrated on the fresh objects. The
AllTracker triangulation diagnostics are observation-quality records, not
calibrated posterior variances.

This study therefore supports transfer claims about point prediction only. It
does not establish calibrated coverage or NEES. Any later uncertainty claim
requires an independently frozen calibration protocol.

## Claim Boundary

The preregistered primary metrics are hidden-identity RMSE and hidden symmetric
Chamfer, evaluated with the candidate conventions bound by the parity contract.
The result may be described as fresh-object transfer under explicit candidate
metric conventions. It is not an official Deform360 3-D state-of-the-art claim
until evaluator parity is independently resolved.

## Execution Order

```text
source admission and cohort lock
-> frame-zero physical backbones
-> causal RGB-prefix measurements
-> target-free pairwise-belief predictions
-> 12-of-12 completeness barrier
-> one outcome-opening operator
-> scoring and object-cluster uncertainty
```
