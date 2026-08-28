# Native Wiring-Post Belief Source Screen

This separate public-simulator experiment is frozen before any wiring rollout.
It does not alter DEFORM, any successful prediction, or any closed source study.
It uses no new recordings, real robot, GPU, held-v8, DLO4/DLO5, official DLO3
evaluation, or reserved Deform360 data. Work remains local and private.

## Question

Does the native DLO-Lab wiring-post task offer enough material-dependent action
value to justify a later uncertainty-aware control study? Slingshot's largely
dominant actions supplied little useful decision headroom. Changing tasks is a
new source screen, not permission to retune the failed Slingshot protocols.

The authoritative machine protocol is `dlolab_wiring_source.protocol()`. The
write-once run lock contains its complete canonical content, the clean local
Git revision, source-module hashes, runtime versions, and public asset hashes.

## Preserved Task

DLO-Lab is pinned at `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796` and its existing
asset archive at SHA-256
`acd483e232f1bb1fbf34078b154825fab3d2ee63b0aa4efc253c4411b368e421`.
The native `Train_Env_Wiring_post`, Pink IK controller, robot, rod, posts, floor,
time step, constraints, contact, grasp, and public target shape are unchanged.
Only the existing native bending/twisting randomization hooks receive specified
values. Inextensibility remains enabled; a stretching parameter is not varied.
The material settings are simulator parameters, not recovered real materials.

The native final reward is exp(-C), where C is the sum of both directed mean
nearest-point distances, mean height above 40 mm, and five times the excess
distance of material point 15 from (0.198, 0.198) beyond 20 mm. We reproduce this
formula from the sealed native trajectory. Native float32 cumulative reward is
also checked but is not the primary decision objective. No reward is redesigned.

## Fixed Actions and Worlds

Seven unique, geometry-designed Cartesian motion sequences plus an exact
duplicate use nine native macros, 200 simulator steps per macro, and ten Pink
micro-controls per macro. No macro translates more than 0.1 m, and orientation
commands remain zero. The first three macros are identical; branch time is 600
steps. The native minimum tool height remains 30 mm.

Actions are prefix hold, nominal routing, four nominal endpoint offsets of
50 mm along either signed x/y direction, and a 25 mm x/y overshoot-and-return.
The nominal path and offsets are fully specified by `action_bank()`. They use
public scene/goal geometry, not a learned or published winning controller.
Slot 7 duplicates nominal slot 1 and is never an additional policy candidate.

Nine equally weighted source worlds form the Cartesian product
E={1000,10000,100000}, G={100,1000,10000}. All other properties are identical.
The first three fresh-process batches repeat the native nominal E/G world.
If they pass, the remaining eight worlds each run once. This is eleven batches,
88 native trajectories, and nine unique material settings. Nominal decision
analysis uses the first repetition, never the best or the average repetition.

## Qualification Before Value Analysis

Every native result is sealed before arithmetic checks. All observables and
full rod/rigid memory must be finite. The checks require unchanged fixed posts
within 1e-9 m, common-prefix coordinates within 1e-5 m, maximum segment-length
error <=10%, rod centers above -10 mm, and grasp distance <=10 mm. Duplicate and
three-repeat coordinates must agree within 1 mm and final rewards within 0.001.
These are operational source budgets, not empirical population coverage bounds.
Each task claim is written before native initialization. Failed qualification,
runtime failure, missing seals, or failed repetition is terminal, with no retry
or replacement. The material screen cannot bypass the rederived repeat gate.

Only after all worlds are sealed does the fixed belief calculation use material
positions 6,12,18,24,29 at zero-based frames 199,399,599 from the nominal action's
shared prefix. It cannot accept future frames. Observations add independent
2 mm Gaussian coordinate noise and one shared 5 mm translation bias per noisy
prefix. These are assumed synthetic sensors, not calibrated real perception.

## Comparators and Gates

An equally weighted nine-particle posterior uses that known correlated noise
model. The action maximizes posterior expected native final reward. Comparators
are the best fixed action across the complete source prior, the nominal world's
best action, MAP material then its best action, and a posterior wrongly ignoring
shared bias. Perfect-information action choice is a headroom ceiling. All ties
use the lowest index. A fixed 8192 common-random-number noise draws per world,
seed 260928, integrate the assumed noise; their standard error is Monte Carlo
integration error, not an interval across independent real executions.

All gates must pass. The best fixed action must exceed prefix hold by 0.01.
Perfect-information gain minus a 0.002 numerical pair margin must exceed 0.01,
with at least two distinct best actions and at least three worlds gaining over
0.01 relative to the best fixed action. Bayesian gain after the same margin must
exceed 0.005 and 5% of the best fixed action's deficit from reward 1. Its adjusted
gain over MAP must exceed 0.002 and over ignored-bias inference must be nonnegative.
No gate is changed after viewing the results.

## Claim Boundary

Even a pass is source feasibility under an exactly specified simulator prior,
not independent generalization, calibrated physical parameters, real sensor
performance, real counterfactual ground truth, or SOTA. The material support is
also the finite evaluation grid. No published wiring controller is supplied in
the pinned assets, so the best-fixed comparator is not an official leaderboard.
Bayesian decision theory itself is not novel. This screen asks whether a more
substantial later contribution has practical headroom under strong controls.

A fail keeps the existing DEFORM results untouched. Fallback means returning the
cached fixed-action forecast/decision, not asserting bitwise native replays.
Results, numerical failures, and unrun denominator entries are all preserved.
