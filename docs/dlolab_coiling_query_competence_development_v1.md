# DLO-Lab Coiling Query-Competence Development Screen

This is a bounded public-simulator development screen. It is not paper evidence,
an official DLO-Lab score, a target result, or permission to alter the positive
wrapping certificate or the negative Slingshot certificate. It uses no new
recordings, protected data, held-v8, DLO4/DLO5, or official DLO3 evaluation.

## Question

Does a third untouched DLO-Lab task contain enough material-dependent decision
headroom for a prospective query-conditional simulator-competence certificate?
Coiling is tested first because its public task has one rope, one gripper, fixed
cone geometry, and an unchanged native final reward. If the frozen development
gate fails, coiling closes without tuning on these worlds. Gathering or
separation would require a separately locked study.

## Frozen Native Task

DLO-Lab is pinned at revision
`c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`. The native
`Train_Env_Coiling`, Genesis rod/rigid solvers, Pink controller, 60-node rope,
cone, floor, robot, contact, grasp, 1 ms step, five substeps, and public reward
are unchanged. Only the existing bending and twisting material hooks receive
specified values. These are simulator settings, not real material estimates.

The native final reward is

```text
exp(-0.1 * sum_i ||x_i - [0, 0, 0.15]||_2).
```

The study rederives this value from each sealed final geometry.

## Fixed Query and Actions

Eight native slots contain seven unique translation-only actions and an exact
duplicate. All share four 200-step macros before branching at native step 800.
The common prefix moves the grasped endpoint from the public initial rope pose
toward the cone. Continuations are prefix hold, four clockwise arcs with fixed
radius/speed variants, one counterclockwise arc, and one radial-in path. Every
macro translation is below 0.1 m; rotations are zero. The paths are computed
only from the public initial rope and cone geometry. Slot 7 duplicates the
medium clockwise action and is not an extra policy.

Nine equally weighted development worlds are the Cartesian product
`E,G in {500, 2000, 8000}`. Every world runs once in a fresh process. This is a
finite simulator prior, not a population sample. All worlds must seal and pass
native qualification before any value analysis.

The observation is the shared-prefix position of rope nodes 5, 15, 30, 45, and
58 at zero-based native frames 399, 599, and 799. Synthetic observations add
independent 2 mm Gaussian noise and one common 5 mm translation bias. The
posterior uses that correlated likelihood. These values are assumptions, not
calibrated sensor noise.

## Qualification and Development Gate

The runner requires finite native observables and complete final solver memory,
reconstructed native reward, common-prefix agreement within 10 micrometres,
duplicate agreement within 1 mm and 0.001 reward, segment-length error at most
10%, rod height above -10 mm, grasp distance at most 20 mm, and fixed cone pose
within 1 nm. A claim is written before each world initializes. Failure is
terminal; no retry or replacement is permitted.

The complete sealed bank compares posterior Bayes action choice with prefix
hold, the best fixed action, MAP-material choice, and perfect-information action
choice. With a 0.002 numerical pair margin, all frozen checks must pass:

- best fixed exceeds prefix hold by at least 0.01;
- adjusted perfect-information headroom is at least 0.005;
- at least two actions are oracle-optimal;
- at least three worlds gain 0.005 over the best fixed action;
- adjusted Bayes gain over best fixed is at least 0.002;
- Bayes captures at least 25% of oracle headroom;
- Bayes is not worse than MAP-material choice.

An 8192-draw common-random-number calculation with seed 270901 integrates only
the declared synthetic observation model. A pass establishes development
feasibility, not independent evidence. It does not automatically authorize a
replication. A new immutable protocol must first select fresh material and
initial-offset worlds, preserve the same action/query/gates, and retain every
failure in its denominator.

## Claim Boundary

The prospective contribution, if later supported, is an exact query-scoped
competence certificate: the same simulator may be admitted for one manipulation
query and rejected with exact fallback for another. It is not a claim that
DLO-Lab, DEFORM, or any backend is globally competent. Existing positive and
negative results remain byte-separated and unchanged.
