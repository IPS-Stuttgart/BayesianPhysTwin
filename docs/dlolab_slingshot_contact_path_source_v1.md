# Native Slingshot: Causal Contact-Path Recovery Screen

The force-only source bank had a small Bayesian gain but insufficient oracle
headroom to pass the unchanged advancement criteria. This new source screen
tests a different control variable: the gripper's vertical path during loading.
It does not retune the failed observation model, inference rule, or gate.

Use the same three native coupling worlds 0.3/0.6/0.9, nominal placement and
material, in order 0.9/0.3/0.6. Every run is a fresh eight-environment CPU
process with the same 900-step horizon and no retry or replacement.

| Slot | Cartesian source | Vertical detour in second macro (m) | Force per finger after step 300 (N) |
|---:|---:|---:|---:|
| 0 | Prior contact action 6 | -0.02 | -24 |
| 1 | Prior contact action 6 | -0.01 | -24 |
| 2 | Prior contact action 6 | +0.01 | -24 |
| 3 | Prior contact action 6 | +0.02 | -24 |
| 4 | Prior contact action 6 | 0 | -24 |
| 5, fallback | Prior contact action 6 | 0 | -3 |
| 6 | Prior contact action 5 | 0 | -24 |
| 7, duplicate fallback | Prior contact action 6 | 0 | -3 |

For slots 0-3, add the declared z detour to macro 2 and subtract it from
macro 3. If either translation exceeds the native 0.1 m norm limit, scale
only its xy components to that limit while retaining z. This is a registered
path change, not a claim that final Cartesian endpoints remain identical.
Rotation commands, the entire first macro, native arm controller, force
limits, and release at step 700 stay unchanged. No position teleport or
hidden-state restart is used. Dips and lifts are both tested because the
source prefix's nearest rod vertex does not identify the contact-patch
geometry well enough to assume a useful sign.

Slots 4/5/6/7 must replay the frozen force screen's slots 2/5/6/7 within
1 micrometre with exact native reward. This retains both the previous best
fixed action and the action choices used by its Bayesian policy. Both
fallbacks also replay the older contact action 6. All entire prefixes must
replay within 1 micrometre. The original native QA, actual force-command
checks, and release checks all remain required. A failed batch stops the
screen; every attempted or unrun registered world is retained in accounting.

The observation remains twelve causal 3D positions at frames 139/219/299,
with 2 mm independent noise plus 5 mm shared xyz bias. The prior is equal
over the same three worlds. Integrate 8192 draws per world with seed 260909.
Use the unchanged posterior-mean, MAP, ignored-bias, strongest fixed action,
and perfect-information comparisons. No actual camera or tactile data is
claimed. The native coupling parameter is not measured Coulomb friction.

Advancement requires every previous criterion: posterior gain at least
0.005 above the new best fixed action, at least 10% of its excess reward
above zero control, at least 0.002 above MAP, and no loss to ignored-bias
inference. Also require at least 0.002 gain over the frozen force-only
posterior policy. Monte Carlo errors are integration errors, not independent
experimental intervals. A passing source calculation would still not
constitute an independent control evaluation or calibrated safety claim.

This adaptive next design is explicitly informed by earlier opened source
results. It cannot rescue or overwrite those failed gates. Its write-once
root is `/home/fpfaff/source-only/dlolab-benchmark-source-v1/contact-path-source-v1`.
The lock binds the clean implementation, source files, exact commands,
previous contact/grip locks, previous grip result, and reference seal IDs.
Workers rederive earlier native gates before running the next world.

No new recording, GPU, robot, protected data, calibration/evaluation world,
held-v8, DLO4/DLO5, public push, or main merge is authorized. Existing DEFORM
results remain unchanged.
