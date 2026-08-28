# Native Slingshot Contact-Realization Source Screen

This source-only experiment asks whether uncertain contact coupling, rather than
the poorly observed elastic parameters, creates identifiable action value. It
does not reopen or revise the failed 32-world belief-control study, the two
early-pull probes, or any DEFORM result. No recording, robot, GPU, protected
target, calibration world, or evaluation-world payload is used. No result here
is independent control confirmation or benchmark parity.

## Mechanism and Fixed Budget

Run exactly three fresh eight-environment CPU batches with Franka native
`coup_friction` equal to 0.3, 0.6, and the original 0.9. The order is 0.9, 0.3,
0.6. Every batch keeps the existing seven action sequences and incumbent
duplicate, 900 steps, nominal E=100000, K=800000, placement x=0, native reward,
controller, release, and nonrobot contact materials unchanged. No action is
optimized on these results. The three-point equal prior is an explicit source
model, not a measured distribution of physical gripper properties.

In this pinned DLO-Lab implementation, gripper contact makes rod vertices
kinematic and scales tangential relative velocity by
`1 - influence * coup_friction`. It is not an ordinary measured Coulomb
coefficient or a validated slip sensor. The adapter changes only the unique
Franka material constructor and verifies every solver geometry coefficient
before entering the native action. It does not modify upstream files.

The 0.9 batch must reproduce the already-open nominal source trace within
1 micrometre and have exactly the same native cumulative rewards before the
other two batches can run. All batches require the existing common-prefix,
duplicate-action, fixed-endpoint, and native-reward checks. A technical or QA
failure is terminal, not replaced or retried. A failed identity control is not
evidence about contact-uncertainty value.

The older source trace duplicated action 0 in its eighth slot, whereas the
current unchanged bank duplicates action 5. The reference is deterministically
indexed by `[0,1,2,3,4,5,6,5]`; all seven unique actions remain byte-identical.
This mapping and the original trace seal are bound before execution.

## Source Information and Decision Value

Use only rod nodes 3/6/8 and the sphere center at prefix frames 139/219/299 to
form the likelihood. Candidate actions share that prefix and branch after
frame 299. The same observation model is retained: 2 mm independent xyz noise
and one 5 mm episode-shared xyz bias. Native contact flags, forces, future
positions, and robot state are not observations of the policy.

The complete source reward table is permitted for an ideal finite-model value
calculation. Integrate 8192 noise draws per source world, seed 260909, using
common draws for numerical comparisons. Report bias-aware posterior mean,
bias-aware MAP, an ignored-shared-bias ablation, the best fixed action, and
perfect information. Report all seven fixed-action values and all three
worlds. Monte Carlo standard errors measure numerical integration error only,
not experimental confidence or transfer uncertainty.

The source screen passes only if all native/identity checks pass, posterior
gain over the strongest fixed action is at least 0.005 and at least 10% of its
excess above the zero-action reward, posterior gain over MAP is at least
0.002, and it does not lose to the ignored-bias ablation. These are inherited
engineering advancement thresholds, not calibrated significance tests. A pass
would justify designing a new continuous-contact source study; it does not
authorize an evaluation automatically. A failure closes this exact finite
contact model and action/observation budget, not contact inference generally.

## Custody

The write-once root is
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/contact-realization-source-v1`.
The lock binds clean local Git HEAD, source/test/doc bytes, exact source-bank
and nominal-reference identities, qualified public assets and CPU runtime,
and unchanged command bytes. Each worker consumes a task claim before native
initialization. Later workers rederive the nominal identity check; a stored
authorization boolean alone is insufficient. No alternate root, retry,
automatic new study, public push, or main-branch promotion is permitted.
