# Native Wrapping Belief Source Screen

This is a new, bounded public-simulator source screen, frozen before any wrapping
rollout. It does not change the successful DEFORM updater, previous predictions,
or any closed Slingshot/wiring experiment. It uses no new recordings, real robot,
GPU, held-v8, DLO4/DLO5, official DLO3 evaluation, PokeFlex continuation, or fresh
or reserved Deform360 data. Work remains local and private.

## Question and Preserved Task

Can an uncertain material belief choose meaningfully better motions on the
native three-post loop-wrapping task than a fixed action or a MAP material
estimate? Earlier source tasks offered little decision headroom. The present
question uses a different public task, not retuned versions of their failures.

The authoritative protocol is `dlolab_wrapping_source.protocol()`. Before the
first native initialization, the write-once lock binds its full content, a clean
Git revision, every implementation/test/verifier file, runtime, upstream source,
and assets. Each batch consumes its own write-once claim before initialization.

DLO-Lab is pinned at `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`, Mushroom-RL at
`ec3364740627da945b8bab6e01d8151edb0f83f1`, and the public asset archive at
SHA-256 `acd483e232f1bb1fbf34078b154825fab3d2ee63b0aa4efc253c4411b368e421`.
The unchanged `Train_Env_Wrapping` supplies the 50-vertex extensible closed loop,
two robots, Pink IK controllers, grasped identities 17/33, floor, three posts,
contact, timestep, solver, reward, and failure rules. Its tool-height floor stays
40 mm. Only the existing bending/stretching randomization hooks are overridden
with fixed material values. Native twisting stiffness remains zero. These are
simulator settings, not identified physical material constants.

For each post, let w be the signed winding number of the closed xy polygon and
d its minimum 3D vertex distance. The unchanged final reward is
`1 - mean((abs(w) - 1)^2) - sum(max(d - 0.015, 0))`.
It may be negative. Native failure penalties near -100 do not count as ordinary
predictions. The native cumulative float32 reward, including its +1 per alive
micro-control offset, is also reconstructed exactly, but is not the decision
objective. No reward, contact, or native termination rule is redesigned.

## Fixed Motions and Worlds

Eight unique geometry-designed motions plus a duplicate occupy nine native
environments. Each has eleven macros, 200 simulator steps per macro and ten
micro-controls per macro: 2200 native steps total. The first three macros are
identical and the branch occurs after step 600. Rotation commands are zero;
no gripper translation exceeds 0.1 m per macro.

The common prefix raises the tools, separates them, and starts a pull toward
the posts. The nominal continuation pulls both tools, lowers them, narrows
their separation, and holds. Other actions hold after the prefix, lower early,
lower late, finish wider, finish narrower, or let either gripper lead. All
relative waypoints are fixed in `action_bank()` from public scene geometry.
Slot 8 exactly duplicates nominal slot 1 and is not an extra candidate.
These are not released winning controller trajectories.

Nine equally weighted source settings form the Cartesian product
`K={20000,100000,500000}` and `E={1000,10000,100000}`. Three fresh processes first
repeat the nominal K=100000/E=10000 batch. Only after rederived repeat checks
pass do the other eight worlds each run once. The denominator is eleven batches
and 99 native trajectories. Decision analysis uses the first nominal repetition,
never the best or average repetition.

## Qualification and Stop Rules

Each native trajectory/memory bundle is sealed before qualification. All fields
must be finite, complete, and bound to the specified materials and actions.
Required checks are:

- final reward reconstruction within 1e-7 and float32 cumulative reward equality;
- all native slots ordinary, with no native failure penalties;
- fixed-post coordinates within 1e-9 m and common-prefix coordinates within 1e-5 m;
- duplicate and repeated position differences at most 1 mm and reward differences
  at most 0.001;
- closed-segment lengths between 0.25 and 3 times their initial lengths;
- rod centers no lower than -10 mm and grasp attachment error at most 10 mm.

The segment check is a broad extensible-rod sanity check, not an inextensibility
claim. These numerical budgets are engineering checks, not population bounds.
A failed check, failed repeat, missing seal, or runtime failure stops the screen.
There is no retry, replacement, relaxed tolerance, or alternate-action rescue.
Accounting separates completed native trajectories, qualified trajectories,
technical failures, and never-run batches. A sealed failed trajectory is not a
successful method prediction. A partial screen cannot pass a value gate.

## Belief and Comparators

After all worlds pass and are sealed, inference receives only nominal-action
positions of identities 0,8,25,41,49 at zero-based prefix frames 199,399,599.
The adapter refuses a full-horizon input. Each observation adds independent
2 mm Gaussian coordinate noise and one shared 5 mm translation per entire
prefix. These are assumed synthetic sensor errors, not calibrated real sensing.

The uniform nine-particle posterior accounts for that shared bias. Its action
maximizes posterior expected native final reward. Controls are the best fixed
action over the finite source prior, the nominal world's best action, MAP world
then best action, and a posterior incorrectly treating the shared bias as absent.
The per-world oracle is an upper-bound headroom control. Ties use lowest index.
There are 8192 common random noise draws per world, seed 260930. Reported standard
errors integrate only assumed observation noise; they are not confidence
intervals across independent physical executions or unknown materials.

Every source gate must pass:

1. Best fixed action beats prefix hold by at least 0.05 reward.
2. Oracle gain over best fixed, minus the 0.002 pairwise numerical margin, is
   at least 0.05.
3. At least two actions are oracle-optimal across worlds, and at least three
   worlds have oracle gain above 0.05 over the best fixed action.
4. Bayesian gain over best fixed, minus 0.002, is at least 0.02 and at least
   5% of the fixed action's deficit from reward 1.
5. Bayesian gain over MAP, minus 0.002, is at least 0.002.
6. Bayesian gain over ignored-bias inference, minus 0.002, is nonnegative.

The machine protocol splits these into eight Boolean checks. There is no
secondary endpoint that can rescue a failed primary decision. The alternate
verifier recomputes winding by an angular-unwrapping implementation and posterior
likelihood using Sherman-Morrison precision instead of Cholesky whitening.
It is a second arithmetic implementation, not an independent human review.

## Claim Boundary

Even a pass is finite-prior source feasibility, not independent generalization,
real counterfactual evidence, calibrated real perception, or point-metric SOTA.
The material grid is both the model support and the evaluation support. No
published controller/checkpoint comparator is available for this exact screen.
Bayesian decision theory is not itself the novel contribution. A pass would
justify designing a later, independently locked control evaluation; it does not
authorize one automatically.

Fallback returns the unchanged cached fixed-action result, not a promise of
bit-identical reruns of the native simulator. Existing successful DEFORM results
and all earlier negative evidence remain byte-identical regardless of outcome.
