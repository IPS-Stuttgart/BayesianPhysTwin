# Public Native Slingshot: Source Qualification

The two earlier procedural action-choice experiments are frozen negative
results. This separate study uses DLO-Lab's published slingshot task, not a
retuned goal, reward, or contact scene. It tests whether the exact benchmark
environment can support a subsequent belief/control comparison.

Pin the official DLO-Lab source at
`c5026a9416b03c6bc5186eba13cd4ffd4c0e7796` and its documented Mushroom-RL fork
at `ec3364740627da945b8bab6e01d8151edb0f83f1`. The official asset ZIP is
149168403 bytes with SHA-256
`acd483e232f1bb1fbf34078b154825fab3d2ee63b0aa4efc253c4411b368e421`.
Its installed members and the native robot assets are hash-bound before use.
The exact source, dependencies, and one-attempt output root are recorded before
initialization. All work is local/private and CPU-only, with no new recordings.

Use the unchanged `Train_Env_Slingshot`, `RobotControllerPink`, `eval_traj`,
reset, native ROD/rigid coupling, and reward. Initialize Genesis explicitly
on CPU/float64 before constructing the environment; use one environment,
headless OSMesa, no cameras, no gradients, and one Torch/BLAS thread.
This runtime choice is not published GPU performance parity.

The native task has a twelve-vertex elastic band, a ball, a target cube, and
a simulated Franka arm. The supplied pregrasp pose remains unchanged. Each
rollout uses the published three-stage, six-coordinate controller interface,
ten controller substeps per stage, 100 settling steps, 600 action steps, and
200 release steps. The fixed source actions are zero motion and a 40 mm pull
along negative y in each stage; all rotations are zero. Run zero, pull, and
the same pull again, in that order. Every call goes through native `eval_traj`.

A read-only step wrapper records native object positions/velocities, gripper
position, and robot configuration after the original `scene.step` returns.
It does not replace or alter the solver, controller, time step, or reward.
Record all 15 native rod and eight rigid state fields at each rollout end.

Before any method comparison require all three 900-step rollouts to finish,
finite arrays, at least 10 mm gripper and band-midpoint motion during the pull,
fixed band endpoint error <=1e-9 m, repeated position error <=1e-6 m, and
repeated memory within rtol=1e-6/atol=1e-9. Also report byte identity, without
pretending tolerance-based agreement is byte identity. This does not yet
qualify arbitrary mid-episode snapshots or prove physical convergence.

The native cumulative reward is reported only as source setup information.
No optimizer, posterior, new controller, or uncertainty claim is evaluated.
A failed qualification is retained without retry or threshold relaxation.
Passing does not automatically authorize a method experiment. That later
experiment must freeze partial observations, world uncertainty, model/baseline
budgets, native task reward, calibration draws, test draws, and success gates.
It must include nominal, MAP, posterior-mean, calibrated mean-only, and
joint-regret controls with matched information and computation.

Primary references: the [official code](https://github.com/UMass-Embodied-AGI/DLO-Lab)
and [DLO-Lab paper](https://arxiv.org/abs/2606.04206). Bayesian control or use of
this simulator is not itself claimed as new. No DEFORM DLO4/DLO5, official
DLO3 evaluation, held-v8, reserved Deform360 targets, physical Causal4D data,
GPU jobs, main-branch changes, or public pushes are involved.
