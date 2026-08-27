# Standard CMA-ES Nominal Controller Development

This is a new, bounded source-development optimizer study. The failed fixed
24-action bank stays failed. Its best native-reward action is openly used as
the starting point, so neither this nominal task nor the selected controller
is an independent target confirmation. CMA-ES is an existing baseline, not a
new contribution. The aim is to establish a competent incumbent before asking
whether a Bayesian controller helps.

Freeze `cma==4.4.4`, seed 260829, initial sigma 0.02, population 16, and exactly
four generations (64 native candidate evaluations). Use the official DLO-Lab
`project_deltas` with 100 mm per-stage translation and 1 radian per-component
rotation bounds. The 18-dimensional initial mean is the highest-reward action
in the complete, retained 24-action bank, with lowest-index tie-breaking.
The optimizer receives only the unchanged native cumulative reward, not a
shaped loss. Every generation proposal is sealed before its two eight-world
batches execute; all returned arrays are sealed before `tell` sees the rewards.

Both isolated and eight-environment qualifications must reverify, including
source/runtime identities and full native memory. Each eight-world batch is
executed in a fresh process. No failed batch or candidate is retried. Failures
are retained and terminate this optimizer attempt. The final best controller
(including the warm start if it remains best) receives exactly one isolated
replay, not a new optimization opportunity.

The replay must move both sphere and cube at least 10 mm, beat native zero
reward by at least 0.01, and keep the tracked gripper at least 80 mm from the
cube. Native position traces must match within 1 micrometre; every final rigid
and rod memory field must match at `rtol=1e-6, atol=1e-9`. Native float32 reward
may differ by at most 1e-5 between executions. Within each execution, reward
must reproduce exactly from the saved trajectory. No failed prior threshold
is changed by these new selected-controller replay requirements.

A pass establishes only nominal source-task competence. It is not Bayesian
improvement, a reproduction of the much larger published CMA-ES budget, real
robot performance, or SOTA. A belief-dependent test still needs a separate
frozen source-world/action design and strong matched point controls. No target,
held-v8, DEFORM protected data, GPU work, new recording, public push, or merge.

Reward competence also does not establish an elastic-launch mechanism: the
native task could admit rigid-contact shortcuts. Before interpreting any
controller as a deformation-dependent intervention, a separate source-frozen
contact or rod-removal audit must show that the deformable object is causally
needed. This optimizer does not authorize or substitute for that audit.
