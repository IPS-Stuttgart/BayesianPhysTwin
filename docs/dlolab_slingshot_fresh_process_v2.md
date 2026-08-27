# Native Slingshot: Fresh-Process Qualification v2

The reused-environment v1.1 gate failed on tiny final velocity/twist differences.
That failure remains immutable. This new source qualification changes only the
execution/reset contract: each of the same zero/pull/pull trajectories starts
in a fresh Python interpreter, with the same pinned native code, official
assets, CPU runtime, seed, robot, controller, physics, reward, and actions.
The 1 micrometre position threshold and full-memory `rtol=1e-6, atol=1e-9`
remain unchanged. There is no post-outcome tolerance relaxation.

Each child consumes a write-once claim before native initialization, executes
exactly one 900-step trajectory, and seals the trace, all 23 rigid/rod state
fields, commands, and realized joint targets. The parent only compares sealed,
rehashed arrays after all three children complete. No child or failed run is
retried. The parent binds the exact failed v1.1 result and identical native
source/runtime before creating a new output directory.

This establishes an execution contract, not controller competence or a Bayesian
result. The zero/pull controls are not a published-method reproduction. A later
task-level experiment requires a separate source-frozen design with matched
controls and an actual decision-value check. No parameter search or method
evaluation is authorized here. This is CPU-only, public simulator-only work;
there are no new recordings, protected outcomes, GPU jobs, push, or main merge.
