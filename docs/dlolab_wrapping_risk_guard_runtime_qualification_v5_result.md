# DLO-Lab wrapping runtime qualification v5 result

## Decision

The one registered Python 3.11 qualification attempt is a terminal technical
failure. It produced zero ordinary constructor seals and zero full-rollout
seals. No retry, replacement, partial score, or scientific study is authorized
by this result.

## What happened

The first fresh worker entered the public DLO-Lab constructor, printed the
initial rope length, returned from `RegisteredWorld(...)`, and completed
`init_cmaes_env`. It then raised the qualification harness error
`native material realization was not captured`.

That check was misplaced. The registered bending/stretching callbacks are
invoked by the reset inside `eval_traj`, not by construction or
`init_cmaes_env`. The v5 constructor probe intentionally stopped before
`eval_traj`, so an empty capture at that point is expected. The same misplaced
check also sat before `eval_traj` in the full-rollout path, making v5 incapable
of reaching its intended native rollout.

This attempt therefore does not qualify Python 3.11, but it also does not
reproduce v4's CPython `subtype_dealloc` crash: the process exited normally with
Python return code 1 after the explicit harness exception.

## Boundary

The result consumed the sole v5 attempt. It read no v4 partial future arrays,
defined no fresh scientific worlds, exposed no scientific outcome, and touched
no protected data. A successor must be a separately frozen qualification. It
may treat material randomization as deferred in constructor-only probes, but it
must verify the exact world realization after reset in every complete rollout.
