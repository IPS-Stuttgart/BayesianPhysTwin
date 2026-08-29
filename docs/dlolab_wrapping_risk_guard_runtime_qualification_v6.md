# DLO-Lab wrapping runtime qualification v6

## Purpose

V5 terminated cleanly on its first constructor probe because the harness
required bending/stretching randomization before the public environment had
entered `eval_traj`. Source inspection and the traceback establish that the
constructor and `init_cmaes_env` had completed; DLO-Lab invokes the registered
material callbacks during the later reset inside `eval_traj`.

V6 is a separate, one-attempt source-only qualification. It retains the same
pinned Python 3.11 runtime and public DLO-Lab workload, but corrects the
instrumentation boundary:

- constructor-only probes require material capture to remain empty and label
  randomization as deferred;
- complete rollouts require both callbacks to run and verify the exact nine
  registered stiffness values after reset.

It does not retry v5, read v5 runtime artifacts, or resume v4.

## Frozen workload and gate

The campaign uses only nine copies of the already-open preflight world in every
fresh process. It runs 24 constructor probes followed by four complete
2,200-microstep rollouts. Each constructor probe serializes finite initial and
native-memory state after `init_cmaes_env` without stepping or exposing reward.
Each full rollout is checked by the unchanged v4 native QA plus the exact
post-reset material realization.

The source gate requires 24/24 and 4/4 ordinary successes. Any exception,
signal exit, missing seal, QA failure, unexpected early randomization, or
post-reset material mismatch terminates the campaign. Retry, replacement,
padding, and partial-denominator qualification are forbidden.

## Evidence boundary

The runner may read committed compact terminal summaries for v4 and v5. It may
not read either campaign's runtime arrays, logs, or partial rewards. It defines
no fresh scientific worlds and produces no accuracy, uncertainty, safety, or
SOTA claim. A pass permits only a separately frozen fresh-world study.
