# DLO-Lab wrapping runtime qualification v6 result

## Decision

The corrected Python 3.11 qualification is a terminal technical failure. It
sealed 22 of 24 independent constructor probes, then constructor 22 exited on
`SIGSEGV`. No full rollout started, no scientific outcome was scored, and no
retry or replacement is authorized.

## What the evidence establishes

The first 22 fresh processes each constructed the nine-world environment,
completed `init_cmaes_env`, serialized finite initial/native-memory state, and
exited normally. This validates the corrected deferred-randomization contract.

The next process printed the registered environment configuration but did not
reach the constructor's initial-length print. Its write-once claim was followed
only by a parent-sealed process failure with return code `-11`. WSL recorded a
Python 3.11 fault at instruction pointer `0x1`. Read-only core inspection places
the active Python stack at `env_wrapping.py:162`, the public Genesis
`scene.build(...)` call reached from `RegisteredWorld(...)`.

V4 had already crashed during the same broad construction phase under Python
3.12.13. V6 therefore rules out the narrow explanation that the instability is
specific to CPython 3.12. The evidence instead points to intermittent native
scene-build/lifetime instability in this WSL execution path. It does not identify
the precise upstream native component.

## Consequence

The public DLO-Lab chance-guard lead remains unscored, but the methodological
idea is not falsified: neither v4 nor v6 produced a complete scientific
denominator. A successor must change the execution boundary rather than retry
this campaign. The least invasive next test is the same source-only gate on a
native Linux host, with the simulator, world, action, and qualification logic
unchanged and with its own one-attempt custody.

## Boundary

V6 read no v4 partial futures or v5 runtime artifacts, defined no fresh
scientific worlds, and touched no protected data. The 22 completed constructor
states are runtime evidence only. They may not be scored or used for method
selection.

