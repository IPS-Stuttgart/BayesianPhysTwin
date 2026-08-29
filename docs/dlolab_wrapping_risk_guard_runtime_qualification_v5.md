# DLO-Lab wrapping runtime qualification v5

## Purpose

The frozen v4 chance-guard study terminated after 69 of 72 future worlds when
CPython 3.12.13 segfaulted in `subtype_dealloc` while constructing the next
native `RegisteredWorld`. No complete-denominator score exists and v4 remains a
terminal technical failure. This qualification asks a narrower source-only
question: can the unchanged public DLO-Lab wrapping workload execute repeatedly
under a pinned Python 3.11 runtime without reproducing the constructor failure?

This is not a retry of v4. It does not read the 69 partial future bundles,
define fresh scientific worlds, evaluate a guard, or produce a task-value
result. A passing qualification permits only a separately committed and frozen
v5 study design.

## Frozen workload

The campaign uses only the already-open v4 preflight world:

```text
stretching_K = 100000.0
bending_E    = 10000.0
```

Every native worker is a fresh Python process. The registered order is:

1. 24 constructor probes, each with nine copies of the preflight world. A probe
   executes the same `RegisteredWorld` constructor and `init_cmaes_env` path as
   a scientific rollout, serializes the finite initial/native-memory state, and
   exits without stepping or exposing reward.
2. Four complete 2,200-microstep rollouts of the same preflight world and the
   unchanged nine-action bank. The existing v4 native QA is recomputed from the
   sealed arrays and rewards.

The source gate passes only with 24/24 ordinary constructor successes and 4/4
ordinary full-rollout successes. Any exception, signal exit, missing seal, QA
failure, or custody mismatch terminates the campaign. There is no retry,
replacement, padding, or partial-denominator result.

## Runtime boundary

The exact runtime is Python 3.11.15 with CPU-only Torch and software OSMesa.
The interpreter and 131-entry package lock are hash-bound in the plan. The
public DLO-Lab/Genesis source snapshot is unchanged. The v5 native adapter
factors the constructor into one shared implementation used by both probe and
full-rollout paths; it does not alter the v4 action, material, geometry, solver,
or QA contracts.

## Evidence boundary

The campaign may read committed source and the compact v4 terminal summary. It
must not read v4 partial future arrays or rewards, protected data, held-v8,
DLO4/DLO5, PokeFlex, Deform360 targets, or any physical recording. No result from
this qualification is an accuracy, uncertainty, safety, SOTA, or scientific
promotion claim.
