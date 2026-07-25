# Deform360 v8.3 Viser Process-Churn Guard

Status: source-only rehearsal passed; formal GPU-1 qualification pending

## Incident

The first full 81-fit source-only child rehearsal for held-v8.3 used code
revision `f3bd24904e3c66bf0f1beb1aba3fbde51f72811c` and the original pinned
Deform360 trainer. It failed before producing a valid lifecycle result when
Viser's private `_check_viser_yarn_running()` enumerated a live process and
that process exited before `Process.cwd()` was read. `psutil` raised
`NoSuchProcess`, while the pinned Viser helper catches only `AccessDenied` and
`ZombieProcess`.

The failure is a host-process enumeration race. It occurs before numerical
training for that fit and is independent of the object state, target outcome,
or reconstruction score. The unguarded rehearsal is preserved at:

```text
/mnt/corsair/florianpfaff/
bpt-v83-gpu0-lifecycle-smoke-f3bd24904e3c66bf0f1beb1aba3fbde51f72811c
```

The pinned upstream Viser source has SHA-256:

```text
3c657a8baa49498e372234f05e4d9baf4c8a45b1aead45276e881987bf9da506
```

## Narrow correction

Held-v8.3 workers now install a byte-bound guard before importing the original
trainer. The guard reproduces the pinned helper's yarn-detection rule and adds
only `psutil.NoSuchProcess` to the two exceptions already ignored upstream.
It does not alter:

- the original Deform360 trainer;
- trainer configuration or iteration count;
- PhysTwin geometry, state, parameters, or likelihoods;
- cohort membership, gates, or statistical thresholds;
- target-query, outcome, or scoring capabilities.

The guard records its own source, the exact upstream Viser source, the ignored
exception set, installation order, and a signed artifact hash. The one-case
worker result, process-isolation qualification, integrity completion, and
held-v8.3 lock all require this provenance. Qualification identifiers advance
from v2 to v3; a prior qualification cannot silently authorize the new worker.

## Source-only rehearsal

Four fresh guarded child processes were exercised on physical GPU 0 outside
the canonical formal qualification root. Each child completed all 81 original
trainer fits. Across the four children:

- every signed child, guard, and gsplat artifact validated;
- process identifiers were distinct;
- all children started at 48 file descriptors and 165 tasks;
- all peaked at 372 file descriptors;
- maximum task growth was 82;
- the exact frozen gsplat extension SHA-256 was
  `2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64`;
- no formal held path, target query, outcome, gate, or score path was supplied
  or accessed.

The guarded rehearsal roots are:

```text
/mnt/corsair/florianpfaff/bpt-v83-gpu0-guarded-child-smoke-dev1
/mnt/corsair/florianpfaff/bpt-v83-gpu0-guarded-multichild-smoke-dev1
```

These runs establish source-only engineering readiness. They are not the
registered process-isolation qualification, a calibration result, a target
result, or evidence of state-of-the-art accuracy.

## Formal gate

The formal qualification remains pinned to physical GPU 1 on `workstation2`
and a canonical revision-specific root. It must run four fresh 81-fit children
and pass the existing resource, source-integrity, process-boundary, and
information-boundary predicates. It must then be independently sealed before
a fresh held-v8.3 lock can be created.

No held-v8.3 target artifact may be opened on the strength of the GPU-0
rehearsal.
