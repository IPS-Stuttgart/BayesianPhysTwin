# Deform360 Process-Isolation Qualification

Status: prospective source-only development protocol

## Motivation

Held-v8.1 attempt 4 failed after two completed target reconstructions because
Nerfstudio viewer, writer, and profiler resources accumulated across cases in
one Python process. Attempt 5 replaced the pinned trainer configuration with a
resource-bounded wrapper. Its source-only numerical-equivalence gate was
inconclusive, so that wrapper is not admitted for another held evaluation.

The next implementation keeps the original pinned Deform360 numerical trainer
unchanged and changes only its lifetime:

1. the parent process consumes the per-case reconstruction capability;
2. it launches one fresh child process with reconstruction inputs only;
3. the child runs one complete 81-fit reconstruction lifecycle with the pinned
   default `NerfstudioSplatTrainer`;
4. the child publishes one checksummed result and exits;
5. only the parent can query the hidden frame-zero identities or score a case.

Process exit, rather than an altered trainer configuration, reclaims
process-global resources.

## Frozen source-only lifecycle test

The admission test uses only the public development Splatfacto input at

```text
/mnt/corsair/florianpfaff/deform360-reusable-sota-v1/
processing-sam2-dev-smoke/004-rubber-band/episode_0001/
splatfacto/.scratch_000000
```

It runs on physical GPU 1 of `workstation2` using the already frozen Python and
Deform360 runtimes. A canonical qualification consists of four sequential
child processes. Every child:

- constructs exactly one original pinned `NerfstudioSplatTrainer`;
- performs 81 one-iteration fits, matching one official reconstruction's
  per-frame trainer lifecycle;
- removes each generated PLY and Nerfstudio output tree after validation;
- records file-descriptor, task, memory, and `RLIMIT_NOFILE` counts after every
  fit;
- receives no held root, target query, outcome, gate, or score path.

The four-child design crosses the three-case region in which the in-process
implementation failed while retaining the exact one-case process boundary
proposed for the next protocol.

## Acceptance rule

The source-only lifecycle arm passes only when all of the following hold:

1. all four children complete all 81 fits;
2. every child starts under the frozen soft `RLIMIT_NOFILE` value of 1024;
3. every child retains at least 256 unused file descriptors throughout its
   complete case lifecycle;
4. child task growth is at most 192 tasks over its pre-fit boundary;
5. each child uses the original trainer and no trainer keyword override;
6. the parent gains at most two file descriptors and two tasks after any child
   exits;
7. child process identifiers are distinct and child starting resource counts
   remain within four descriptors and four tasks of one another;
8. all source and materialized input bindings remain unchanged;
9. the code and Deform360 checkouts are clean and exactly bound;
10. no formal held path or target-side artifact is supplied or accessed.

The output root is consumed by its exact code revision. A failed or incomplete
qualification is not retried in place; any correction requires a disclosed new
revision and a fresh root.

## Claim boundary

Passing this test establishes only that per-case process isolation can execute
the original trainer lifecycle repeatedly without cross-case resource
accumulation. It is not a numerical-accuracy result, a calibration result, or a
state-of-the-art claim. It does not reopen or reinterpret attempt 5's frozen
equivalence decision.

A new held protocol may be prepared only after this source-only lifecycle gate
passes. That protocol must bind the isolated worker and result schema, forbid
fallback to the in-process wrapper, preserve all scientific cohorts and gates,
and keep target queries and scoring in the parent process.
