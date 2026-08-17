# Deform360 v6 clean execution-worktree repair

## Observed failure

Protected-main workflow run `31465387504` reached the first registered physical
source case, materialized its complete source payload, and then stopped at
`stage-prefix:026-sock-cloth-ep0007`.

The retained execution receipt reports
`status="source-technical-failure-retained"`, zero physical manifests, and zero
source-prediction seals. The bounded stage log identifies the exact failure:

```text
ValueError: repository is dirty
```

The primary checkout was not modified in a tracked file. It contained the four
intentional untracked dependency checkouts created by the workflow:
Deform360, official PhysTwin, SAM2, and Causal4D source discovery. The historical
checksum-bound adapter correctly requires a completely clean execution
repository, so passing the primary checkout was inconsistent with the workflow
layout.

## Repair

The active wrapper now creates a detached Git worktree at the exact
`BPT_SOURCE_SHA` and verifies both its revision and empty porcelain status before
running the immutable selector wrapper and archived science runner. The delegated
run starts inside that exact worktree. Its workspace and Python source root are
also bound to the same path, so relative lock, script, configuration, and module
references resolve inside the repository validated by the historical adapter.

Dependency repositories remain at their existing frozen paths and are not copied
into the validated execution tree. The worktree is removed through the existing
exit trap. The frozen selector-wrapper blob, archived science-runner blob,
physical-upstream revision, prepared-source inventory, source cohort, camera
roster, estimator, covariance schedule, prediction barrier, and target-closed
information boundary are unchanged.

## Information boundary

This is a technical execution repair. It does not open a development suffix,
confirmation payload, fresh target payload, or target outcome. It does not
authorize replacement, tuning, target selection, claim promotion, or a
scientific conclusion. The next protected-main run must still produce all 100
source-prediction seals before any downstream fresh-object action is admissible.
