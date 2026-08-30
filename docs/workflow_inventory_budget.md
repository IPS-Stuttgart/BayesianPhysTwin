# GitHub Actions inventory budget

## Purpose

The workflow lifecycle policy governs ownership, expiry, permissions,
concurrency, and immutable action references for changed workflows. The
inventory budget adds a repository-wide ratchet: deleting a workflow must lower
the checked-in ceiling in the same change, so freed capacity cannot be silently
spent later. A separately reviewed permanent workflow may raise the ceiling only
when the same change records that increase explicitly.

The budget is an engineering and provenance control. It does not alter an
estimator, protocol, target-access boundary, artifact, metric, or scientific
claim.

## Original baseline

At source revision
`45e1f4454d50fd1970af13578f0383872814125e`, the default branch contained:

- 95 ordinary `.yml` or `.yaml` files directly below `.github/workflows`;
- 12 temporary-looking historical launch, inventory, report, or revalidation
  files; and
- no managed temporary workflow in the lifecycle inventory.

The machine-readable contract is
`.github/quality/workflow-inventory-budget-v1.json`.

## Completed one-shot retirement and current reviewed increases

The twelve historical one-shot files are absent from `.github/workflows`. Their
exact Git blobs are retained below
`archive/github-actions/retired-one-shot-v1/`, together with a strict manifest
that binds:

- every original and archived path;
- the original Git blob SHA-1;
- the exact byte count;
- the source revision from which retirement occurred; and
- the historical contract-test blobs that continue to exercise the archived
  workflow bytes.

Subsequent lifecycle work continued to ratchet the active inventory down to 81
ordinary workflows with zero temporary-looking files. The permanent
`cross-intervention-criterion-evidence.yml` workflow was a deliberately reviewed
addition: it runs the target-free controlled falsification study, requires
primary/replay byte identity, binds regenerated output to the retained result,
and records the exact reviewed head and canonical Python/NumPy runtime.

The permanent `deform360-real-evaluation.yml` workflow is a second explicitly
reviewed addition. It exposes one maintained, request-triggered, read-only path
for bounded public Deform360 evaluation on `gpuserver4090`; runs synthetic
causal-boundary contracts on hosted infrastructure; excludes the registered
reserved-object roster before payload access; uploads aggregate evidence only;
and fixes every claim and fresh-confirmation authorization flag to false. Its
addition raises the checked-in ceiling by one, from 82 to 83, without changing
the retirement target.

The exact active inventory and targets are therefore:

- 83 checked-in workflows;
- zero temporary-looking workflow files;
- a retirement target of at most 81 checked-in workflows; and
- a retirement target of zero temporary-looking workflows.

The two-workflow retirement gap is intentional and visible. A future
consolidation or retirement should lower the ceiling in the same change rather
than silently reusing that capacity.

Validate the active inventory with:

```bash
python tools/quality/check_workflow_inventory_budget.py
```

Validate exact archival preservation and the inactive original paths with:

```bash
python tools/quality/check_retired_workflow_archive.py
```

The full test suite runs both checks.

## Enforced behavior

The inventory check fails when:

- the workflow count rises above or falls below the exact recorded ceiling;
- the temporary-looking roster differs from the exact allowlist;
- the contract contains duplicate keys, unknown fields, coerced Boolean counts,
  invalid paths, an unsorted allowlist, or contradictory targets; or
- an active workflow path is a symlink or escapes the repository.

The archive check separately fails when:

- a retired original path reappears under `.github/workflows`;
- an archived workflow or historical contract-test blob changes;
- a recorded byte count, path, total, or Git blob SHA-1 changes;
- the archive manifest is malformed, reordered, duplicated, or noncanonical; or
- an archive path is missing, nonregular, symlinked, or outside the repository.

Restoring an archived launcher requires a separately reviewed workflow and
protocol decision. Copying the historical file back into the active directory
is deliberately rejected.

## Relationship to other controls

This budget complements rather than replaces
[workflow lifecycle](workflow_lifecycle.md):

- lifecycle metadata governs the safety and ownership of changed workflows;
- the inventory budget governs aggregate count and temporary-looking identity;
- the retired archive preserves historical bytes without activating them;
- the scheduled lifecycle inventory reports classification and policy debt; and
- the Actions registry audit distinguishes checked-in files from historical
  registry entries retained by GitHub after YAML deletion.

None of these controls establishes estimator accuracy, provider competence,
calibrated uncertainty, target-independent evidence, deployment authorization,
or state of the art.
