# GitHub Actions inventory budget

## Purpose

The workflow lifecycle policy governs ownership, expiry, permissions,
concurrency, and immutable action references for changed workflows. The
inventory budget adds a repository-wide monotone ratchet: deleting a workflow
must lower the checked-in ceiling in the same change, so freed capacity cannot
be silently spent later.

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

## Completed one-shot retirement

The twelve historical one-shot files are now absent from
`.github/workflows`. Their exact Git blobs are retained below
`archive/github-actions/retired-one-shot-v1/`, together with a strict manifest
that binds:

- every original and archived path;
- the original Git blob SHA-1;
- the exact byte count;
- the source revision from which retirement occurred; and
- the historical contract-test blobs that continue to exercise the archived
  workflow bytes.

The one-shot retirement reduced the inventory to 83 workflows without requiring
a replacement launcher. A later permanent Deform360 source-receipt router raised
the reviewed ceiling to 84. The stale `agent-source-snapshot.yml` workflow was
then retired because it triggered only on one historical agent branch and had no
remaining repository consumer. The claim-bearing receipt router remains active,
and the exact inventory is again:

- 83 checked-in workflows; and
- zero temporary-looking workflow files.

Both values are now the current ratchet and completed retirement target.

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
