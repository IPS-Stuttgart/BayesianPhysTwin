# GitHub Actions inventory budget

## Purpose

The lifecycle policy prevents newly added or modified workflows from bypassing
ownership, expiry, permission, concurrency, and immutable-action requirements.
It deliberately grandfathered untouched historical files, however, so it did not
stop the checked-in workflow inventory from remaining permanently large or from
growing through a delete-and-replace sequence.

The inventory budget adds a repository-wide monotone ratchet. It does not
reinterpret historical workflows or delete evidence-bound files. It records the
current checked-in count and the exact temporary-looking retirement roster, then
makes any change to either quantity explicit and testable.

## Frozen baseline

At source revision
`45e1f4454d50fd1970af13578f0383872814125e`, the default branch contains:

- 95 ordinary `.yml` or `.yaml` files directly below `.github/workflows`;
- 12 temporary-looking historical launch, inventory, report, or revalidation
  files; and
- no managed temporary workflow in the lifecycle inventory.

The machine-readable contract is
`.github/quality/workflow-inventory-budget-v1.json`. It freezes the count and the
sorted list of all 12 temporary-looking paths. A similarly named replacement is
not interchangeable with an allowlisted historical file.

## Enforced behavior

Run the checker with:

```bash
python tools/quality/check_workflow_inventory_budget.py
```

The full Python test suite also validates the checked-in repository against the
contract. The check fails when:

- the workflow count rises above the recorded ceiling;
- a workflow is removed without lowering the ceiling in the same change;
- a temporary-looking path is added, replaced, renamed, or removed without
  updating the exact allowlist;
- the contract contains duplicate keys, unknown fields, coerced Boolean counts,
  invalid paths, an unsorted allowlist, or contradictory targets; or
- a workflow path is a symlink or resolves outside the repository.

Requiring cleanup to lower the ceiling prevents a later pull request from
silently spending the freed capacity on another workflow. Requiring exact
allowlist equality prevents one historical launcher from being replaced by a new
one-shot file while preserving the same count.

## Retirement target

The current nonblocking consolidation target is:

- at most 84 checked-in workflows; and
- zero temporary-looking workflow files.

The 84-file target permits one maintained parameterized entry point if retiring
the 12 historical launchers requires a consolidated replacement. A cleanup that
needs no replacement should lower the ceiling further.

The target is not permission to delete the 12 files blindly. Several are named
by byte-level regression tests, source-gate caller identities, issue receipts, or
frozen execution records. Their retirement must first preserve, in ordinary
content-addressed artifacts or manifests:

- the exact workflow path and source revision;
- the protocol and authorization identity;
- relevant workflow run, attempt, and artifact identifiers;
- retained artifact digests and terminal status; and
- the distinction between source-only, target, confirmation, and technical
  failure boundaries.

After those dependencies are migrated, the cleanup pull request removes the
resolved workflows, deletes their paths from the allowlist, and lowers
`maximum_checked_in_workflows` to the new exact count. The checker then makes
that reduction irreversible without another explicit contract change.

## Relationship to the lifecycle and registry audits

This budget complements rather than replaces
[workflow lifecycle](workflow_lifecycle.md):

- lifecycle metadata governs the safety and ownership of changed workflows;
- the inventory budget governs aggregate count and temporary-looking identity;
- the scheduled lifecycle inventory reports classification and policy debt; and
- the Actions registry audit distinguishes checked-in files from historical
  registry entries that GitHub retains after YAML deletion.

None of these controls establishes estimator accuracy, provider competence,
calibrated uncertainty, target-independent evidence, deployment authorization,
or state of the art.
