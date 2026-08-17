# GitHub Actions workflow lifecycle

BayesianPhysTwin has a large historical workflow surface because exact protocol
executions, repairs, diagnostics, and publication steps were retained while the
research program evolved. Some of those files still identify scientifically
relevant source revisions or operational history. Their existence is not, by
itself, authorization to delete them, and green workflow execution is not
scientific evidence.

The repository uses an incremental lifecycle ratchet. Existing untouched
workflows are inventoried as legacy until they are deliberately classified. A
pull request may not add, copy, rename, or modify a workflow without explicit
lifecycle metadata and the minimum supply-chain controls below. Deleting a
workflow does not reclassify it; evidence-aware archival and registry cleanup
remain separate maintainer decisions.

## Permanent workflows

A new or modified permanent workflow starts with:

```yaml
# workflow-lifecycle: permanent
# workflow-owner: IPS-Stuttgart maintainers
```

Permanent workflows must:

- represent a maintained repository capability rather than one execution;
- use a stable, non-temporary filename;
- declare top-level least-privilege `permissions`;
- declare top-level `concurrency`;
- pin every external action to a full 40-character commit SHA;
- avoid `pull_request_target`; and
- keep experiment-specific behavior in versioned Python, shell, configuration,
  or protocol files.

Prefer one parameterized `workflow_dispatch` or reusable `workflow_call` entry
point over several nearly identical experiment workflows.

## Temporary workflows

A temporary workflow is an exceptional, manually dispatched migration or repair.
It starts with:

```yaml
# workflow-lifecycle: temporary
# workflow-owner: IPS-Stuttgart maintainers
# workflow-issue: #123
# workflow-expiry: 2026-09-01
```

Temporary workflows must be `workflow_dispatch`-only. They may not run on push,
pull request, schedule, repository dispatch, or `pull_request_target`. The same
permissions, concurrency, and immutable-action requirements apply. Once the
expiry date passes, the quality gate fails until the workflow is removed or a
reviewed successor lifecycle is declared.

A temporary workflow is not an evidence archive. Before removal, preserve any
scientifically relevant exact revision, protocol identifier, run ID, artifact
ID, and digest in the owning manifest, evidence record, or immutable tag.

## Metadata and file boundary

Lifecycle metadata belongs in the leading comment header before the first YAML
key. The recognized keys are `workflow-lifecycle`, `workflow-owner`,
`workflow-issue`, and `workflow-expiry`. Duplicate or unknown `workflow-*` keys
are rejected when a workflow enters the ratchet; this catches misspelled expiry
or ownership declarations instead of silently ignoring them.

A changed workflow must be an ordinary repository-local `.yml` or `.yaml` file
directly below `.github/workflows`. Symlinked workflow definitions are rejected.
This prevents the reviewed Git path from resolving to different bytes outside
the active workflow directory.

## Quality gate

Run the same incremental policy used by CI from a checkout of the exact head
revision:

```bash
python tools/quality/check_workflow_policy.py \
  --base origin/main \
  --head "$(git rev-parse HEAD)"
```

The checker resolves both revisions to commits and refuses to inspect the
working tree when its `HEAD` differs from `--head`. The managed preflight checks
out the contributor head explicitly, fetches the comparison base when needed,
and applies the ratchet to added, copied, modified, and renamed workflow files.
Untouched legacy one-shot files remain grandfathered; cleanup must remain
evidence-aware rather than becoming a blind filename rewrite.

All managed temporary workflows are still checked for expiry on every policy
run, even when they were not changed in the current diff.

## Read-only inventory

Generate a complete machine-readable inventory and a concise operator report:

```bash
python tools/quality/check_workflow_policy.py \
  --inventory-only \
  --inventory-json workflow-lifecycle-inventory.json \
  --inventory-markdown workflow-lifecycle-report.md
```

The scheduled branch-lifecycle workflow publishes this inventory alongside its
branch report. The JSON includes every checked-in workflow; the Markdown summary
caps the legacy temporary-looking table to keep the Actions summary usable.

## Registry entries versus checked-in files

GitHub retains Actions workflow registry entries and historical runs after the
corresponding YAML file is removed from the default branch. The value returned as
`total_count` by the Actions workflows API is therefore a registry-history count,
not the number of workflow definitions currently present in the repository.

Audit both surfaces explicitly:

```bash
GITHUB_TOKEN="<read-token>" \
python tools/maintenance/workflow_registry_audit.py \
  --repository IPS-Stuttgart/BayesianPhysTwin \
  --repository-root . \
  --output-json workflow-registry-inventory.json \
  --output-markdown workflow-registry-report.md
```

The audit classifies each registry entry as checked in, orphaned and active,
orphaned and disabled, or orphaned with another state. It also reports workflow
files that are present in the checkout but not yet visible in the registry. The
inventory is content-addressed independently of its generation timestamp.

The scheduled branch-lifecycle audit runs this comparison with read-only
`actions` and `contents` permissions. It never disables a workflow. An orphaned
active registry entry may be disabled only through a separate audited maintainer
action after confirming that its YAML is absent and that any relevant run,
artifact, revision, and evidence identity have been preserved. Disabling the
registry entry does not delete historical runs or turn them into scientific
evidence.

## Cleanup order

1. Identify whether a workflow path or its exact source revision is named by a
   protocol, result, configuration, evidence record, paper handoff, or tag.
2. Move reusable execution logic into a maintained script and a parameterized
   permanent workflow.
3. Preserve evidence-bound revisions and artifact identities independently of
   the workflow filename.
4. Delete only workflows whose operational and reproducibility dependencies are
   resolved.
5. Use the registry audit to distinguish deleted YAML from still-active GitHub
   registry entries, then disable only reviewed orphaned entries.
6. Classify the small permanent control plane explicitly and retain the audit
   artifact with the maintenance record.

Do not reinterpret workflow consolidation as a change to an estimator, dataset,
information boundary, fallback decision, or scientific claim.
