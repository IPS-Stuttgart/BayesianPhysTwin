# Provenance-safe branch lifecycle

BayesianPhysTwin keeps exact source revisions inside protocols, manifests,
workflows, and evidence records. A branch may therefore be operationally stale
while its head commit still matters scientifically. Branch cleanup must preserve
that distinction.

The scheduled **Branch lifecycle audit** is deliberately read-only. It inventories
all repository branches, associates same-repository pull requests, scans tracked
evidence and protocol files for exact 40-character head revisions, checks whether
each head is reachable from `main`, and records exact tags that already preserve
the head. It never creates, moves, or deletes a Git ref.

## Run locally

Fetch every head and tag before running the audit:

```bash
git fetch --prune origin \
  '+refs/heads/*:refs/remotes/origin/*' \
  '+refs/tags/*:refs/tags/*'

GITHUB_TOKEN=... python tools/maintenance/branch_lifecycle_audit.py \
  --repository-root . \
  --repository IPS-Stuttgart/BayesianPhysTwin \
  --default-branch main \
  --stale-days 60 \
  --output-json branch-lifecycle-inventory.json \
  --output-markdown branch-lifecycle-inventory.md
```

The token is optional for public API access but strongly recommended to avoid the
unauthenticated rate limit. The command uses only read-only REST requests and
local read-only Git commands.

## Classifications

`retain-default`, `retain-protected`, `retain-open-pr`, and `retain-recent` are
not cleanup candidates.

`deletion-candidate-merged` means the stale branch head is an ancestor of the
current default-branch head. `deletion-candidate-tagged` means an exact tag
already preserves the stale head. Both are **human review queues**, not deletion
authorizations.

`retain-evidence-needs-tag` means a tracked evidence, protocol, result,
configuration, documentation, or workflow file names the exact head, while the
head is neither reachable from `main` nor preserved by a tag.
`retain-archive-needs-tag` applies the same preservation rule to an `archive/`
branch whose branch ref is the only known durable reference.

`review-closed-pr-unpreserved` and `review-unreferenced-unmerged` require manual
inspection. A closed pull request does not guarantee that its exact head remains
reachable after a squash merge, and absence from the scanned evidence roots is
not proof that an unmerged experiment has no value.

## Cleanup procedure

1. Review the JSON inventory at an exact repository revision.
2. Resolve every open pull request before considering its branch.
3. For a referenced or archived head that is not reachable from `main`, create an
   annotated immutable tag at the exact reported revision. The report includes a
   conservative tag-name suggestion; replace it with a protocol-specific
   `evidence/...` tag when a registered protocol supplies a better identity.
4. Verify the tag target independently with `git rev-parse <tag>^{commit}`.
5. Delete only branches reported as already preserved, or unmerged branches for
   which a maintainer has separately demonstrated that no source, evidence,
   review, or reproducibility dependency remains.
6. Retain the JSON report alongside the maintenance record. Do not reinterpret a
   branch cleanup report as scientific evidence or claim authorization.

The audit intentionally does not enable automatic branch deletion. Ref deletion
is an explicit repository-maintenance action outside GitHub Actions.
