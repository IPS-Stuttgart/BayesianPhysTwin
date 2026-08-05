# Reviewable pull-request and workflow policy

## Purpose

A pull request must expose the exact source that reviewers, tests, and the final
merge will evaluate. Content-addressing and reproducible generation remain
important, but they do not justify hiding proposed source in encoded transport
files or replacing the reviewed commit from inside a pull-request workflow.

This policy keeps code review, CI evidence, and the merged tree aligned.

## Required pull-request shape

Every source change must be present as ordinary files in the pull-request diff.
In particular:

- do not commit generated source as Base64 chunks, archives, patches, or files
  under `.agent/` for later extraction;
- do not use a pull-request workflow to rewrite, commit, force-push, or replace
  the branch under test;
- do not grant `contents: write` to a workflow triggered by `pull_request` or
  `pull_request_target`;
- validate the exact checked-out commit, with generated reports uploaded as
  workflow artifacts rather than committed back to the branch; and
- build generated source locally before publication, then commit the final
  human-reviewable files.

A workflow may produce caches, wheels, reports, videos, checksums, or other
non-source outputs. Those outputs should be uploaded as artifacts and must not
change the source revision being reviewed.

## Release automation boundary

A release or maintenance workflow may require repository write permission only
when all of the following hold:

1. it is not triggered by `pull_request` or `pull_request_target`;
2. its write permission is explicit and limited to the required job;
3. it operates on a tag, a protected release branch, or an explicit
   `workflow_dispatch` input;
4. it never presents its generated commit as having been reviewed in an earlier
   pull-request diff; and
5. the resulting revision receives its own validation before publication.

Normal experiment, contract, compatibility, and evidence workflows should use
`contents: read` and `persist-credentials: false`.

## Scientific evidence boundary

Claim-bearing workflows may validate frozen inputs, run preregistered analyses,
and publish immutable result artifacts. They must not use target outcomes to
rewrite the estimator, protocol, source revision, or evidence definition under
review. A changed method requires a new reviewed source revision and, where the
information boundary requires it, a new protocol or target cohort.

## Enforcement

`tests/test_pull_request_workflow_integrity.py` fails when:

- a committed path contains a `.agent` source-transport directory;
- a pull-request workflow grants `contents: write`;
- a pull-request workflow invokes `git push` or resets against `origin`; or
- a pull-request workflow decodes hidden source transport from `.agent/` or
  Base64 chunks.

The test deliberately checks repository policy rather than one named workflow,
so future experiment branches receive the same protection.
