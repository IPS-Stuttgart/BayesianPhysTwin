# Deform360 calibration download resilience

## Purpose

The frozen official-Hub calibration-source execution is allowed to download only
the exact files already admitted by the names-only plan for the ten registered
calibration objects. The twelve confirmation objects and all target outcomes
remain forbidden.

Workflow run `31144893508` passed the 10/10 names-only admission gate and the
independent confirmation-boundary check, but failed before producing a download
manifest because the unauthenticated Hugging Face Xet token endpoint returned
HTTP 429. This is a transport failure, not a calibration-support result.

## Retry boundary

The acquisition path now:

- disables Xet for this lane before importing the Hub download client;
- accepts one optional repository secret, `HF_TOKEN`, passed to the reusable
  workflow as the narrower `hf_token` secret;
- never prints or persists the token value;
- caps concurrent downloads at two workers, even when an older caller requests
  a larger value;
- reuses an already present file only after its exact size and SHA-256 identity
  verify;
- retries only HTTP 429, selected HTTP 5xx responses, and recognized connection
  or timeout failures;
- honors a numeric `Retry-After` header and otherwise uses deterministic bounded
  exponential backoff;
- stops after six total attempts per file; and
- fails immediately on authentication/authorization errors, unknown revisions or
  files, local path violations, size mismatch, or digest mismatch.

The Hugging Face client retains its ordinary incomplete-download cache, so a
retry may resume transport internally. The claim-bearing output still contains
only files whose final bytes match the sealed plan.

## Frozen scientific boundary

This change does not alter:

- dataset repository or revision;
- processing repository or revision;
- the 10-calibration/12-confirmation object split;
- selected file names, declared sizes, LFS digests, or object accounting;
- the 8/10 overall and 4/5-per-stratum support gates;
- technical-failure retention or the no-replacement rule;
- the visual provider, estimator, physical query, guard, threshold, comparison
  arms, calibration artifacts, evidence ledger, or confirmation decision rule.

A retry-exhausted download remains a failed calibration-source execution with a
strict terminal record. It is not converted into scientific support and does not
authorize confirmation access.
