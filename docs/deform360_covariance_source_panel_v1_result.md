# Deform360 Covariance-Only Source Result v1

Date: 2026-08-27

Status: terminal `source-technical-negative`; confirmation is not authorized.

## Question

The frozen public-data study asked whether the covariance-only
`independent_endpoint_v1` donor could improve probabilistic forecasts while
leaving the exact `last_residual` point mean byte-identical. The protocol first
required a complete panel of 100 prefix-only source records. Only after that
barrier could source suffixes be opened for the registered NLL, coverage, and
width decision; a positive source decision was required before any of the 12
closed confirmation sessions could be processed.

## Inventory

The third registered metadata-only inventory run, GitHub Actions run
`33012437014`, completed successfully on protected `main` revision
`d772b8ba84e52b99beb22e1aab2a37d766abab77`:

- all `10/10` frozen source units were present;
- `5,432` header records were retained;
- the inventory ID was
  `254fc7e09e339d22d6030cc9d6a467cba9c01f757910942cd574c922341649db`;
- no array values, source suffix, confirmation payload, or target outcome was
  read.

Malformed public files with an `.npy` suffix were retained as explicit
header-error metadata. They were not dropped, replaced, or interpreted as
NumPy arrays.

## Registered Attempt

The sole registered producer dispatch created GitHub Actions run
`33012751418`. It consumed the write-once attempt ledger before source-value
processing, verified the frozen upstream source execution and compact artifact,
and then stopped during source-panel materialization with:

```text
ValueError: metric gauge lacks eight independent causal clusters
```

The producer sealed `0/100` records and emitted the registered bounded
technical receipt:

- status: `source-technical-negative`;
- diagnostic: `provider-materialization-failure`;
- terminal stage: `source-panel-production`;
- receipt ID:
  `e705d19c50273e2b1f09ff1ccd09fd4cf08b250241f2d0f4c4f0a17bcb775df3`.

No source suffix was opened, no metric was computed, and no confirmation data
was accessed.

## Decision

The frozen source gate is not evaluable because the mandatory 100-record
barrier does not exist. Under the preregistered no-retry and complete-accounting
rules, this is a terminal negative for this exact public provider path:

- do not retry the producer;
- do not lower the eight-cluster gauge threshold;
- do not replace the failed unit;
- do not score a partial source subset; and
- do not process the 12 confirmation sessions.

This result does not show that `independent_endpoint_v1` covariance is
inaccurate or miscalibrated. It shows that the frozen disjoint visual metric
provider could not materialize the complete source panel needed to evaluate
that covariance claim. A future attempt requires a separately preregistered
provider or observation contract and a fresh source scope.

## Evidence

The compact, source-value-free capsule is under
`results/sota/diagnostics/deform360_covariance_source_panel_v1/`. It contains
the exact technical receipt, attempt ledger, traceback, content-addressed
summary, and file hashes. The original Actions artifact is
`covariance-source-technical-receipt-v1-33012751418` with artifact digest
`sha256:7396755b951d55cd928988798609ac61f1b11e113f6eabc849de0559965606a6`.
