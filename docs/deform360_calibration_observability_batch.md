# Deform360 calibration observability batch

## Purpose

The per-object observability producer validates one visual-reference versus
visual-plus-contact comparison, and the object-balanced report validates the
complete ten-object result. The remaining assembly risk was manual: a caller had
to invoke the producer ten times, retain failures, avoid duplication or
replacement, and then pass the exact files to the report builder.

`build_deform360_calibration_observability_batch.py` removes that manual gap. One
invocation accounts for every frozen Stage-0 calibration object, produces all ten
case files through the existing claim-bearing producer, constructs the report,
self-validates the staged directory, and publishes the directory atomically.

The command does not open confirmation payloads and does not consume target
outcomes.

## Portable batch specification

The specification is strict JSON with exactly three root fields:

```json
{
  "schema": "bayesian-phystwin.deform360-calibration-observability-batch-spec",
  "schema_version": 1,
  "cases": []
}
```

`cases` must contain exactly ten unique physical-object IDs and must equal the
frozen Stage-0 calibration cohort. This equality is checked before any declared
matrix or evidence file is opened. Row order is not scientifically meaningful;
the command canonicalizes rows by object ID when computing the semantic
specification ID.

All file names in a row are canonical relative POSIX paths below `--input-root`.
Absolute paths, `..`, Windows separators, path normalization, missing files,
non-ordinary files, and symlinked path components are rejected.

### Evaluated row

```json
{
  "mode": "evaluated",
  "object_id": "<frozen calibration object ID>",
  "reference_marginal_precision": "object/reference-precision.npy",
  "candidate_marginal_precision": "object/candidate-precision.npy",
  "contact_anchor_artifact": "object/contact-anchor.json"
}
```

The shared physical-query Jacobian is supplied once on the command line. The
per-object producer independently reloads and validates the complete source
protocol, Stage-0 protocol, selection lock, visual-provider lock, names-only
plan, download manifest, prepared-source result, and successful terminal
execution record before reading the matrices.

### Retained technical-failure row

```json
{
  "mode": "technical-failure",
  "object_id": "<frozen calibration object ID>",
  "failure_evidence": "object/observability-failure.txt",
  "failure_reason": "registered observability factorization failure"
}
```

A technical failure remains in the ten-object denominator and cannot be replaced.
The row still produces a strict case artifact with the exact failure-evidence
SHA-256 and the same source lineage as the evaluated cases.

## Claim-bearing command

```bash
python scripts/science/build_deform360_calibration_observability_batch.py \
  --batch-spec /calibration/observability-batch-spec.json \
  --input-root /calibration/observability-inputs \
  --source-protocol \
    protocols/deform360_official_hub_calibration_source_v1.json \
  --stage0-protocol \
    protocols/deform360_official_hub_visuotactile_v1.json \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock /sealed/visual-provider-lock.json \
  --calibration-source-plan /sealed/calibration-source-plan.json \
  --calibration-source-download /sealed/calibration-download-manifest.json \
  --calibration-source-run-record /sealed/execution-manifest.json \
  --calibration-source-result /sealed/calibration-source-result.json \
  --query-jacobian \
    /calibration/observability-inputs/shared/physical-query-jacobian.npy \
  --implementation-revision "$(git rev-parse HEAD)" \
  --output-dir /sealed/deform360-calibration-observability-batch-v1
```

The output directory must not already exist. Concurrent publishers targeting the
same path are serialized by an exclusive publication lock; an existing lock or
output fails closed and is never removed or replaced by a process that does not
own it.

## Output directory

A successful or scientifically valid support-negative execution contains:

```text
batch-manifest.json
calibration-observability-report.json
SHA256SUMS
cases/<case-id-01>.json
...
cases/<case-id-10>.json
```

The case filenames are content-addressed. `SHA256SUMS` is canonical, sorted, and
covers the manifest, report, and all ten cases. The command reloads every case and
the report, checks all identities and hashes, rejects unexpected files or
symlinks, and verifies that the report binds the exact ten generated case files
before publication.

The manifest contains no local input paths. It binds:

- a semantic specification ID, invariant to row order and JSON formatting;
- the exact specification-file SHA-256;
- the implementation revision and physical-query identity;
- the ten content-addressed case files and exact file SHA-256 values;
- the report ID, report SHA-256, status, and support gate; and
- the exact shared query-file SHA-256 and closed information boundary.

## Exit semantics

- `0`: all ten cases were published and the frozen support rule passed;
- `3`: all ten cases were published, but observability support was insufficient;
- `2`: specification, provenance, input, case, report, validation, or publication
  failure.

Exit `3` is a completed negative scientific result. Its directory is retained and
must not be rescued through object replacement or target-informed reclassification.
The unchanged support rule remains at least 8/10 evaluated objects and at least
4/5 evaluated objects in each stratum.

## Downstream use

The generated `calibration-observability-report.json` is the artifact consumed by
`seal_deform360_calibration_with_observability.py`. A support-negative report
cannot authorize confirmation opening. A supported report is still mechanism
evidence only; the Stage-1 evidence ledger and all other authorization contracts
must independently pass before any of the twelve confirmation payloads are
opened.
