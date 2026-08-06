# ClaimBundleV1

`ClaimBundleV1` is the portable, content-addressed handoff between a finalized BayesianPhysTwin run and a paper or independent reviewer. It binds the exact run manifest, the decisive-evidence summary, participating repository revisions, claim IDs, freeze identifiers, and optional paper-facing figures or tables into one deterministic JSON descriptor.

The bundle is an evidence transport and verification format. It does not create a scientific claim, authorize confirmation-data access, or make an exploratory run confirmatory.

## Required inputs

A bundle requires:

- a valid `RunManifestV2` with a valid `paper_evidence_bindings_v1` profile;
- a `controlled` or `confirmatory` run classification;
- at least one claim ID;
- nonempty method-freeze, protocol, split, and baseline identifiers;
- one decisive-evidence summary using the stable summary, threshold-native risk-coverage, matched-count risk-coverage, and exact-fallback contracts; and
- exact agreement between the run manifest and evidence summary for `protocol_id` and `statistical_unit`.

All repositories recorded by the run manifest must be clean and pinned to exact 40-character revisions. The bundle copies those states into its own content-addressed descriptor and rechecks them during validation.

## Build a bundle

Paths may be absolute or relative to `--artifact-root`. Every emitted artifact path is normalized relative to that root.

```bash
bpt evidence bundle build claim-bundle.json \
  --artifact-root results/deform360-confirmation \
  --run-manifest run-manifest.json \
  --evidence-summary evidence-summary.json \
  --claim-binding paper-claim-binding.json \
  --figure risk_coverage=risk-coverage.svg \
  --figure calibration=interval-calibration.pdf \
  --table-data object_results=object-results.csv
```

Additional immutable files can be included with repeated `--supporting NAME=PATH` arguments. Names and relative paths must be unique.

The build command validates semantic bindings before writing the bundle. It reports the resulting bundle ID, run-manifest ID, evidence fingerprint, and artifact/repository/claim counts as JSON.

## Validate a bundle

Validate the descriptor alone:

```bash
bpt evidence bundle validate claim-bundle.json
```

Re-hash every artifact and re-run the manifest, paper-evidence, and decisive-evidence semantic checks:

```bash
bpt evidence bundle validate claim-bundle.json \
  --artifact-root results/deform360-confirmation \
  --require-claim-binding
```

Full validation fails if:

- the bundle descriptor or bundle ID was altered;
- an artifact is missing or its size or SHA-256 digest changed;
- the run manifest or paper-evidence profile no longer validates;
- the summary uses a different protocol or statistical unit;
- a participating repository, claim ID, freeze ID, split ID, baseline ID, or claim boundary differs from the bound evidence; or
- a required paper claim-binding artifact is absent.

## Artifact roles

The schema permits exactly one `run_manifest`, exactly one `evidence_summary`, and at most one `claim_binding`. Any number of `figure`, `table_data`, and `supporting` artifacts may be added. Every artifact record contains:

- a unique semantic name;
- a semantic role;
- a portable relative path;
- a SHA-256 digest;
- the exact byte size; and
- a media type.

The bundle ID is the SHA-256 digest of the canonical descriptor. It is independent of JSON indentation and key order but changes whenever any semantic field, repository state, artifact identity, claim identifier, or freeze identifier changes.

## Recommended workflow boundary

Use GitHub-hosted runners for schema, CLI, unit, package, and content-address checks. Use a protected self-hosted environment only for data- or GPU-bound execution. The self-hosted job should publish the finalized artifacts; a read-only hosted job should then build and validate `ClaimBundleV1` from those artifacts before paper claim binding.

For a one-time confirmation experiment, freeze the provider and calibrated policy before opening confirmation outcomes. Building a valid bundle after the run records that boundary; it does not replace the original access-control and evidence-use ledger.
