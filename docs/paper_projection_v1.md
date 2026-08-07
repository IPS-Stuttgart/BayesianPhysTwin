# PaperProjectionV1

`PaperProjectionV1` is the strict, read-only paper-ingestion view of a verified
[`ClaimBundleV1`](claim_bundle_v1.md). It closes the last semantic gap between a
content-addressed bundle and a paper claim registry: every projected claim must
resolve to exactly one compact-table row owned by that claim, and both the result
artifact and compact table must be declared outputs of the bound run manifest.

The projection does not create or promote a scientific claim, authorize access
to confirmation data, alter an estimator, select a guard, or make an exploratory
run confirmatory. It only transports already frozen claim evidence into a
smaller deterministic document that a paper repository can validate without
reimplementing BayesianPhysTwin's bundle semantics.

## Additional validation

Before emitting a projection, BayesianPhysTwin performs full
`ClaimBundleV1` validation and then requires:

- exactly one version-1 `bayesian_phystwin.claim_evidence_bindings` artifact;
- no migration exception for any projected claim;
- one binding for every bundle claim and no unknown binding;
- result and table references whose names, paths, and SHA-256 digests agree with
  both the bundle and the run manifest's output artifacts;
- a JSON table using the version-1
  `bayesian_phystwin.compact_claim_table` schema;
- globally unique compact-table row IDs;
- exactly one referenced row for every claim;
- exact equality between each row's `claim_id` and its binding; and
- finite JSON evidence values.

A CSV or arbitrary JSON table can remain a general `ClaimBundleV1` artifact, but
it cannot be projected as paper claim evidence. The strict projection accepts
only the compact-table schema used by the paper evidence validator.

## Generate paper-facing artifacts

```bash
bpt evidence bundle project-paper \
  results/deform360-confirmation/claim-bundle.json \
  --artifact-root results/deform360-confirmation \
  --output results/deform360-confirmation/paper-projection.json \
  --markdown results/deform360-confirmation/paper-projection.generated.md
```

The JSON output is content-addressed by `projection_id`. The optional Markdown
contains the same claim rows and repository lock in a human-reviewable form.
Both files are written atomically and refuse replacement unless `--force` is
supplied.

The command prints a compact JSON receipt containing the projection ID, source
bundle ID, output paths, and claim count.

## Paper repository boundary

A paper-side importer should treat the projection as immutable input and verify:

1. the projection ID;
2. the expected bundle and run-manifest identities;
3. exact claim-ID coverage;
4. exact equality between projected evidence and the corresponding
   `claims.json` entry; and
5. exact repository revisions for every cited source.

Generated prose or tables must fail CI when they drift from the projection.
Paper-side code must not infer a stronger claim from metric signs, thresholds,
or status labels; interpretation remains explicit in the claim registry.

For the frozen Deform360 experiment, projection happens only after the sealed
calibration and one-time confirmation workflow has produced its immutable
result. This command does not change the 10-object calibration cohort, the 12
unopened confirmation objects, the six required comparison arms, or any
confirmation-opening authorization.
