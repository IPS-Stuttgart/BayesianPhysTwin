# Reproducible run manifests

Bayesian-PhysTwin result bundles use the
`bayesian_phystwin.run_manifest` schema. Version 1 remains supported exactly as
released. New manifests created by the CLI use version 2, which binds the wider
cross-repository and runtime evidence context needed by paper-facing results.

## Version policy

`RunManifestV1` is unchanged and its original loader remains available from
`bayesian_phystwin.run_manifest`. Existing frozen bundles retain their exact
content addresses.

`RunManifestV2` lives in `bayesian_phystwin.run_manifest_v2`. The grouped
manifest CLI creates V2 records and validates both V1 and V2. V2 adds:

- automatic primary-repository revision and dirty-state discovery;
- exact role-bearing locks for participating PhysTwin, Prob4D, Causal4D, paper,
  environment, or other dependency repositories;
- portable runtime metadata and explicitly selected environment variables;
- claim, method-freeze, protocol, split, and baseline identifiers; and
- a timestamp-independent `evidence_fingerprint`.

A schema change never reinterprets an existing version.

## Repository state

When `--revision` is omitted, the command reads the exact revision, GitHub
`owner/name` remote, tracked changes, and untracked files from
`--repository-root`. A dirty checkout fails closed unless `--allow-dirty` is
supplied.

The explicit `--revision` path remains available outside a Git checkout. V2
requires a full 40-character commit and records `--dirty` only when the caller
also acknowledges it with `--allow-dirty`.

Additional participating repositories are supplied as a JSON array:

```json
[
  {
    "repository": "Jianghanxiao/PhysTwin",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "dirty": false,
    "role": "upstream"
  },
  {
    "repository": "FlorianPfaff/Prob4D",
    "revision": "89abcdef0123456789abcdef0123456789abcdef",
    "dirty": false,
    "role": "observation"
  }
]
```

Supported related-repository roles are `upstream`, `observation`, `downstream`,
`paper`, `environment`, and `dependency`. Related repositories cannot use the
`primary` role or duplicate another repository identity.

## Creating V2

Create the manifest only after all named outputs are immutable:

```bash
bpt run manifest create runs/example/manifest.json \
  --run-id phystwin-full22-anchor-v1 \
  --classification confirmatory \
  --statistical-unit interaction \
  --command-line 'bpt experiment run confirm-phystwin-bayesian-anchor ...' \
  --repository-root . \
  --related-repositories-json runs/example/repositories.lock.json \
  --configuration-json runs/example/config.lock.json \
  --information-boundary-json runs/example/information-boundary.json \
  --runtime-json runs/example/runtime.json \
  --environment-variable CUDA_VISIBLE_DEVICES \
  --claim-id bpt.full22_anchor_released_contract \
  --method-freeze-id full22-anchor-method-v1 \
  --protocol-id official-ordered-22-v1 \
  --split-id development3-confirmation19-v1 \
  --baseline-id released-phystwin-reproduction-v1 \
  --artifact-root runs/example \
  --input method_lock=method_lock.json \
  --output-artifact metrics=metrics.json
```

`runtime.json` can add numerical execution details Python cannot infer
portably, such as GPU model, CUDA and driver versions, Warp and Torch builds,
container-image digest, and deterministic versus atomic spring-force mode.
It cannot replace Python version, operating system, machine, processor, byte
order, or the explicitly selected environment captured by the runtime. Runtime,
configuration, information-boundary, and related-repository JSON reject
duplicate object keys and non-finite constants rather than relying on parser
coercion. Environment variables are never collected wholesale; only explicitly
named canonical identifiers are recorded.

Every V2 manifest binds:

- exact primary and related repository states;
- the executed command and finite configuration;
- result classification (`controlled`, `exploratory`, `confirmatory`,
  `diagnostic`, or `infrastructure`);
- statistical unit and information boundary;
- random seeds and installed package versions;
- runtime metadata;
- named claim and protocol identifiers; and
- SHA-256 identities and byte sizes of input and output artifacts.

## Two V2 identities

`manifest_id` covers the complete finalized record, including `created_utc` and
free-form notes.

`evidence_fingerprint` excludes only the creation timestamp and notes. It
remains stable when the same scientific record is copied into a paper bundle
with different archival commentary. It still covers repositories, command,
configuration, information boundary, artifacts, runtime, claims, and protocol
identifiers.

Copy the finalized JSON with the result bundle. Do not recreate it during paper
assembly.

## Validation

Validate either version and all referenced files after copying or extraction:

```bash
bpt run manifest validate runs/example/manifest.json \
  --artifact-root runs/example
```

Artifact paths are stored relative to `--artifact-root`. Validation rejects path
traversal, missing files, changed sizes, changed digests, schema drift, and
content-address tampering. V2 additionally rejects evidence-fingerprint drift,
nonexact repository revisions, duplicate repository identities, and malformed
claim/runtime records.

A manifest does not turn a run into scientific evidence by itself. Method
freezing, split integrity, target-data sealing, negative controls, statistical
analysis, and claim review remain separate requirements. Paper-promotable
entries in the canonical `BayesianPhysTwin-Paper` claim registry should cite
the validated manifest ID, evidence fingerprint, and compact result-artifact
digests.

Only `bpt` is installed. Historical manifests may retain removed `bpt-*`
command strings as immutable provenance, while current non-stable commands use
the `experiment`, `diagnostic`, or `archive` lifecycle runner selected by the
registry.
