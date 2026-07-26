# Reproducible run manifests

Bayesian-PhysTwin experiments can attach a content-addressed
`bayesian_phystwin.run_manifest` version-1 record to each compact result bundle.
The manifest records both an immutable run instance and a timestamp-independent
scientific evidence fingerprint.

A manifest binds:

- the primary repository revision and whether the checkout was dirty;
- exact revisions and roles for participating PhysTwin, Prob4D, Causal4D, paper,
  environment, or other dependency repositories;
- the executed command and finite configuration object;
- result classification (`controlled`, `exploratory`, `confirmatory`,
  `diagnostic`, or `infrastructure`);
- the statistical unit and information boundary;
- random seeds, installed package versions, and portable runtime metadata;
- selected, explicitly named environment variables;
- claim, method-freeze, protocol, split, and baseline identifiers; and
- SHA-256 identities and byte sizes of named input and output artifacts.

## Repository state

When `--revision` is omitted, the command reads the exact revision, GitHub
`owner/name` remote, tracked changes, and untracked files from
`--repository-root`. A dirty checkout fails closed unless `--allow-dirty` is
supplied. The explicit `--revision` path remains available for controlled
manifest construction outside a Git checkout, but it requires a full
40-character Git revision.

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
`paper`, `environment`, and `dependency`. Frozen experiments should list every
repository whose code or contract affects the result rather than recording only
the Bayesian-PhysTwin checkout.

## Creating a manifest

Create the manifest after all named outputs are immutable:

```bash
bpt run manifest create runs/example/manifest.json \
  --run-id phystwin-full22-anchor-v1 \
  --classification confirmatory \
  --statistical-unit interaction \
  --command-line 'bpt-confirm-phystwin-bayesian-anchor ...' \
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

`runtime.json` can add hardware and numerical execution details that Python
cannot infer portably, for example GPU model, CUDA and driver versions, Warp and
Torch build information, container-image digest, and deterministic versus
atomic spring-force mode. Environment variables are never collected wholesale;
only names passed with `--environment-variable` are recorded.

## Two identities

`manifest_id` covers the complete manifest, including `created_utc` and free-form
notes. It identifies one finalized run record.

`evidence_fingerprint` excludes only the creation timestamp and notes. It is
stable when the same scientific evidence record is copied into a paper bundle or
regenerated with different archival commentary. It still covers repository
states, command, configuration, information boundary, artifacts, runtime,
claims, and protocol identifiers.

Copy the finalized JSON with the result bundle. Do not recreate the manifest
while assembling the paper.

## Validation

Validate the manifest and all referenced files after copying or extracting a
bundle:

```bash
bpt run manifest validate runs/example/manifest.json \
  --artifact-root runs/example
```

Artifact paths are stored relative to `--artifact-root`. Validation rejects path
traversal, missing files, changed byte sizes, changed digests, schema drift,
evidence-fingerprint drift, and manifest-content tampering.

The manifest does not by itself turn a run into scientific evidence. Method
freezing, split integrity, negative controls, statistical analysis, target-data
sealing, and claim review remain separate requirements. Paper-promotable claims
should reference validated manifest IDs and compact result-artifact digests in
the canonical `BayesianPhysTwin-Paper` claim registry.

The legacy `bpt-run-manifest` entry point remains available. The grouped `bpt`
command is the preferred discoverable interface for stable operations; existing
experiment-specific `bpt-*` entry points remain supported for frozen runs.
