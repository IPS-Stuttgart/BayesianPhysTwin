# Reproducible run manifests

Bayesian-PhysTwin experiments can attach a content-addressed
`bayesian_phystwin.run_manifest` version-1 record to each compact result bundle.
The manifest separates an implemented workflow from a paper-facing result by
recording the exact evidence boundary used for that run.

A manifest binds:

- repository and exact revision, including whether the checkout was dirty;
- the executed command and finite configuration object;
- result classification (`controlled`, `exploratory`, `confirmatory`,
  `diagnostic`, or `infrastructure`);
- the statistical unit and information boundary;
- random seeds and installed package versions; and
- SHA-256 identities and byte sizes of named input and output artifacts.

Create one after the run outputs have been finalized:

```bash
bpt run manifest create runs/example/manifest.json \
  --run-id phystwin-full22-anchor-v1 \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --classification confirmatory \
  --statistical-unit interaction \
  --command-line 'bpt-confirm-phystwin-bayesian-anchor ...' \
  --configuration-json runs/example/config.lock.json \
  --information-boundary-json runs/example/information-boundary.json \
  --artifact-root runs/example \
  --input method_lock=method_lock.json \
  --output-artifact metrics=metrics.json
```

Validate the manifest and all referenced files after copying or extracting a
bundle:

```bash
bpt run manifest validate runs/example/manifest.json \
  --artifact-root runs/example
```

Artifact paths are stored relative to `--artifact-root`. Validation rejects
path traversal, missing files, changed byte sizes, changed digests, schema
changes, and manifest-content tampering. The manifest does not by itself turn a
run into scientific evidence: method freezing, split integrity, statistical
analysis, negative controls, and claim review remain separate requirements.

The legacy `bpt-run-manifest` entry point remains available. The grouped `bpt`
command is the preferred discoverable interface for stable operations; existing
experiment-specific `bpt-*` entry points remain supported for frozen runs.
