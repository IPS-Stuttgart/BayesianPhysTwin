# Open-27 Source Transfer Contract

## Purpose

The frozen pairwise bias-aware source evaluator needs four already-open
Deform360 input roots. Those roots currently exist only outside the authorized
`gpuserver4090` workspace. Copying them informally would leave the source result
dependent on an unrecorded cohort, missing files, or silently changed arrays.

This transfer contract inventories and stages exactly the required 27-case
bytes before any candidate score is computed. It does not authorize access to
held-v8, select a fresh object, or evaluate a target.

## Bound Inputs

For every fixed source case, the manifest binds:

- the physical prediction seal and its sealed prediction archive;
- the already-open source outcome manifest and opaque target payload;
- the raw-camera prefix measurement archive;
- the cycle/correlation-aware covariance archive;
- the selected physical/persistence baseline archive.

Every file is bound by logical root, relative path, byte length, and SHA-256.
The manifest has a canonical content digest. Validation also checks the
cross-file 76-frame material-node shape, frame-zero identity, update frames,
center identities, camera count, support masks, and covariance shape.

The target pickle is never deserialized during inventory, staging, or
validation. Its bytes are checked only against the already-open source outcome
manifest. Scoring remains confined to the separate frozen evaluator.

## Independent Staging

On a host that is already authorized to read the four open-source roots:

```bash
python -m bayesian_phystwin.cli.deform360_pairwise_bias_aware_transfer stage \
  --source-root /path/to/independent-source-v1 \
  --measurement-root /path/to/raw-camera-measurements \
  --uncertainty-root /path/to/cycle-uncertainty \
  --selected-baseline-root /path/to/selected-raw-baselines \
  --destination /new/path/to/open27-transfer-v1
```

The destination must not exist. Files are first copied into a partial
directory, fully rehashed and schema-checked, and then atomically renamed.
Copy the completed directory to `gpuserver4090`, then run:

```bash
python -m bayesian_phystwin.cli.deform360_pairwise_bias_aware_transfer \
  validate --bundle-root /path/to/open27-transfer-v1
```

The source comparison can consume only that validated bundle:

```bash
python -m bayesian_phystwin.cli.deform360_pairwise_bias_aware_source \
  --bundle-root /path/to/open27-transfer-v1 \
  --output /new/path/to/source-v1-result
```

The result records the canonical transfer-manifest digest. Validation failure
stops the run; there is no best-effort or partial-cohort mode.

## Claim Boundary

This contract improves provenance and operational reproducibility only. It
does not add empirical evidence. The 27 episodes are already-open source
development cases. Even a passing source gate can justify only a separately
locked fresh-object evaluation after the independent held-v8 all-attempt
exclusion manifest becomes available.
