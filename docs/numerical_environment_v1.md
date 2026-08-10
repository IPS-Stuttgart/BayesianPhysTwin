# Numerical environment profile v1

Exact repository revisions are necessary but not sufficient to reproduce a
numerical result. Python, NumPy, SciPy, BLAS/LAPACK, dependency resolution, and
thread or determinism settings can change a result without changing the source
revision. `NumericalEnvironmentV1` records that state as a strict,
content-addressed runtime profile suitable for a `RunManifestV2`.

The profile is additive. Existing run manifests remain valid, and creating a
profile does not alter a method, protocol, threshold, dataset, split, or
scientific claim.

## Captured evidence

A profile records:

- the Python implementation, patch version, and compiler;
- the NumPy version and normalized `numpy.show_config()` output, including the
  reported BLAS/LAPACK build configuration;
- the SciPy version when SciPy is installed;
- byte order and logical CPU count;
- a fixed allowlist of numerical execution controls, recording unset variables
  explicitly as `null`;
- the complete sorted installed-distribution inventory and its SHA-256 digest;
- an optional dependency-lock basename, SHA-256 digest, and byte size; and
- a `profile_id` covering the entire canonical payload.

The allowlist currently includes common BLAS/OpenMP thread controls and
framework determinism controls. Arbitrary environment variables are not
collected, which avoids accidentally publishing credentials or unrelated
machine state.

## Capture after installing the locked environment

Use the resolver input that actually created the environment. A post-hoc
`pip freeze` is useful diagnostic evidence, but it is not a substitute for the
input lock or constraints file.

```bash
python -m bayesian_phystwin.numerical_environment_v1 capture \
  runs/example/numerical-runtime.json \
  --dependency-lock requirements.lock
```

The command does not overwrite an existing output unless `--force` is supplied.
It prints the profile ID, installed-distribution count, and dependency-lock
identity.

The resulting file is a runtime JSON fragment:

```json
{
  "numerical_environment_v1": {
    "profile_id": "...",
    "schema_name": "bayesian_phystwin.numerical_environment",
    "schema_version": 1
  }
}
```

The actual payload contains the complete fields listed above.

## Bind it to a run manifest

Pass the fragment through the existing `RunManifestV2` CLI and also register the
lockfile as a manifest input. The nested profile binds the lock identity; the
manifest artifact record additionally permits file-presence, size, and digest
verification after copying or extraction.

```bash
bpt run manifest create runs/example/manifest.json \
  --run-id example-confirmation-v1 \
  --classification confirmatory \
  --statistical-unit interaction \
  --command-line 'python scripts/run_example.py' \
  --repository-root . \
  --configuration-json runs/example/config.json \
  --information-boundary-json runs/example/information-boundary.json \
  --runtime-json runs/example/numerical-runtime.json \
  --protocol-id example-protocol-v1 \
  --artifact-root . \
  --input dependency_lock=requirements.lock \
  --output-artifact metrics=runs/example/metrics.json
```

`RunManifestV2.evidence_fingerprint` then covers the complete numerical profile
in addition to repositories, configuration, information boundary, artifacts,
claims, and protocol identifiers.

## Validate before promotion or paper assembly

Validate the profile itself:

```bash
python -m bayesian_phystwin.numerical_environment_v1 validate \
  runs/example/numerical-runtime.json \
  --require-dependency-lock
```

Then validate the complete run manifest and its artifacts:

```bash
bpt run manifest validate runs/example/manifest.json \
  --artifact-root .
```

Profile validation rejects:

- unknown or missing schema fields;
- noncanonical scalar values and non-finite JSON;
- duplicate JSON keys;
- an incomplete execution-control set;
- unsorted, duplicate, or inconsistent installed distributions;
- NumPy or SciPy versions that disagree with the distribution inventory;
- altered NumPy configuration or distribution digests;
- an altered profile ID;
- an absent dependency lock when one is required; and
- oversized or malformed runtime fragments.

## Interpretation boundary

The profile records the environment that executed a result. It does not prove
that every dependency is deterministic, that hardware behaved identically, or
that a scientific protocol was valid. Confirmation still requires the frozen
method and protocol, sealed information boundary, dataset and split identities,
negative controls, proper scoring, and the applicable evidence-decision gate.

For future Full-22, Prob4D, and Causal4D evidence workflows, capture the profile
immediately after environment installation and before any candidate forecasts
or target-bearing scoring. Preserve the runtime fragment, the dependency lock,
and the finalized `RunManifestV2` together with the numerical outputs.
