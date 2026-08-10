# Numerical environment profile v1

Source revisions alone do not identify a numerical experiment. Results can change
when the Python patch release, NumPy build, BLAS/LAPACK implementation, thread
controls, installed distributions, dependency lock, or container image changes.

`bayesian_phystwin.numerical_environment_v1` provides a strict,
content-addressed record of those inputs. It complements `RunManifestV2`; it does
not replace repository provenance, dataset hashes, split hashes, seeds, or the
scientific protocol.

## What the profile records

A v1 profile contains:

- Python implementation, version, compiler, operating-system description,
  architecture, processor string, and byte order;
- the imported NumPy version and normalized `numpy.__config__.show()` output,
  including the available BLAS/LAPACK build information;
- a closed allowlist of execution controls such as `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `PYTHONHASHSEED`, and
  `CUBLAS_WORKSPACE_CONFIG`;
- either the complete installed-distribution inventory or a caller-selected
  inventory that always includes NumPy;
- an optional SHA-256 digest and size for a dependency lock;
- an optional immutable OCI container digest.

The `profile_id` is the SHA-256 digest of the canonical profile descriptor. It
changes when any recorded numerical input changes. The JSON loader rejects
unknown or missing fields, duplicate keys, non-finite JSON, noncanonical package
names, mismatched NumPy versions, malformed digests, and content-address drift.

Only explicitly allowlisted environment variables are collected. Arbitrary
environment variables, credentials, and tokens are not copied into the profile.

## Capture before numerical imports in the runner

Set execution controls in the shell or workflow before Python imports NumPy or
SciPy. Capturing a value after a numerical backend has initialized records the
value but cannot retroactively apply it.

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Capture and publish the profile before producing predictions:

```python
from bayesian_phystwin.numerical_environment_v1 import (
    capture_numerical_environment_profile,
    numerical_environment_runtime_binding,
    write_numerical_environment_profile,
)

profile = capture_numerical_environment_profile(
    dependency_lock="requirements.lock",
    container_image_digest=(
        "sha256:0123456789abcdef0123456789abcdef"
        "0123456789abcdef0123456789abcdef"
    ),
)
write_numerical_environment_profile(
    "evidence/numerical-environment-v1.json",
    profile,
)

runtime_binding = numerical_environment_runtime_binding(profile)
```

`write_numerical_environment_profile` refuses to replace an existing artifact by
default. Pass `overwrite=True` only when intentionally writing a different output
location or rebuilding a disposable workspace. Immutable evidence stores should
never overwrite a published profile.

For small smoke tests, a selected package inventory can reduce fixture size:

```python
profile = capture_numerical_environment_profile(
    dependency_lock="requirements.lock",
    package_names=("numpy", "scipy", "bayesian-phystwin"),
)
```

Promotion and archival runs should normally retain the complete installed
inventory by leaving `package_names=None`.

## Bind the profile to `RunManifestV2`

Store the profile JSON as a named input or output artifact and merge the small
runtime binding into the manifest runtime metadata. For example:

```python
runtime_environment = {
    **default_runtime_environment(
        selected_environment_variables=(
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ),
    **numerical_environment_runtime_binding(profile),
}
```

The run manifest should additionally record the profile JSON as a hashed artifact.
This gives two independent checks:

1. the manifest binds the expected `profile_id` in runtime metadata;
2. the artifact record binds the exact serialized profile bytes.

A consumer should load the profile with
`load_numerical_environment_profile(...)` and require the manifest binding, loaded
`profile_id`, and artifact digest to agree before accepting evidence.

## Cache and rerun policy

A cached experimental cell is reusable only when all scientific inputs and the
numerical environment match. A robust cache key should cover at least:

```text
protocol version
candidate source revision
runner source revision
dataset and split digests
configuration digest
seed
numerical environment profile_id
```

Do not treat similar package versions, the same NumPy version with a different
build configuration, or the same lockfile name with a different digest as cache
hits. Missing profiles should be represented as missing evidence, not silently
replaced by the current environment.

## Full-22 adoption

The Full-22 discrepancy tournament can adopt the profile without changing the
scientific cells:

1. set deterministic execution controls at workflow scope;
2. install from a hashed lock or immutable container image;
3. capture one profile after installation and before candidate execution;
4. copy the profile into every cell's immutable evidence directory, or bind the
   shared artifact and its `profile_id` in every cell manifest;
5. make the aggregator require one identical expected profile identity for all
   cells that are intended to be numerically matched;
6. reject selective-rerun cache entries whose profile identity differs;
7. preserve the profile beside the final decision artifact and paper inputs.

Candidate isolation may intentionally produce different package inventories. In
that case, preregister the expected profile per candidate and compare paired cells
only under their registered identities rather than forcing one profile across all
candidates.

## Scope and limitations

The profile records material execution inputs; it does not prove bitwise
determinism. Hardware behavior, nondeterministic kernels, process scheduling, and
third-party libraries can still affect results. Workflows should continue to use
fixed seeds, deterministic algorithms where available, repeated runs where
scientifically appropriate, and result-level diagnostics.

A profile mismatch is a provenance failure. A profile match is evidence that the
recorded numerical environment agrees, not a substitute for validating the
scientific computation.
