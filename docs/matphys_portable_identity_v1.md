# Portable MatPhys identity v1

The official MatPhys producer-v1 bundle records exact local source paths. That
is useful for same-host verification, but an absolute path is not a portable
scientific identity: the same checkpoint and replay bytes mounted at two
different locations otherwise receive different content identifiers.

`bayesian_phystwin.matphys_portable_identity_v1` adds a certificate without
changing the frozen producer-v1 contract. It separates two identities:

- `portable_artifact_id` hashes repositories, revisions, model and parameter
  digests, frame and information boundaries, deterministic output digests, and
  the path-independent causal proposal;
- `host_status_id` additionally hashes the exact local source bundle and source
  file paths used for optional same-host reverification.

The portable certificate contains no checkpoint, spring-field, replay, or
parameter absolute path. Moving the certificate bundle therefore preserves its
scientific identity. Moving or remounting source files changes only the
host-local receipt.

## Build

Start from a validated official producer-v1 bundle:

```python
from bayesian_phystwin.matphys_portable_identity_v1 import (
    materialize_matphys_portable_identity,
)

result = materialize_matphys_portable_identity(
    "outputs/matphys-official-producer-v1",
    "outputs/matphys-portable-identity-v1",
)
print(result["portable_artifact"]["portable_artifact_id"])
print(result["source_verification"]["host_status_id"])
```

The output contains exactly:

```text
matphys-portable-identity.json
source-verification.json
SHA256SUMS
```

The checksum manifest covers the portable certificate. The source receipt is
self-addressed by `host_status_id`; changing a path or digest invalidates that
receipt without changing the definition of the scientific certificate.

## Validate

Portable custody and all path-independent semantics:

```python
from bayesian_phystwin.matphys_portable_identity_v1 import (
    validate_matphys_portable_identity,
)

validate_matphys_portable_identity(
    "outputs/matphys-portable-identity-v1"
)
```

Also rehash every local source and rederive the certificate from the original
producer bundle:

```python
validate_matphys_portable_identity(
    "outputs/matphys-portable-identity-v1",
    verify_sources=True,
)
```

The second form is intentionally host-specific. It fails when a source path is
missing, traverses a symlink, has changed bytes, or no longer derives the same
portable certificate.

## Causal proposal

A causal producer bundle contains a portable proposal derived from the existing
MatPhys proposal. Its identity retains:

- exact MatPhys and PhysTwin revisions;
- target-excluded training identities;
- the prefix/future boundary;
- checkpoint and spring-field digests;
- proposal strength and source-artifact lineage; and
- the existing claim boundary.

Absolute checkpoint and spring-field locations are omitted. Published per-case
parity bundles contain no causal proposal and remain explicitly ineligible for a
causal or deployment claim.

## Shared custody primitives

The internal module `bayesian_phystwin._artifact_custody` centralizes regular
file and directory validation, no-overwrite staging, byte-exact copying, exact
file rosters, and canonical checksum manifests. Scientific schemas continue to
own their own fields and inference semantics; only low-level custody behavior is
shared.

## Evidence boundary

A portable certificate establishes path-independent identity and reproducible
custody for an already generated official MatPhys producer artifact. It does not
establish official paper parity, physical accuracy, calibration, unseen-object
transfer, deployment safety, Causal4D value, or state of the art. Native source
qualification and the registered guarded value experiment remain separate
requirements.
