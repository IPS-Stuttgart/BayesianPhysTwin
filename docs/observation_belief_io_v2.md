# Bounded and atomic ObservationBelief I/O v2

`ObservationBeliefV1` remains the portable observation contract and retains its
existing content identity. The explicit module
`bayesian_phystwin.observation_belief_io_v2` adds a stricter execution boundary
for archives received from another process, workflow, host, or repository. The
module is deliberately not re-exported from the package root, so callers opt in
to the v2 resource and durability semantics explicitly.

## Reading

```python
from bayesian_phystwin.observation_belief_io_v2 import (
    ObservationBeliefIOLimitsV2,
    load_observation_belief_bounded_v2,
)

belief = load_observation_belief_bounded_v2(
    "observation-belief.npz",
    limits=ObservationBeliefIOLimitsV2(
        maximum_archive_bytes=512 * 1024 * 1024,
        maximum_uncompressed_bytes=2 * 1024 * 1024 * 1024,
        maximum_observation_count=5_000_000,
        maximum_factor_rank=4096,
    ),
)
```

Before NumPy loads an array, the v2 reader:

- opens an ordinary regular file without following a symbolic link where the
  operating system supports `O_NOFOLLOW`;
- bounds the archive, individual members, total uncompressed bytes, descriptor,
  NPY headers, observation rows, declared frames, groups, and factor rank;
- requires the exact closed ZIP-member roster;
- rejects duplicate, encrypted, directory, object-dtype, trailing-byte, or
  unsupported-compression members;
- checks every NPY shape and exact dtype from its bounded header before array
  allocation;
- parses the descriptor as duplicate-free finite JSON with a closed field set;
- reconstructs `ObservationBeliefV1` and verifies its content address; and
- rejects a file whose descriptor, size, inode, or timestamps change during the
  read.

The defaults are intentionally finite. A registered workflow that requires a
larger valid artifact must raise the corresponding budget explicitly rather
than disabling validation.

## Writing

```python
from bayesian_phystwin.observation_belief_io_v2 import (
    save_observation_belief_atomic_v2,
)

save_observation_belief_atomic_v2("observation-belief.npz", belief)
```

The writer serializes into a same-filesystem temporary file, flushes and syncs
it, reloads it through the bounded v2 reader, checks the unchanged artifact
identity, and only then performs `os.replace`. The previous authoritative target
therefore remains intact after serialization or verification failure. The
parent directory is synced where the platform supports it.

This is additive. Existing `save_observation_belief` and
`load_observation_belief` remain available for frozen reproductions. New
cross-process and cross-repository workflows should prefer the v2 boundary.

## Scientific boundary

A valid v2 read proves archive integrity, resource-bounded decoding, schema
consistency, and content identity. It does not establish provider competence,
covariance calibration, physical-state identifiability, downstream prediction
benefit, deployment safety, or state of the art.
