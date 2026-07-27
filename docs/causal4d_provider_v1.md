# Causal4D provider API v1

`bayesian_phystwin.causal4d_provider_v1` is the supported integration surface
for Causal4D. It centralizes the implementation-private dependencies that were
previously imported directly by the downstream repository.

## Manifest and artifacts

`causal4d_provider_manifest()` reports:

- provider package version and optional exact Git revision;
- provider API/schema version 1;
- declared execution and artifact capabilities;
- `TwinBelief` and `GraphBelief` artifact schema version 1.

The module also exposes stable names for artifact loading and hashing, target
validity, residual lifting, the immutable fixed Bayesian-anchor endpoint, and
the diagnostic operations currently consumed by Causal4D. Diagnostic symbols
are resolved through an explicit lazy registry, so importing the provider or
rendering CLI help does not eagerly load every historical experiment module.
The registry is a closed compatibility surface rather than a generic helper
namespace.

## Trusted legacy artifacts

Released PhysTwin inputs still include pickle files that cannot be migrated
retroactively. New code should access them through the narrower
`bayesian_phystwin.causal4d_artifacts_v1` boundary and
`load_trusted_legacy_phystwin_pickle()`.

The loader:

- requires a lowercase SHA-256 digest obtained from an independently trusted
  run manifest, protocol lock, or release inventory;
- verifies the digest before invoking pickle deserialization;
- requires an explicit top-level artifact kind (`mapping`, `sequence`, or
  `ndarray`);
- can require named keys for mapping artifacts; and
- rejects changed bytes and incompatible top-level representations.

Digest verification does not sandbox pickle. A caller must never trust a digest
provided alongside an otherwise untrusted pickle. The digest establishes byte
identity only when its source is already trusted. All newly generated
cross-repository artifacts remain JSON/NPZ-only.

The artifact module also exposes an immutable released raw-track map. It
requires a trusted digest for the processed `final_data.pkl` before delegating
to the historical correspondence implementation. This keeps legacy pickle
identity checks ahead of deserialization while preserving the exact released
track semantics.

The separate module keeps this compatibility exception out of the normal
replay protocol and can be retired independently when legacy PhysTwin inputs no
longer require pickle.

## Execution protocol

`PhysTwinReplayProvider` is a runtime-checkable protocol with a narrow surface:

- `set_group_log_scales()`;
- `set_controller_points()`;
- `replay_initial()`;
- `replay_restart()`;
- `close()`.

`create_official_replay_provider()` returns
`OfficialPhysTwinReplayProvider`, which owns the Torch/Warp simulator details
and resource cleanup. Causal4D should not access the wrapped simulator,
Torch, or Warp objects directly in normal execution code.

## Versioning policy

BPT 0.4.x provides `causal4d_provider_v1`; Causal4D accepts
`bayesian-phystwin>=0.4,<0.5` for normal development. Backward-compatible
changes may be added to this module in 0.4.x. Removing or changing a required
operation needs a new provider module/API version and a new BPT compatibility
minor line.

Frozen experiments continue to record and install exact Git revisions. The
version range is for upgradeable development environments; it does not replace
experiment locks.

Both repositories contain cross-repository tests. They validate the manifest,
verify endpoint parity and immutable arrays, exercise every versioned facade,
and require Causal4D source and scripts to import only the explicit provider,
graph-provider, public-study-provider, and legacy-artifact modules.
