# Causal4D provider API v1

`bayesian_phystwin.causal4d_provider_v1` is the supported historical
compatibility surface for Causal4D. It centralizes the implementation-private
dependencies that were previously imported directly by the downstream
repository while preserving frozen scientific and diagnostic consumers.

New production integrations should use the narrower role-specific provider
facades documented alongside this module. Provider v1 is retained for
compatibility; it is not the destination for new capabilities.

## Manifest and artifacts

`causal4d_provider_manifest()` reports:

- provider package version and optional exact Git revision;
- provider API/schema version 1;
- declared execution and artifact capabilities;
- `TwinBelief` and `GraphBelief` artifact schema version 1.

The module also exposes stable names for artifact loading and hashing, target
validity, residual lifting, and the diagnostic operations currently consumed
by Causal4D. These functions deliberately delegate to BPT's internal
implementations so those implementations can move without changing the
cross-repository import path.

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

## Versioning and no-growth policy

BPT 0.4.x provides `causal4d_provider_v1`; Causal4D accepts
`bayesian-phystwin>=0.4,<0.5` for normal development. Existing operations may
receive bug fixes, security hardening, and internal delegation changes that
preserve their public meaning and behavior.

The aggregate provider-v1 and scientific-provider-v1 export inventories are
frozen no-growth surfaces. CI pins their exact canonical inventories. Adding a
new exported capability to either v1 facade is not a backward-compatible
maintenance change: it requires a role-specific versioned provider or a new API
version and an explicit downstream compatibility review.

Frozen experiments continue to record and install exact Git revisions. The
version range is for upgradeable development environments; it does not replace
experiment locks.

Both repositories contain cross-repository tests. They validate the manifest,
verify every facade name imported by Causal4D, prevent accidental export growth,
and prevent new imports from underscore-prefixed BPT modules or functions.
