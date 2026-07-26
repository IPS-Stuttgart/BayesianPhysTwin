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
validity, residual lifting, and the diagnostic operations currently consumed
by Causal4D. These functions deliberately delegate to BPT's internal
implementations so those implementations can move without changing the
cross-repository import path.

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
verify every facade name imported by Causal4D, and prevent new imports from
underscore-prefixed BPT modules or functions.
