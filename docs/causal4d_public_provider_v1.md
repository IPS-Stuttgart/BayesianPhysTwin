# Causal4D public-data provider API v1

`bayesian_phystwin.causal4d_public_provider_v1` is the narrow versioned surface
for Causal4D public-data studies that still reuse source-locked
Bayesian-PhysTwin experiment semantics.

The module advertises two capability groups:

- `deform360_selective_virtual_sensing` for the frozen source-window,
  artifact, protocol, and scoring operations used by the Deform360 study;
- `phystwin_track_objective` for the released reliability-aware track objective
  used by public Warp feasibility runners.

All symbols are resolved from an explicit lazy registry. Merely importing the
provider does not load the underlying public-data modules or optional runtime
stacks. Adding a new symbol requires a reviewed registry entry, a manifest
capability, and a cross-repository use case; this module is not a generic
re-export namespace.

The public-study provider is distinct from replay provider v1, immutable replay
provider v2, the graph provider, and the hash-locked legacy-artifact boundary.
Frozen experiments still record exact Git revisions in addition to the provider
manifest.
