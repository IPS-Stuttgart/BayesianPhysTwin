# Causal4D graph provider API v1

`bayesian_phystwin.causal4d_graph_provider_v1` is the supported lightweight
surface for graph construction and released controller grouping used by
Causal4D.

It exists separately from `causal4d_provider_v1` so that graph and controller
geometry can be imported without loading Torch, Warp, the official PhysTwin
checkout, or experiment-specific analysis modules.

## Public operations

The module exposes:

- `PhysTwinSpringGraph` and `PhysTwinSpringGraphConfig`;
- `build_phystwin_spring_graph()`;
- `controller_hand_count()`;
- `infer_controller_groups()`; and
- `causal4d_graph_provider_manifest()`.

Causal4D production code must import these names from the versioned provider
module rather than from `phystwin_graph` or
`phystwin_controller_sensitivity` directly. This lets the underlying graph and
experiment modules move while preserving the cross-repository contract.

## Manifest and compatibility

The graph-provider manifest records the Bayesian-PhysTwin package version,
optional exact Git revision, API version, and two explicit capabilities:

- `phystwin_spring_graph`;
- `controller_grouping`.

It also declares `PhysTwinSpringGraph` artifact schema version 1. Consumers must
validate both the required capabilities and this schema version before using the
provider in an upgradeable environment.

Frozen experiments continue to bind an exact Bayesian-PhysTwin revision.
Upgradeable development environments may use the package version and graph API
version, but must fail closed when the required provider module, capability, or
artifact schema is absent.

## Dependency boundary

The module is NumPy-only. Constructing a graph does not initialize PhysTwin or
Warp. Simulator construction and replay remain owned by
`bayesian_phystwin.causal4d_provider_v1` and its `PhysTwinReplayProvider`
protocol.

Graph construction preserves PhysTwin's object-then-controller spring ordering,
float32 radius-neighbor semantics, rest lengths, masses, and explicit object
point boundary. Controller grouping preserves the released one- or two-hand
case convention and deterministic spatial partition.

## Versioning policy

Backward-compatible operations may be added to this module during the BPT 0.4
line. Removing or changing graph ordering, controller grouping, array units, or
return types requires a new versioned provider module.
