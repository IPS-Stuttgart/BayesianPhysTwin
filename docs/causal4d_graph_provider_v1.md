# Causal4D graph provider API v1

`bayesian_phystwin.causal4d_graph_provider_v1` is the supported lightweight
surface for graph construction and released controller grouping used by
Causal4D.

It complements the immutable replay contract in `causal4d_provider_v2` so graph
and controller geometry can be imported without loading Torch, Warp, the
official PhysTwin checkout, or experiment-specific analysis modules. Provider
v1 remains available only for frozen compatibility paths.

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

## Provider-v2 relationship

The graph provider uses the canonical metadata helpers from
`bayesian_phystwin.contracts.provider`, inherits the package-version fallback
from `causal4d_provider_v2`, and declares the v2 provider as its parent API in
the manifest. It does not create another replay protocol or reinterpret the
immutable v2 replay requests and trajectories.

The replay and graph manifests are validated independently because they evolve
at different rates. A frozen run records both exact identities; an upgradeable
environment requires the expected parent-provider metadata as well as graph
capabilities and artifact schemas.

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
version, but must fail closed when the required provider module, parent API,
capability, or artifact schema is absent.

## Dependency boundary

The module is NumPy-only. Constructing a graph does not initialize PhysTwin or
Warp. New simulator construction and replay use
`bayesian_phystwin.causal4d_provider_v2` and its immutable
`PhysTwinReplayProviderV2` request/result contract. Historical provider-v1
execution remains unchanged for exact frozen revisions.

Graph construction preserves PhysTwin's object-then-controller spring ordering,
float32 radius-neighbor semantics, rest lengths, masses, and explicit object
point boundary. Controller grouping preserves the released one- or two-hand
case convention and deterministic spatial partition.

## Versioning policy

Backward-compatible operations may be added to this module during the BPT 0.4
line. Removing or changing graph ordering, controller grouping, array units,
return types, or the declared provider-v2 parent requires a new versioned graph
provider module.
