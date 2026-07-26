# Causal4D provider API v1

`bayesian_phystwin.causal4d_provider_v1` is the stable compatibility boundary
for downstream Causal4D code. Causal4D should not import underscore-prefixed
helpers from Bayesian-PhysTwin implementation modules.

The facade has three responsibilities:

1. expose a content-addressed provider manifest with explicit capabilities and
   artifact-schema versions;
2. define the `PhysTwinReplayProvider` protocol and validated
   `ReplayTrajectory` return type;
3. publish stable names for the existing graph, lifting, validity, simulator,
   replay, metric, and structural-diagnostic primitives.

Implementation code remains in its owning Bayesian-PhysTwin modules. The facade
resolves those functions lazily, so importing the contract does not import
Torch, Warp, OpenCV, or an official PhysTwin checkout.

## Provider manifest

```python
from bayesian_phystwin.causal4d_provider_v1 import provider_manifest

manifest = provider_manifest("<exact git revision>")
print(manifest.manifest_id)
print(manifest.capabilities)
print(manifest.artifact_schema_versions)
```

The base capabilities intentionally match Causal4D's
`PhysicalBeliefProviderManifest` contract:

- `artifact_checksums`;
- `particle_endpoint_position`;
- `particle_endpoint_velocity`;
- `physical_parameter_particles`.

Additional capabilities declare official-Warp replay, residual lifting, graph
construction, full-covariance observation updates, and the versioned
observation/discrepancy artifacts.

## Replay protocol

A provider implementation supplies:

```python
class PhysTwinReplayProvider:
    manifest: PhysicalBeliefProviderManifest

    def replay_initial(*, frame_count: int) -> ReplayTrajectory: ...
    def replay_restart(
        position_m,
        velocity_mps,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> ReplayTrajectory: ...
```

`ReplayTrajectory` requires matching finite `(T, N, 3)` position and velocity
arrays and makes them read-only. This lets Causal4D test a backend against the
protocol without depending on one simulator class.

## Compatibility policy

- Frozen experiments may continue to pin an exact Bayesian-PhysTwin commit.
- Normal development should validate the semantic provider manifest and
  required artifact versions in addition to the package version.
- Adding a capability is backward compatible.
- Renaming a facade symbol, changing artifact canonicalization, or changing
  replay semantics requires a new provider API module.
- Private implementation names may change without requiring Causal4D changes.

Print the installed manifest with:

```bash
bpt-provider-manifest --provider-revision "$(git rev-parse HEAD)"
```
