# Causal4D provider API v2

`bayesian_phystwin.causal4d_provider_v2` is the typed replay and geometry
boundary for new Causal4D development. Provider API v1 remains available for
frozen experiments and existing consumers.

## Why v2 exists

Provider API v1 successfully removed direct downstream imports of
underscore-prefixed Bayesian-PhysTwin helpers, but its replay protocol still
used mutable sequencing:

1. set the spring parameters;
2. replace the controller trajectory;
3. invoke an initial or restart replay.

That sequence made a replay depend on hidden provider state. Restart replays
also returned positions without the matching velocity history, and bare arrays
carried no request, configuration, state, frame, or time-step identity.

Provider API v2 replaces that sequence with one immutable request and one
immutable result.

## Contracts

The replay contract types live in `bayesian_phystwin.contracts.replay` and are
re-exported by the provider module:

- `InitialReplayRequestV1` specifies the complete parameter and controller state
  for a replay from the released initial state;
- `RestartReplayRequestV1` additionally carries an explicit endpoint position
  and velocity state;
- `ReplayTrajectoryV1` returns position and velocity histories, frame IDs, the
  physical frame interval, and the request/configuration/state identifiers;
- `PhysTwinReplayProviderV2` exposes only `replay()` and `close()` plus immutable
  provider metadata.

Every array supplied to a request or result is copied into a contiguous NumPy
array and made read-only. The provider rejects a request whose
`simulator_configuration_id` does not match its fixed simulator configuration.
An initial-state request must also identify the released initial state owned by
the provider. A restart request may identify a particle-specific `TwinBelief`
endpoint state.

## Example

```python
from bayesian_phystwin.causal4d_provider_v2 import (
    InitialReplayRequestV1,
    create_official_replay_provider_v2,
)

provider = create_official_replay_provider_v2(
    official_repo,
    data,
    optimal,
    checkpoint_path,
    graph,
    num_surface_points=num_surface_points,
    original_count=original_count,
    dt=simulation_step_dt_s,
    num_substeps=num_substeps,
    self_collision=self_collision,
    simulator_configuration_id=configuration_id,
    released_initial_state_id=initial_state_id,
    spring_parameterization="grouped",
    device=device,
)
try:
    trajectory = provider.replay(
        InitialReplayRequestV1(
            request_id="theta-grid-0007",
            simulator_configuration_id=configuration_id,
            initial_state_id=initial_state_id,
            group_log_scales=particle_log_scales,
            controller_points_m=controller_points,
            frame_count=train_end_frame,
        )
    )
finally:
    provider.close()
```

`ReplayTrajectoryV1.dt_s` is the physical interval between returned frames,
computed as the simulator integration step `dt` multiplied by `num_substeps`.

## Owned core and compatibility boundary

The stable implementation is split by responsibility:

- `bayesian_phystwin.contracts` owns provider metadata and replay DTOs;
- `bayesian_phystwin.phystwin.artifacts` owns content hashing;
- `bayesian_phystwin.phystwin.geometry` owns target validity and residual
  lifting;
- `bayesian_phystwin.phystwin.replay` owns state extraction and initial/restart
  rollout execution.

`causal4d_provider_v1` now forwards these operations to the owned modules while
retaining its historical names and mutable protocol. The advanced diagnostic
wrappers, unchecked legacy pickle entry point, raw simulator initialization,
state extraction, and direct simulator-array mutation are deliberately absent
from v2. Released pickle compatibility belongs in the separate hash-locked
`bayesian_phystwin.causal4d_artifacts_v1` boundary; new cross-repository
artifacts remain JSON/NPZ.

The only remaining implementation-private replay seam is construction of the
released Warp simulator. It is isolated behind
`_initialize_official_simulator()` inside the owned replay module so the
constructor can be moved without changing either provider API.

## Migration from v1

| Provider v1 | Provider v2 |
|---|---|
| `set_group_log_scales(x)` | `request.group_log_scales=x` |
| `set_controller_points(u)` | `request.controller_points_m=u` |
| `replay_initial(frame_count=n)` | `replay(InitialReplayRequestV1(..., frame_count=n))` |
| `replay_restart(x, v, ...)` | `replay(RestartReplayRequestV1(..., position_m=x, velocity_mps=v))` |
| tuple or position array | `ReplayTrajectoryV1` with positions, velocities, frames, timing, and provenance |

Frozen runs may continue to pin provider v1 and an exact Git revision. New
Causal4D code should use v2 and record the provider manifest plus request and
trajectory identifiers in its run manifest.
