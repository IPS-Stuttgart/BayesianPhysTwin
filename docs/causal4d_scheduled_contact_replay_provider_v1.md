# Scheduled-contact replay provider v1

## Status

This is an additive, experimental contract under the public Causal4D provider-v2
surface. It defines the exact boundary required to turn a finite posterior over
joint contact schedules into continuously simulated physical trajectories.

The contract does **not** claim that the current official PhysTwin/Warp provider
implements dynamic scheduled contact. A runtime object must separately satisfy
`ScheduledContactReplayProviderV1`; the package-level provider manifest advertises
only `scheduled_contact_replay_contracts`, not a working scheduled-contact
backend.

No frozen Causal4D estimator, Bayesian-PhysTwin result, physical protocol, or
paper claim changes through this contract.

## Scientific purpose

A contact posterior is not physically meaningful merely because its regime paths
have valid probabilities. Every retained complete schedule must drive one
uninterrupted simulator execution. Independently simulated segments must not be
spliced, because contact switching changes velocity, internal stress, deformation,
and future force transmission.

The request therefore binds:

- the complete joint regime schedules and their Causal4D schedule identity;
- normalized path weights and retained prior mass;
- the physical endpoint position and velocity state;
- physical-parameter log scales and the complete controller trajectory;
- an explicit strictly increasing frame timebase;
- finite-area contact patches as weighted physical-state node sets;
- normal and tangential stiffness plus friction for every path, contact, and
  frame; and
- simulator-configuration, initial-state, request, contact, and path identities.

Sticking and slipping frames require at least one contact node with positive
weights summing to one. Inactive and detached frames apply no patch. Padding uses
node index `-1` with exactly zero weight. Repeated or out-of-range nodes are
rejected.

## Result boundary

`ScheduledContactReplayResultV1` contains:

- one complete position and velocity history per requested schedule;
- nonnegative conditional physical/discrepancy variance;
- the unchanged contact IDs, path IDs, regime paths, and timebase;
- the exact request content identity;
- provider name, version, and immutable revision; and
- a replay-result identity over all provenance and numerical output bytes.

`validate_scheduled_contact_replay_result()` rejects request, schedule, path,
configuration, state, regime, or timebase drift before a downstream rollout bank
is formed.

The initial version is deliberately all-or-nothing. A provider must either return
all requested finite trajectories or fail the call. It must not silently omit a
failed path and renormalize the remaining support. A future partial-publication
contract would need explicit per-path technical failures and composed retained
prior-mass accounting.

## Shapes

For `K` paths, `G` contacts, `T` frames, `N` physical nodes, and maximum patch
size `M`:

```text
regime_paths               (K, G, T)
controller_points_m        (T, C, 3)
position_m                 (N, 3)
velocity_mps               (N, 3)
frame_times_s              (T,)
contact_node_indices       (K, G, T, M)
contact_node_weights       (K, G, T, M)
normal_stiffness_npm       (K, G, T) after validation/broadcast
tangential_stiffness_npm   (K, G, T) after validation/broadcast
friction_coefficient       (K, G, T) after validation/broadcast
positions_m                (K, T, N, 3)
velocities_mps             (K, T, N, 3)
```

The conditional variance may be scalar or have shape `(N, 3)`, `(K, N, 3)`, or
`(K, T, N, 3)`. This matches Causal4D's multi-contact rollout-bank variance
boundary without forcing a provider to manufacture path- or frame-specific
uncertainty it does not own.

## Example provider

```python
from bayesian_phystwin.causal4d_provider_v2 import (
    ScheduledContactReplayProviderV1,
    ScheduledContactReplayRequestV1,
    ScheduledContactReplayResultV1,
)


class MyScheduledProvider:
    simulator_configuration_id = "phystwin-config-sha256"
    provider_revision = "40-hex-source-revision"

    def replay_scheduled_contacts(
        self,
        request: ScheduledContactReplayRequestV1,
    ) -> ScheduledContactReplayResultV1:
        positions, velocities, variance = run_continuous_schedule_bank(request)
        return ScheduledContactReplayResultV1.from_request(
            request,
            positions_m=positions,
            velocities_mps=velocities,
            conditional_variance_m2=variance,
            provider_name="my-phystwin-provider",
            provider_version="0.1.0",
            provider_revision=self.provider_revision,
        )

    def close(self) -> None:
        ...


assert isinstance(MyScheduledProvider(), ScheduledContactReplayProviderV1)
```

## Next implementation gate

The next provider implementation should add finite-area moving contact to one
continuous PhysTwin/Warp execution per schedule, including explicit stick, slip,
detach, release hysteresis, and force-transmission semantics. It should first be
evaluated on a locked source-only observed-reset competence panel against both
persistence and the no-contact diagnostic control. Contract validity is not
mechanics competence, calibration, transfer, or physical-experiment evidence.
