# Act--sense--fallback physical-twin certificate v1

This controlled experiment is the first implementation step toward a physical
twin that decides whether to execute a physical action, gather
decision-relevant information, or restore a caller-owned fallback without first
identifying one latent state.

## Mechanism

The finite hypothesis bank contains an occluded rope with:

- two possible tether sides;
- two friction levels; and
- three visually distinguishable but action-irrelevant texture states.

The terminal actions are `pull_left`, `pull_right`, and `hold`. Two registered
diagnostic probes are available:

- `tug_side`, a two-outcome probe that reveals the loss-relevant tether side;
- `camera_texture`, a three-outcome probe that reveals more state entropy but
  no action-relevant information.

Every complete outcome-contingent action map is enumerated. Probe cost is added
to the terminal loss, and the existing exact quotient certificate is applied to
the expanded plan set.

## Results

The deterministic checked result covers three cases.

1. **Act:** six latent states remain possible, but all require `pull_left`.
   The direct action has zero worst-case regret and no probe is used.
2. **Sense:** all twelve states remain possible. The higher-information texture
   camera removes `ln(3)` nats, while the side tug removes only `ln(2)` nats.
   Nevertheless, the exact decision certificate selects the side tug because
   its best contingent plan has worst-case regret `0.20`, versus `1.55` for the
   camera.
3. **Fallback:** with tolerance reduced from `0.25` to `0.10`, no direct or
   contingent plan is admissible. The returned action is exactly the registered
   `hold` fallback.

Run:

```bash
python -m experiments.act_sense_fallback_occluded_rope_v1.run --check
```

## Boundary

This is a controlled finite-hypothesis mechanism, not real robot execution.
Probe outcomes are deterministic and their costs and loss matrix are registered
inputs. The result does not validate a perception provider, resettable physical
probing, target-domain transport, calibration, deployment, or safety.

The next empirical gate is a source-frozen block-level action-choice study on
the complete Tracking Cloth self-collision factorial. Its three release
configurations are real physical interventions repeated for every material and
repetition. That study must keep the fresh repetition numerically closed until
the action loss, probe semantics, and policy are frozen.
