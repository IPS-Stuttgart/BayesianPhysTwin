# SOFA and MuJoCo material replay adapters v1

## Purpose

`material_trajectory_engine_replays_v1` closes two concrete engine-integration
gaps without changing the portable material-trajectory contract or adding heavy
runtime dependencies to BayesianPhysTwin.

It provides dependency-free replay shims for:

- **SOFA FEM** through a Vec3 `MechanicalObject.position.value` state and one
  registered `Sofa.Simulation.animate(root, dt)` step; and
- **MuJoCo Flex** through the exact per-flex slice of `data.flexvert_xpos`, using
  `model.flex_vertadr` and `model.flex_vertnum`, with one registered
  `mujoco.mj_step(model, data)` step.

The module imports neither SOFA nor MuJoCo. Callers construct native engine
objects and inject the public step callback. The resulting replay objects satisfy
`MaterialTrajectoryReplayV1` structurally and can be passed directly to
`produce_material_trajectory_backend`.

This is **adapter coverage**, not backend qualification. It does not change the
backend portfolio evidence stages, promote SOFA or MuJoCo to an active
qualification slot, or make any source-value, calibration, target-transfer,
Prob4D-benefit, or Causal4D-benefit claim.

## SOFA FEM

Use the material FEM `MechanicalObject`, not mapped visualization or collision
coordinates:

```python
import Sofa

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    SofaMechanicalObjectReplayV1,
)


def build_replay():
    root, mechanical = build_fresh_sofa_scene()
    Sofa.Simulation.initRoot(root)
    return SofaMechanicalObjectReplayV1(
        mechanical_object=mechanical,
        root_node=root,
        animate_callback=Sofa.Simulation.animate,
        time_step_s=float(root.dt.value),
        context=root,
    )
```

The adapter reads `mechanical_object.position.value`, requires floating finite
`(N, 3)` coordinates, returns an owning contiguous copy, and advances exactly the
registered `time_step_s`. CPU SOFA normally needs no explicit synchronization;
a caller can provide `synchronize_callback` when its integration requires one.

The registered mechanical state must already be expressed in metres in
`right-handed-z-up-world-v1`. Unit conversion and frame conversion belong in the
engine-side scene/wrapper and must be frozen before qualification.

## MuJoCo Flex

Select one physical Flex by its model index:

```python
import mujoco

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    MuJoCoFlexReplayV1,
)


def build_replay():
    model, data = build_fresh_mujoco_model_and_data()
    mujoco.mj_forward(model, data)
    return MuJoCoFlexReplayV1(
        model=model,
        data=data,
        flex_id=REGISTERED_FLEX_ID,
        step_callback=mujoco.mj_step,
        context=data,
    )
```

MuJoCo stores all Flex vertices in the global `data.flexvert_xpos` array. The
adapter uses `model.flex_vertadr[flex_id]` and `model.flex_vertnum[flex_id]` to
preserve the exact registered vertex slice and order. It rejects missing or
non-integer slice metadata, invalid flex IDs, empty/negative slices, non-finite or
non-floating positions, and slices that exceed the runtime data array.

As with SOFA, the selected positions must already have the physical units and
coordinate frame required by the portable contract. Do not replace Flex vertices
with render mesh, skin, or contact-only coordinates.

## Portable producer integration

The adapters deliberately stop at the replay boundary. Publication remains the
existing producer flow:

```python
from bayesian_phystwin.material_trajectory_producer_v1 import (
    produce_material_trajectory_backend,
)

artifact = produce_material_trajectory_backend(
    output_dir=output_dir,
    backend_kind="sofa-fem-v1",  # or "mujoco-flex-v1"
    replay_factory=build_replay,
    driven_control=driven_control,
    zero_action_control=zero_action_control,
    # exact runtime/provenance/topology fields omitted here
    ...
)
```

`replay_factory` is still called exactly twice for independently constructed
driven and zero-action worlds. All existing fixed-identity, frame-zero equality,
provenance, source-custody, deterministic materialization, and exact-fallback
rules remain in force. The produced `physical-prediction.npz` is therefore the
same simulator-neutral six-array contract consumed by BayesianPhysTwin, Prob4D,
and Causal4D.

## Why these two adapters

SOFA exposes a standardized material-state and animation surface and is the most
useful independent FEM reference among the currently registered material
backends. MuJoCo likewise exposes a stable, explicit Flex-vertex state in its
model/data API and offers a fast controls-oriented deformable baseline. Both can
therefore be integrated with small structural shims rather than bespoke exporter
formats.

Warp remains better served by `CallbackMaterialTrajectoryReplayV1`: Warp is a
kernel framework and does not define one universal deformable-state object whose
semantics would justify a single canonical adapter. PositionBasedDynamics and
PhysX should receive specialized shims only when one exact native state surface,
persistent identity definition, and topology boundary are frozen.

## Scientific next step

The portfolio policy still makes JAX-FEM and Genesis MPM the active
qualification candidates. These SOFA/MuJoCo shims improve implementation
coverage while those two candidates advance from native execution to source
physics and matched source-value evidence. SOFA or MuJoCo should enter the active
qualification funnel only after a slot opens or materially stronger source-side
evidence justifies replacing a current candidate.
