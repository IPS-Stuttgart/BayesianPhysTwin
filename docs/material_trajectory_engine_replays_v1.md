# SOFA, MuJoCo, PBD, Genesis MPM, and Warp FEM replay adapters v1

## Purpose

`material_trajectory_engine_replays_v1` closes concrete engine-integration gaps
without changing the portable material-trajectory contract or adding heavy
runtime dependencies to BayesianPhysTwin.

It provides dependency-free replay shims for:

- **SOFA FEM** through a Vec3 `MechanicalObject.position.value` state and one
  registered `Sofa.Simulation.animate(root, dt)` step;
- **MuJoCo Flex** through the exact per-flex slice of `data.flexvert_xpos`, using
  `model.flex_vertadr` and `model.flex_vertnum`, with one registered
  `mujoco.mj_step(model, data)` step;
- **PositionBasedDynamics / XPBD** through pyPBD
  `SimulationModel.getParticles().getVertices()` and one registered
  `TimeStep.step(simulation_model)` step; and
- **Genesis MPM** through one `MPMEntity.get_state()` particle roster and a
  registered `Scene.step()` output interval; and
- **Warp FEM** through a degree-1 FEM `DiscreteField.dof_values` displacement
  roster plus the frozen reference positions in the exact same node order.

The module imports none of those simulator packages. Callers construct native
engine objects and inject them into the structural adapters. The resulting
objects satisfy `MaterialTrajectoryReplayV1` and can be passed directly to
`produce_material_trajectory_backend`.

This is **adapter coverage**, not backend qualification. It does not change the
backend portfolio evidence stages, promote any adapter to an active
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
`(N, 3)` coordinates, returns an owning contiguous copy, and advances exactly
the registered `time_step_s`. CPU SOFA normally needs no explicit
synchronization; a caller can provide `synchronize_callback` when its integration
requires one.

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
non-integer slice metadata, invalid flex IDs, empty or negative slices,
non-finite or non-floating positions, and slices that exceed the runtime data
array.

As with SOFA, the selected positions must already have the physical units and
coordinate frame required by the portable contract. Do not replace Flex
vertices with render mesh, skin, or contact-only coordinates.

## PositionBasedDynamics / XPBD

The registered PBD identity is the global simulation-particle order. At the
pinned upstream revision, pyPBD exposes that state directly through
`SimulationModel.getParticles()` and `ParticleData.getVertices()` and exposes
the native time-step surface as `TimeStep.step(SimulationModel)`.

```python
from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    PositionBasedDynamicsReplayV1,
)


def build_replay():
    model, time_step = build_fresh_pypbd_scene()
    return PositionBasedDynamicsReplayV1(
        simulation_model=model,
        time_step=time_step,
        context=model,
    )
```

The adapter records the complete global `ParticleData` roster rather than one
render mesh or one constraint-local subset. `getVertices()` must expose finite
floating `(N, 3)` rows in the registered metre/z-up convention. The common
producer rejects particle insertion, deletion, or reordering because every frame
must retain the frame-zero shape and identity.

A qualifying PBD runtime must additionally bind the exact simulation-model
construction, complete constraint graph, solver parameters, collision setup,
assets, substeps, time step, and pyPBD/PositionBasedDynamics revision. The
adapter itself does not infer those semantics from particle coordinates.

## Genesis MPM entity state

Genesis exposes the dynamic state of one MPM entity through
`MPMEntity.get_state()`. The public state carries particle positions and active
membership with shapes `(B,N,3)` and `(B,N)`. Select one environment explicitly:

```python
from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    GenesisMPMEntityReplayV1,
)


def build_replay():
    scene, deformable = build_fresh_genesis_mpm_scene()
    return GenesisMPMEntityReplayV1(
        entity=deformable,
        environment_index=0,
        step_callback=scene.step,
        synchronize_callback=synchronize_genesis_device,
        context=scene,
    )
```

The adapter detaches and moves tensor-like state to the host before converting
it to NumPy. At construction it freezes the selected environment's complete
particle count and active mask. Every later capture must preserve both exactly;
particle insertion, deletion, activation, or deactivation fails closed rather
than changing the registered material identity. Only initially active particles
are published, in their original entity-local row order.

The selected positions must already be metres in
`right-handed-z-up-world-v1`. A qualifying producer must additionally bind the
entity construction, particle sampling and order, constitutive model, grid and
boundary configuration, rigid/MPM coupling, time step and substeps, device,
Genesis revision, and the exact scene-step wrapper. The adapter deliberately
does not reach into private solver fields or use renderer particles.

## Warp FEM displacement fields

`WarpFEMDisplacementReplayV1` addresses the common Warp FEM representation in
which a vector-valued `DiscreteField` stores **displacements** at finite-element
degrees of freedom rather than absolute material positions.

```python
from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    WarpFEMDisplacementReplayV1,
)


def build_replay():
    scene = build_fresh_warp_fem_scene()
    return WarpFEMDisplacementReplayV1(
        displacement_field=scene.displacement_field,
        reference_positions_m=scene.reference_node_positions_m,
        step_callback=scene.step_one_output_interval,
        synchronize_callback=scene.synchronize,
        context=scene,
    )
```

The adapter is intentionally narrow:

- `displacement_field.degree` must be exactly one;
- `displacement_field.dof_values` must expose Warp's `numpy()` host-transfer
  surface;
- `reference_positions_m` is copied and frozen at construction;
- the reference and displacement rosters must both be finite floating `(N, 3)`
  arrays and retain the same shape; and
- each capture returns
  `reference_positions_m + displacement_field.dof_values`.

Degree one is required because `warp-fem-v1` registers persistent FEM
**mesh-node** identity. Higher-order fields introduce additional interpolation
DOFs whose rows are not identical to a simple mesh-node roster. A higher-order
profile would need an explicit, separately reviewed identity contract rather
than silently reusing `warp-fem-v1`.

The reference roster must correspond to the exact space partition and row order
of `dof_values`. For a qualifying runtime, bind the basis/function-space
construction, partition, topology, quadrature, constitutive law, contact setup,
integrator/solver, step/substep policy, device/stream identity, and exact Warp
revision. The adapter cannot infer those facts from the arrays alone.

The caller should still provide an explicit synchronization callback for the
stream/device used by the solver. Warp's native array `numpy()` path performs a
host transfer, but the replay contract keeps synchronization explicit so custom
stream ownership and producer provenance remain auditable.

## Portable producer integration

The adapters deliberately stop at the replay boundary. Publication remains the
existing producer flow:

```python
from bayesian_phystwin.material_trajectory_producer_v1 import (
    produce_material_trajectory_backend,
)

artifact = produce_material_trajectory_backend(
    output_dir=output_dir,
    backend_kind="genesis-mpm-v1",
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

## Why these adapters

SOFA exposes a standardized material-state and animation surface and is a useful
independent FEM reference. MuJoCo exposes a stable, explicit Flex-vertex state
and offers a fast controls-oriented deformable baseline. pyPBD exposes the
global particle array used by its rope, rod, cloth, and soft-body models.
Genesis exposes entity-local MPM state without requiring BayesianPhysTwin to
import its runtime, making the active MPM qualification candidate executable
through the same fixed-identity producer contract.
Warp's FEM field API provides an especially direct path from a displacement DOF
roster to the registered mesh-node material contract without importing Warp into
the base package.

PhysX remains on an explicit producer-side boundary for now: its authoritative
deformable-volume simulation positions are GPU device buffers and require
engine/CUDA-specific synchronization and host-copy ownership. Wrapping that copy
in a nominally dependency-free class would not remove the actual integration
obligation.

## Scientific next step

The portfolio policy still makes JAX-FEM and Genesis MPM the active
qualification candidates. These SOFA, MuJoCo, PBD, Genesis, and Warp shims
improve implementation coverage while those two candidates remain
source-physics-qualified but below source value after their first frozen value
arms failed. Another backend should enter the active qualification funnel only
after a slot opens or materially stronger source-side evidence justifies
replacing a current candidate.
