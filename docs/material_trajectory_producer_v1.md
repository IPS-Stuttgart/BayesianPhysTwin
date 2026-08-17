# Fresh-replay material trajectory producer v1

## Purpose

`material_trajectory_producer_v1` turns an external simulator execution into the
existing strict `material-trajectory-v1` bundle without importing that simulator
into BayesianPhysTwin. It is intended for the highest-value material backends
whose state can be exposed as one persistent ordered set of material points:

1. NVIDIA Warp FEM;
2. SOFA FEM; and
3. PositionBasedDynamics XPBD/PBD.

The same producer can be used for Genesis MPM, MuJoCo Flex, or a future PhysX
adapter when their wrappers satisfy the same identity contract. PhysX remains
experimental: a wrapper must expose the deformable **simulation-mesh** vertex
order, not render or collision vertices, before its output is scientifically
admissible.

The producer adds no engine dependency and introduces no new downstream
artifact family. It creates the four-array raw material trajectory and strict
runtime manifest, then delegates to the existing materializer. BayesianPhysTwin,
Prob4D, and Causal4D continue to consume the unchanged six-array
`physical_rollout_v1` contract.

## Replay contract

A producer-side wrapper implements three methods:

```python
class MaterialTrajectoryReplayV1(Protocol):
    def synchronize(self) -> object: ...
    def get_material_positions_m(self) -> object: ...
    def step(self) -> object: ...
```

The wrapper must obey all of these rules:

- `get_material_positions_m()` returns floating `(S, 3)` positions in metres;
- positions are already transformed into `right-handed-z-up-world-v1`;
- state index `s` identifies the same material node or particle in every frame;
- the shape, dtype, topology, and ordering remain fixed for the complete replay;
- `synchronize()` makes pending engine work visible before host capture; and
- `step()` advances exactly one registered output interval.

CPU engines should implement `synchronize()` as a no-op. Warp wrappers should
normally call the relevant Warp synchronization operation. A wrapper around an
asynchronous tensor may additionally return an object exposing
`block_until_ready`, `detach`, `cpu`, or `numpy`; the producer copies the final
value into contiguous host memory.

For small integrations, `CallbackMaterialTrajectoryReplayV1` builds the wrapper
from three zero-argument callbacks:

```python
from bayesian_phystwin.material_trajectory_producer_v1 import (
    CallbackMaterialTrajectoryReplayV1,
)

replay = CallbackMaterialTrajectoryReplayV1(
    synchronize_callback=synchronize_engine,
    positions_callback=read_fixed_material_positions_m,
    step_callback=advance_one_output_step,
    context=engine_scene,
)
```

## Fresh driven and zero-action arms

`produce_material_trajectory_backend` calls `replay_factory` exactly twice. Each
call must construct a fresh scene and wrapper. The first instance is driven by
`driven_control`; the second is driven by `zero_action_control`.

The execution order is fixed:

1. synchronize and capture frame zero;
2. invoke the arm-specific control for transition `k`;
3. advance one output step;
4. synchronize and capture frame `k + 1`; and
5. repeat until the registered frame count is reached.

The producer rejects reused wrapper objects, different frame-zero states,
changed state shape or dtype, non-finite positions, duplicate or out-of-range
queries, and invalid action support. Exact frame-zero equality is intentional:
seed, sampling, discretization, material ordering, initial solver state, and
initial boundary conditions must match between the two arms.

The object-level freshness guarantee cannot detect two distinct Python wrappers
that alias the same hidden engine state. The producer therefore also records the
protocol as an attestation, and source qualification must test repeated fresh
executions.

## Publication example

```python
from bayesian_phystwin.material_trajectory_producer_v1 import (
    CallbackMaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)


def replay_factory() -> CallbackMaterialTrajectoryReplayV1:
    scene = build_fresh_scene()
    return CallbackMaterialTrajectoryReplayV1(
        synchronize_callback=scene.synchronize,
        positions_callback=scene.fixed_material_positions_m,
        step_callback=scene.step,
        context=scene,
    )


def driven_control(k, replay):
    replay.context.apply_registered_action(k)


def zero_action_control(k, replay):
    del k
    replay.context.apply_zero_action()


artifact = produce_material_trajectory_backend(
    output_dir="output/warp-fem-v1",
    backend_kind="warp-fem-v1",
    replay_factory=replay_factory,
    driven_control=driven_control,
    zero_action_control=zero_action_control,
    frame_count=76,
    material_query_indices=query_indices,
    action_support=action_support,
    engine_revision="<exact 40- or 64-character revision>",
    engine_version="<installed engine version>",
    producer_repository="owner/backend-producer",
    producer_revision="<exact producer revision>",
    producer_version="producer-v1",
    producer_artifacts={
        "producer.py": "<sha256>",
        "scene.json": "<sha256>",
    },
    topology_sha256="<sha256 of fixed topology and ordering>",
    device="cuda",
    device_name="<exact device name>",
    time_step_s=1.0 / 120.0,
    scene_id="registered-source-case-v1",
    model_kind="deformable-solid",
    constitutive_model="neo-hookean",
    integrator="implicit-euler",
    solver="registered-native-solver",
    substeps=4,
    engine_parameters={
        "density_kg_m3": 1000.0,
        "young_modulus_pa": 500000.0,
        "poisson_ratio": 0.35,
    },
)
```

The result is the standard bundle:

```text
material-trajectory-backend.json
physical-prediction.npz
SHA256SUMS
provenance/material-trajectory-rollout.npz
provenance/material-runtime.json
```

The runtime manifest additionally binds:

- the exact producer repository, revision, version, and source-file hashes;
- the topology digest;
- the frame-zero state and query-index digests;
- synchronization before every capture;
- two independently constructed replay instances;
- control-before-step action timing; and
- the `fresh-replay-control-before-step-v1` execution protocol.

The standard validator recomputes checksums, reloads the no-pickle archive,
rederives every physical member, and checks the exact file roster.

## Engine-specific wrappers

### Warp FEM

Use one fixed FEM reference domain and node order. The positions callback should
read the state array corresponding to those FEM nodes, not visualization
vertices. The synchronization callback must complete pending Warp work before
the host copy. Bind basis, quadrature, constitutive law, integrator, contact
settings, topology bytes, and the exact Warp revision.

Warp FEM is the first implementation target because it stays close to the
existing Warp ecosystem while adding strain-native constitutive hypotheses. It
should nevertheless be compared against the existing PhysTwin/Warp spring path,
not treated as its automatic replacement.

### SOFA FEM

Read the registered `MechanicalObject` state in its fixed node order. Do not
substitute mapped visual or collision coordinates. Construct fresh driven and
zero-action simulations, and bind the topology container, mappings, force
fields, contact pipeline, linear/nonlinear solvers, time integration, and scene
source hashes. A synchronous CPU SOFA wrapper may use a no-op synchronization
callback.

SOFA is the preferred independent reference because its solver and contact
stack differs materially from the existing Warp paths.

### PositionBasedDynamics XPBD/PBD

Read particles in the fixed `SimulationModel` order and bind the complete
constraint graph. Rebuild fresh models for the two arms; do not reset a model
that retains multipliers, warm starts, or hidden constraint state. Bind solver
iterations, substeps, compliance/stiffness parameters, collision handling, and
all topology and constraint assets.

XPBD/PBD is a deliberately different, inexpensive hypothesis family for rope,
rod, cloth, and compliant-surface cases. Its value is complementary model error,
not constitutive fidelity by assumption.

## Scientific boundary

A successfully published bundle establishes execution order, fixed material
identity, units, coordinate frame, source custody, deterministic portable
materialization, and compatibility with the shared downstream contract. It does
not establish:

- physical fidelity or material-parameter identification;
- gradient correctness;
- calibrated predictive uncertainty;
- fresh-object or fresh-session transfer;
- Prob4D calibration benefit;
- Causal4D intervention benefit;
- deployment safety; or
- state of the art.

Promotion requires the common source-only material-backend competence protocol:
incumbent-relative future error, numerical floor, parameter sensitivity,
non-degenerate spread, grouped proper-score calibration, runtime and failure
accounting, guarded non-harm, and byte-exact fallback.
