# External material-trajectory backends v1

## Scope

Bayesian-PhysTwin already contains the official PhysTwin/Warp spring path, the
MatPhys spring-proposal adapter, and an optional Newton implicit-MPM backend.
This extension uses one strict fixed-material trajectory boundary for six
complementary simulator families:

| profile | canonical engine | material identity | intended role |
| --- | --- | --- | --- |
| `warp-fem-v1` | `NVIDIA/warp` | FEM mesh-node index | lowest-friction GPU-differentiable FEM path |
| `sofa-fem-v1` | `sofa-framework/sofa` | mechanical node index | independent mature FEM/multiphysics reference |
| `genesis-mpm-v1` | `Genesis-Embodied-AI/genesis-world` | material particle index | contact-rich MPM and differentiable-MPM experiments |
| `position-based-dynamics-v1` | `InteractiveComputerGraphics/PositionBasedDynamics` | simulation particle index | fast XPBD/PBD rope, rod, cloth, and soft-body baseline |
| `physx-fem-v1` | `NVIDIA-Omniverse/PhysX` | deformable simulation-mesh vertex index | high-throughput GPU FEM/contact reference |
| `mujoco-flex-v1` | `google-deepmind/mujoco` | flex vertex index | lightweight control-oriented compatibility path |

The engines remain optional and run in their own pinned environments. The base
`bayesian_phystwin` package does not import or vendor them. An engine producer
exports fixed material trajectories; the adapter validates custody and emits
the same six-array physical archive already consumed by Bayesian-PhysTwin.
Prob4D and Causal4D therefore do not need engine-specific branches.

This is a compatibility extension, not evidence that any new engine improves
held-out prediction. In particular, the PhysX and MuJoCo Flex profiles remain
experimental until their exact producer/runtime combinations pass the common
source-only replay, calibration, and non-harm gates.

## Why these profiles

The selected engines add complementary mechanisms without duplicating the
already admitted spring and Newton-MPM paths:

1. **Warp FEM** is the most practical next producer. It adds strain- and
   constitutive-model-based FEM while sharing the GPU/autodiff ecosystem already
   used by existing Warp paths.
2. **SOFA FEM** remains the strongest independent FEM and multiphysics reference
   for contact-rich soft-robotics cases.
3. **Genesis MPM** provides a broad particle-based large-deformation and contact
   family. It complements the pinned Newton smoke rather than replacing it.
4. **PositionBasedDynamics XPBD/PBD** provides a fast, deliberately different
   constraint-based baseline for rope, rod, cloth, and soft-body hypotheses.
5. **PhysX deformables** provide a high-throughput GPU FEM/contact reference, but
   the producer must expose simulation-mesh state without depending on render or
   collision-mesh ordering.
6. **MuJoCo Flex** remains a compact controls-oriented path for rapid ablations.

Isaac Lab, Drake, Brax, PyBullet, RaiSim, NimblePhysics, Taichi-only MPM, and
LAMMPS are deliberately deferred. They either wrap an already represented
solver, focus primarily on rigid/control workflows, require a more specialized
material readout, or offer a weaker fit to the current fixed-material-query
contract. They can be added later as another exact profile without changing the
portable physical archive.

## Raw trajectory archive

The producer writes a deterministic, no-pickle NPZ with exactly four arrays:

- `driven_material_positions_m`: floating `(T,S,3)` positions;
- `zero_action_material_positions_m`: the matched zero-action trajectory;
- `material_query_indices`: unique persistent state indices of shape `(N,)`;
- `action_support`: one residual-independent value in `[0,1]` per query.

`T >= 2`, `S >= 1`, and both trajectories must share an exact frame-zero state.
The producer must transform positions into metres in
`right-handed-z-up-world-v1` before publication. Query index `n` must denote the
same FEM node, MPM particle, simulation particle, deformable simulation vertex,
or flex vertex at every frame. Producers with remeshing, particle birth/death,
or topology-changing identities are not admissible under v1.

## Engine-specific producer rules

The common archive deliberately excludes engine-native topology and solver
state, but each producer must bind enough provenance to make its material-index
interpretation auditable:

- **Warp FEM:** retain one fixed reference domain and FEM node ordering for both
  driven and zero-action scenes. Synchronize the Warp device before copying each
  frame and bind the topology, quadrature/basis choice, integrator, constitutive
  law, and exact Warp revision in the runtime metadata.
- **SOFA FEM:** bind the `MechanicalObject` node order, topology container,
  mappings, force fields, contact pipeline, and solver configuration. Mapped or
  visual nodes may not silently replace the registered mechanical identities.
- **Genesis MPM:** retain material-particle identities with no birth, death, or
  reordering. Driven and zero-action trajectories must come from fresh matched
  scenes rather than from a reset that retains hidden solver state.
- **PositionBasedDynamics:** bind the `SimulationModel` particle order and the
  complete constraint graph. Driven and zero-action trajectories must use fresh
  matched scenes; a producer may not select particles from future residuals.
- **PhysX FEM:** export vertices from the deformable simulation mesh, not the
  collision or render mesh. Bind cooked mesh identities, simulation-buffer
  ordering, GPU runtime, and copy synchronization. The profile stays
  experimental until those guarantees are demonstrated by a native producer.
- **MuJoCo Flex:** bind the flex vertex order, element topology, contact options,
  and exact model bytes. Any accelerated runtime must bind both MuJoCo and
  accelerator revisions.

## Runtime manifest

The adjacent JSON manifest is content addressed and binds:

- the exact engine repository, revision, version, producer version, Python
  version, device, and time step;
- the profile's solver family and material-identity kind;
- scene, model, constitutive model, integrator, solver, substeps, and finite
  engine parameters;
- frame, state, and query counts;
- the raw archive SHA-256; and
- a future-blind information boundary.

The exact profile facts are discoverable with:

```bash
python -m bayesian_phystwin.cli.material_trajectory_backend profiles
```

A producer should build its manifest from the profile returned by
`get_material_backend_profile`, use an exact 40- or 64-character lowercase
engine revision, compute `runtime_id` with `_portable_contracts.content_id`, and
publish it with `_portable_contracts.write_atomic_json`.

## Materialization and validation

Adapt one producer export:

```bash
python -m bayesian_phystwin.cli.material_trajectory_backend materialize \
  material-trajectory-rollout.npz \
  material-runtime.json \
  /path/to/backend-bundle
```

Validate the complete bundle and rederive every physical member:

```bash
python -m bayesian_phystwin.cli.material_trajectory_backend validate \
  /path/to/backend-bundle
```

The self-contained output contains:

```text
material-trajectory-backend.json
physical-prediction.npz
SHA256SUMS
provenance/material-trajectory-rollout.npz
provenance/material-runtime.json
```

Validation fails closed on changed engine/profile facts, units, coordinate
frame, information boundary, counts, source bytes, file roster, query identity,
non-finite values, frame-zero drift, physical arrays, or checksums. The output
is deterministic for identical source bytes and runtime metadata.

## Downstream contract

The adapter maps persistent engine state to:

- `prediction_m`;
- exact `persistence_m`;
- `driven_readout_m`;
- `zero_action_readout_m`;
- `action_support`; and
- `frame_zero_points_m`.

That is the existing `physical_rollout_v1` interface. Backend-specific native
state stays in the producer artifact. Bayesian inference, Prob4D observation
beliefs and covariance structure, and Causal4D interventions continue to depend
on the portable material-query trajectories rather than on an engine import.

## Advancement gate

Treat all profiles as compatibility producers. Promote one exact runtime to a
claim-bearing backend only after a source-only protocol establishes:

1. exact frame-zero alignment in metres and in the registered coordinate frame;
2. stable material identities throughout the full rollout;
3. repeated driven and zero-action replays that quantify the numerical floor;
4. a disjoint-prefix improvement or complementary error mode relative to the
   incumbent physical bank;
5. non-degenerate predictive spread from a parameter or model ensemble;
6. Prob4D calibration and grouped reliability checks on unopened groups;
7. Causal4D intervention consistency without future observations or outcomes;
8. an exact incumbent fallback for failed production or source gates.

Recommended implementation order: Warp FEM first, SOFA second, Genesis third,
PositionBasedDynamics fourth, PhysX fifth, and MuJoCo Flex sixth. Keep Newton MPM
as the pinned MPM reference while Genesis matures, and do not replace the
official PhysTwin/Warp path until matched prospective evidence supports that
decision.

## Upstream references

- NVIDIA Warp: <https://github.com/NVIDIA/warp>
- SOFA: <https://github.com/sofa-framework/sofa>
- Genesis: <https://github.com/Genesis-Embodied-AI/genesis-world>
- PositionBasedDynamics: <https://github.com/InteractiveComputerGraphics/PositionBasedDynamics>
- NVIDIA PhysX: <https://github.com/NVIDIA-Omniverse/PhysX>
- MuJoCo: <https://github.com/google-deepmind/mujoco>
- MuJoCo Warp acceleration: <https://github.com/google-deepmind/mujoco_warp>
