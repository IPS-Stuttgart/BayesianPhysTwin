# FEniCSx and PyElastica backend profiles v1

This extension adds two **experimental** material-backend families without
introducing another downstream artifact schema:

| Priority | Profile | Engine family | Persistent material identity |
| ---: | --- | --- | --- |
| 9 | `fenicsx-fem-v1` | FEniCSx / DOLFINx distributed variational FEM | registered global geometry-node index |
| 10 | `pyelastica-cosserat-rod-v1` | PyElastica Cosserat-rod dynamics | native rod-node index |

Drake deformable FEM remains priority 8. The priority numbers describe the
recommended implementation order; they are not evidence of physical accuracy.
Both new profiles use the existing `material-trajectory-v1` transport and
therefore publish the same six-array `physical_rollout_v1` archive already used
by BayesianPhysTwin, Prob4D, and Causal4D integration paths.

## Why these two families

The existing portfolio already represents differentiable and GPU FEM, classical
FEM, MPM, XPBD/PBD, PhysX deformables, MuJoCo Flex, and Drake deformables.
FEniCSx and PyElastica add distinct model classes rather than another wrapper for
an already represented solver:

- **FEniCSx / DOLFINx** provides an independent distributed variational-FEM
  reference. It is useful for constitutive-law, weak-form, discretization, and
  nonlinear-solver cross-checks against the GPU and robotics-oriented FEM paths.
- **PyElastica** provides a dedicated Cosserat-rod hypothesis for rope, cable,
  catheter, and other slender-body cases. This is especially relevant to the
  rope-first public-data and real-object evaluations.

The extension deliberately stops at executable compatibility and provenance.
Neither profile is promoted to `supported`, and neither is assumed to outperform
the incumbent bank.

## Dependency boundary

The base package does not import or vendor DOLFINx, MPI, PETSc, or PyElastica.
A producer-side environment constructs a fresh native simulator for each replay
arm and passes a dependency-free adapter to
`produce_material_trajectory_backend`.

The adapters live in:

```python
from bayesian_phystwin.material_trajectory_replay_adapters_v1 import (
    DolfinxDisplacementReplayV1,
    PyElasticaRodReplayV1,
)
```

## FEniCSx / DOLFINx adapter

`DolfinxDisplacementReplayV1` accepts a registered reference geometry and a
callback returning the current displacement in exactly the same global node
order:

```python
replay = DolfinxDisplacementReplayV1(
    reference_positions_m=global_reference_geometry_m,
    displacement_callback=scene.global_displacement_m,
    step_callback=scene.advance_one_output_step,
    synchronize_callback=scene.synchronize,
    context=scene,
)
```

The reference and displacement arrays must both have shape `(S, 3)`, share one
floating dtype, and contain only finite values. The adapter copies arrays into
contiguous host memory and publishes `reference + displacement`.

A DOLFINx wrapper must establish material identity **before** constructing the
adapter. Raw rank-local rows, ghost degrees of freedom, local degree-of-freedom
ordering, and repartition-dependent concatenations are not a valid global
material order. The producer manifest must bind at least:

- the exact DOLFINx, PETSc, MPI, and Python environment revisions;
- mesh, geometry, form, boundary-condition, and material source hashes;
- the global geometry-node map and its content digest;
- scalar type, partition, nonlinear and linear solver configuration;
- time integrator, step size, tolerances, and synchronization policy; and
- every external model or asset used to construct the scene.

The profile was reviewed against DOLFINx `v0.11.0.post0`, commit
`fefdb2201b80a8f59527de2d461b9056906661d8`. Native execution remains external
to BayesianPhysTwin.

## PyElastica adapter

`PyElasticaRodReplayV1` reads the native `rod.position_collection` matrix and
converts PyElastica's `(3, N)` node layout to contiguous `(N, 3)` material rows:

```python
replay = PyElasticaRodReplayV1(
    rod=scene.rod,
    step_callback=scene.advance_one_output_step,
    synchronize_callback=scene.synchronize,
    context=scene,
)
```

The rod must retain at least two persistent nodes. The matrix must be floating
and finite. Node insertion, deletion, remeshing, rod replacement, or a change in
assembly ordering invalidates the v1 identity contract.

The producer manifest must bind at least:

- the exact PyElastica and Python environment revisions;
- rod count, assembly ordering, node count, and connectivity;
- rest geometry, radii, density, constitutive and damping parameters;
- constraints, joints, contacts, friction, and external forcing;
- time-stepper, substeps, step size, and callback timing; and
- source hashes for the scene, controls, and any external assets.

The profile was reviewed against PyElastica `v1.0.0`, commit
`b087f1399f9be2fdd2fcf3768689f7735a96f7ab`. Native execution remains external
to BayesianPhysTwin.

## Fresh replay and information boundary

Both profiles inherit the common producer rules:

1. `replay_factory` creates two distinct simulator instances from identical
   registered inputs.
2. Frame zero is captured before either control callback runs.
3. The driven control is applied immediately before each driven step.
4. The zero-action callback is applied immediately before each reference step.
5. State is synchronized before every capture.
6. Driven and zero-action frame-zero arrays must be byte-identical in shape,
   dtype, values, and material order.
7. Query indices are fixed at frame zero and cannot use future observations or
   outcomes.
8. Publication is deterministic and content-addressed.

The material trajectory is projected into the existing physical-rollout fields:

- `prediction_m`;
- `persistence_m`;
- `driven_readout_m`;
- `zero_action_readout_m`;
- `action_support`; and
- `frame_zero_points_m`.

Prob4D and Causal4D therefore need no engine-specific branch. They continue to
consume the versioned observation and physical-rollout contracts and exact
cross-repository provenance.

## Promotion gate

Compatibility is not backend competence. Before either profile can move from
`experimental` to `supported`, a pinned native reference scene should provide:

- deterministic repeated publication under the registered runtime;
- a numerical-floor test against an independent reference or refinement study;
- parameter sensitivity in the physically expected directions;
- matched driven and zero-action replay integrity;
- runtime, memory, and failure-rate evidence;
- source-only calibration with no target-outcome tuning;
- guarded non-harm relative to the incumbent physical fallback; and
- exact fallback identity when production or admission fails.

The next scientific step should qualify one rope-oriented PyElastica scene and
one independent FEniCSx solid scene. Additional backend families should be added
only after a locked experiment identifies a model-class gap not covered by the
current bank.

## Upstream references and license boundary

- FEniCSx / DOLFINx: <https://github.com/FEniCS/dolfinx>
- PyElastica: <https://github.com/GazzolaLab/PyElastica>

Both are separately installed external runtimes. Their source, binaries,
examples, meshes, assets, and generated outputs are not bundled or relicensed by
BayesianPhysTwin. See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)
for the recorded distribution boundary.
