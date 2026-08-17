# Canonical external material-backend registry v1

## Purpose

BayesianPhysTwin has two version-1 transport contracts for external deformable
simulators:

- the Lagrangian export contract introduced for JAX-FEM and an early Genesis
  bridge; and
- the material-trajectory contract used by Warp FEM, SOFA, Genesis, the
  PositionBasedDynamics XPBD/PBD family, PhysX deformables, and MuJoCo Flex.

Both transports produce the existing six-array `physical_rollout_v1` archive.
They differ only in producer/runtime metadata and artifact custody. New engines
must therefore extend one canonical registry rather than create a third public
artifact family.

The canonical Python entry point is:

```python
from bayesian_phystwin.material_backend_v1 import (
    describe_material_backend_profiles,
    materialize_material_backend,
    validate_material_backend,
)
```

The transport-specific `lagrangian_backend_v1` and
`material_trajectory_backend_v1` modules remain available so that existing
content-addressed artifacts keep their exact interpretation.

For engine-facing JAX-FEM execution, `jax_fem_producer_v1` records matched fresh
driven and zero-action solve sequences directly into the existing Lagrangian
transport. It accepts total nodal displacements from one fixed reference mesh,
synchronizes JAX-like values before capture, and binds its own source digest
alongside the caller's exact wrapper, scene, mesh, and environment artifacts.
See [`jax_fem_producer_v1.md`](jax_fem_producer_v1.md) for the replay protocol,
callback wrapper, provenance requirements, and qualification boundary.

For engine-facing material-trajectory execution,
`material_trajectory_producer_v1` records matched fresh driven and zero-action
simulations directly into the existing material-trajectory transport. It is
dependency-free and currently targets Warp FEM, SOFA FEM, and
PositionBasedDynamics XPBD/PBD first. See
[`material_trajectory_producer_v1.md`](material_trajectory_producer_v1.md) for
the replay protocol, callback wrapper, provenance requirements, and
engine-specific integration rules. Both producers are experimental execution
surfaces; neither creates a third artifact contract or changes Prob4D/Causal4D
consumers.

## Canonical families

| Priority | Canonical family | Engine | Status | Producer profile IDs |
| --- | --- | --- | --- | --- |
| 1 | `jax-fem-quasistatic-v1` | JAX-FEM | preferred | `jax-fem-quasistatic-v1` |
| 2 | `warp-fem-v1` | NVIDIA Warp FEM | supported | `warp-fem-v1` |
| 3 | `sofa-fem-v1` | SOFA | supported | `sofa-fem-v1` |
| 4 | `genesis-mpm-v1` | Genesis World | supported | `genesis-mpm-v1`, legacy `genesis-world-mpm-v1` |
| 5 | `position-based-dynamics-v1` | PositionBasedDynamics XPBD/PBD | supported | `position-based-dynamics-v1` |
| 6 | `physx-fem-v1` | NVIDIA PhysX deformables | experimental | `physx-fem-v1` |
| 7 | `mujoco-flex-v1` | MuJoCo Flex | experimental | `mujoco-flex-v1` |

Each family must have exactly one non-legacy default variant. Canonical
resolution therefore cannot depend on tuple or insertion order. Additional
legacy variants are allowed only for immutable artifact compatibility. Priority
ties are ordered deterministically by canonical family ID.

The duplicate Genesis identifiers are one canonical family. New materialization
uses `genesis-mpm-v1` and the material-trajectory transport. The earlier
`genesis-world-mpm-v1` identifier remains a legacy transport variant so frozen
artifacts and source revisions are not rewritten.

Warp FEM is the lowest-friction GPU-differentiable FEM extension because the
project already uses Warp-based physical paths while the new profile preserves
fixed FEM-node identity. PositionBasedDynamics adds a deliberately different
XPBD/PBD rope, rod, cloth, and soft-body baseline. PhysX adds a high-throughput
GPU FEM/contact reference, but remains experimental until a standalone producer
has demonstrated deterministic access to the simulation-mesh vertex ordering
and complete runtime provenance.

The ranking records implementation priority only. It is not evidence that one
engine has better physical fidelity. A backend advances scientifically only
through a separately frozen competence experiment with incumbent-relative
prediction, numerical-floor, sensitivity, calibration, runtime, failure, and
guarded-update endpoints.

## Command line

The existing experiment route is retained to avoid command churn, but it now
dispatches every registered material backend:

```bash
bpt experiment run materialize-lagrangian-backend profiles

bpt experiment run materialize-lagrangian-backend materialize \
  raw-rollout.npz runtime.json output/backend \
  --profile warp-fem-v1

bpt experiment run materialize-lagrangian-backend validate output/backend
```

`--profile` is an optional assertion. The runtime manifest remains authoritative
and selects the transport through exactly one of:

- `backend_profile` for `lagrangian-export-v1`; or
- `backend_kind` for `material-trajectory-v1`.

A canonical family assertion accepts any registered transport variant of that
family. A retained transport-specific producer ID is an exact assertion. For
example, requesting `genesis-world-mpm-v1` cannot silently accept a
`genesis-mpm-v1` material-trajectory runtime. Materialization also fails when
the requested family, runtime profile, and runtime schema disagree. Validation
fails when a directory contains zero or multiple recognized artifact manifests.

`python -m bayesian_phystwin.cli.material_trajectory_backend` remains a thin
compatibility shim for the same canonical CLI. It is not a second extension
point.

## Extension rule

A new backend contribution must:

1. add one canonical family or one variant of an existing family;
2. declare exactly one non-legacy default variant per family;
3. use one of the two admitted transport contracts;
4. preserve persistent material identity and the common physical-rollout map;
5. bind exact engine, source, runtime, units, frame, and information-boundary
   identities; and
6. keep compatibility evidence separate from accuracy, calibration, physical
   benefit, intervention benefit, safety, and state-of-the-art claims.

Do not introduce another top-level backend contract merely because an engine
uses a different solver family. Transport differences belong in a registered
variant, while solver and identity differences belong in profile metadata.
