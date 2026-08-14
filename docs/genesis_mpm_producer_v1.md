# Genesis MPM producer v1

## Purpose

`bayesian_phystwin.genesis_mpm_producer_v1` turns two fresh Genesis MPM
replays into the strict external-physics archive and runtime manifest used by
Bayesian-PhysTwin. The module imports only NumPy and Bayesian-PhysTwin. It does
not import Genesis and does not add Genesis, Torch, or a GPU runtime to the base
package.

The producer uses the public Genesis surfaces needed at the boundary:

- `MPMEntity.get_particles_pos()` for persistent material-particle positions;
- `Scene.step()` for one simulator step; and
- an optional `env_index` for selecting exactly one environment from a batched
  scene.

Genesis currently returns unbatched particle positions as `(P,3)` and batched
positions as `(B,P,3)`. The producer accepts `(P,3)` directly, accepts a
singleton batch, or requests one exact environment. It rejects unresolved
multi-environment output.

## Replay contract

The caller supplies one `replay_factory`. It is invoked exactly twice: once for
the driven replay and once for the zero-action replay. Every invocation must
return a fresh, already-built `(scene, mpm_entity)` pair with identical initial
particle sampling, ordering, material parameters, and coordinate frame.

Frame zero is captured before a control callback is invoked. For transition
`k`, the corresponding callback runs and then `scene.step()` advances the
simulation. Therefore an archive with `T` frames contains the initial state and
`T-1` controlled transitions.

The producer fails closed when:

- the two fresh scenes do not have bit-identical frame-zero particles;
- particle count, order-facing shape, or dtype changes during a replay;
- a tensor is non-floating or contains non-finite values;
- a batched result is not reduced to one exact environment;
- query indices are empty, duplicated, non-integral, or out of range;
- action support is not a finite numeric vector in `[0,1]`; or
- the output path exists or traverses a symbolic link.

The output NPZ is deterministic, no-pickle, timestamp independent,
content-hashable, and published without clobbering an existing artifact.

## Minimal integration skeleton

Initialize Genesis once in the producer repository, then provide a factory and
both control arms:

```python
from pathlib import Path

import genesis as gs

from bayesian_phystwin.genesis_mpm_producer_v1 import (
    produce_genesis_mpm_backend,
)


gs.init(backend=gs.gpu)
known_targets = [...]  # fixed before target outcomes are opened


def replay_factory():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, substeps=20),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.5),
            grid_density=64,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    actuator = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.55),
            size=(0.12, 0.12, 0.05),
            fixed=False,
        ),
        name="actuator",
    )
    deformable = scene.add_entity(
        material=gs.materials.MPM.Elastic(E=5e4, nu=0.3, rho=1000),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.35),
            size=(0.15, 0.15, 0.15),
        ),
        name="deformable",
    )
    scene.build()
    # Add any contact/attachment constraints here. The same factory is used for
    # both arms, so the initial scene and particle order remain identical.
    del actuator
    return scene, deformable


def driven_control(k, scene, deformable):
    del deformable
    actuator = scene.get_entity(name="actuator")
    actuator.set_qpos(known_targets[k])


def zero_action_control(k, scene, deformable):
    del k, deformable
    actuator = scene.get_entity(name="actuator")
    actuator.set_qpos(ZERO_ACTION_QPOS)


result = produce_genesis_mpm_backend(
    raw_rollout_path=Path("raw-rollout.npz"),
    runtime_manifest_path=Path("runtime.json"),
    replay_factory=replay_factory,
    driven_control=driven_control,
    zero_action_control=zero_action_control,
    frame_count=len(known_targets) + 1,
    query_entity_indices=QUERY_PARTICLE_INDICES,
    action_support=QUERY_ACTION_SUPPORT,
    engine_revision=EXACT_GENESIS_GIT_REVISION,
    engine_version=gs.__version__,
    producer_repository="owner/genesis-producer",
    producer_revision=EXACT_PRODUCER_GIT_REVISION,
    coordinate_frame="right-handed-z-up-world-v1",
    time_step_s=0.002,
    topology_sha256=PARTICLE_SAMPLING_AND_SCENE_TOPOLOGY_SHA256,
    material_model="genesis-mpm-elastic",
    observation_end_frame_exclusive=OBSERVATION_END_FRAME,
    parameterization={
        "young_modulus_pa": 50000.0,
        "poisson_ratio": 0.3,
        "density_kg_m3": 1000.0,
        "substeps": 20,
        "grid_density": 64,
    },
    producer_artifacts={
        "configs/scene.json": SCENE_CONFIG_SHA256,
        "assets/deformable-source.mesh": SOURCE_MESH_SHA256,
    },
)
```

The example intentionally leaves query-particle correspondence and known
control construction in the producer repository. Those choices are
experiment-specific and must be frozen using frame-zero/source information,
not future observations or target outcomes.

## Batched scenes

For a Genesis scene with multiple environments, pass `env_index=<integer>`.
The producer calls `get_particles_pos(envs_idx=[env_index])` and then requires a
single returned batch. Run separate producer calls for separate environments so
that each artifact has one unambiguous particle identity and one causal action
history.

## Provenance requirements

Bind at least the following in the runtime manifest and producer artifacts:

- the exact Genesis source revision and reported package version;
- the exact producer repository revision;
- the source mesh or particle-sampling input digest;
- the scene/configuration digest, coordinate frame, and simulation time step;
- MPM material parameters, grid density, substeps, contact, and attachments;
- the frame-zero method used to select persistent query particles; and
- the source-only method used to define `action_support`.

`topology_sha256` should bind the complete particle-sampling and scene-topology
specification, not only the particle count. The runtime separately binds the
ordered frame-zero particle positions through `entity_identity_sha256`.

## Downstream materialization

After production, the generic external-backend commands remain unchanged:

```bash
python -m bayesian_phystwin.cli.external_physics_backend materialize \
  raw-rollout.npz runtime.json genesis-bundle

python -m bayesian_phystwin.cli.external_physics_backend validate \
  genesis-bundle
```

A successful bundle proves deterministic custody and contract conformance. It
does not establish that Genesis improves a target protocol. Advancement still
requires the source-only comparison, calibration, frozen guard, and exact
incumbent fallback described in `external_physics_backends_v1.md`.
