# External physics backend profiles v1

## Purpose

Bayesian-PhysTwin uses one simulator-neutral adapter for external deformable
physics engines. Engine-specific code runs in an opt-in producer environment and
exports persistent entity trajectories plus a strict runtime manifest. The core
adapter then projects stable entity indices into the existing six-array
`physical_rollout_v1` contract.

No engine runtime or heavy dependency is imported into the base package. The
boundary also preserves the division of responsibility between repositories:
Prob4D owns observation uncertainty, Bayesian-PhysTwin evaluates and guards a
physical candidate, and Causal4D consumes only the selected belief.

## Prioritized built-in profiles

The built-ins cover complementary mechanics, inference, and deployment roles:

1. `genesis-mpm-v1` — broad differentiable MPM and coupled contact;
2. `jax-fem-v1` — differentiable FEM material identification and ensembles;
3. `warp-fem-v1` — GPU-differentiable custom constitutive and contact work;
4. `physx-fem-v1` — high-throughput GPU surface/volume deformable reference;
5. `sofa-fem-v1` — mature contact-rich FEM and soft-robotics reference;
6. `mujoco-flex-v1` — fast controls-oriented flexible-body baseline;
7. `position-based-dynamics-v1` — rope, rod, cloth, and soft-body XPBD/PBD
   baseline; and
8. `drake-fem-v1` — robotics systems-and-controls integration candidate.

The order is a development priority, not an empirical ranking. A profile is a
versioned compatibility descriptor, not evidence that an engine improves a
target task. Exact engine and producer revisions are bound per runtime artifact.

The dedicated Newton MPM compatibility path remains separate while it is
migrated to the generic producer boundary. Additional research systems should
enter through the plugin interface until they expose a stable producer with
persistent entity identity and exact provenance.

## Implemented engine-facing producers

### Genesis MPM

`bayesian_phystwin.genesis_mpm_producer_v1` consumes a factory for fresh,
already-built Genesis scenes and their MPM entities. It uses only the public
`get_particles_pos()` and `scene.step()` surfaces. The producer:

- creates independent driven and zero-action replays;
- captures frame zero before either action callback;
- supports exact selection of one environment from a batched scene;
- requires bit-identical initial particle positions and order;
- rejects particle shape, dtype, identity, or finite-value drift; and
- writes the deterministic raw archive and bound runtime manifest.

The module itself does not import Genesis. See `genesis_mpm_producer_v1.md` for
an integration skeleton, batching rules, and provenance requirements.

### JAX-FEM

`bayesian_phystwin.jax_fem_producer_v1` consumes a factory for fresh JAX-FEM
replay wrappers. A wrapper returns fixed reference mesh nodes and executes one
load or time-step solve. The selected solver field must be a floating `(N,3)`
nodal displacement relative to the fixed reference points. The producer:

- supports the usual JAX-FEM list of solution fields or one direct array;
- synchronizes JAX-like arrays through `block_until_ready()` when available;
- calls each control callback immediately before the corresponding solve;
- requires independent driven and zero-action solve sequences;
- rejects reference-mesh drift, invalid field selection, and displacement
  shape, dtype, or finite-value errors; and
- writes the same generic raw archive and runtime manifest as every backend.

The module itself does not import JAX or JAX-FEM. Native 2-D problems must be
explicitly embedded into a frozen 3-D coordinate frame by the producer. See
`jax_fem_producer_v1.md` for quasi-static, transient, multi-field, and
provenance guidance.

Both producers establish the software and custody path only. Neither backend is
claim-bearing until its exact runtime passes source-only qualification and the
separate advancement gate below.

## Raw producer archive

A producer writes one no-pickle NPZ with exactly four members:

- `driven_entity_positions_m`: floating `(T,E,3)` positions;
- `zero_action_entity_positions_m`: the same shape, dtype, and exact frame-zero
  entity order;
- `query_entity_indices`: a unique integer vector selecting persistent entities;
- `action_support`: one floating value in `[0,1]` per selected entity.

`T` must be at least two. Positions are in metres in a producer-declared frame.
The same entity index must denote the same material particle, mesh node,
mechanical state, flex vertex, or PBD particle at every frame. Remeshing or
resampling is admissible only after the producer establishes and records a
stable material correspondence.

The adapter derives:

- `prediction_m` from the driven trajectory;
- exact frame-zero `persistence_m`;
- `driven_readout_m` from the driven trajectory;
- `zero_action_readout_m` from the independent zero-action replay;
- unchanged `action_support`; and
- `frame_zero_points_m`.

## Runtime manifest and bundle

List the built-in profiles:

```bash
python -m bayesian_phystwin.cli.external_physics_backend profiles
```

Build a content-addressed runtime manifest after an external producer has
written its raw archive:

```bash
python -m bayesian_phystwin.cli.external_physics_backend runtime \
  genesis-mpm-v1 raw-rollout.npz runtime.json \
  --engine-revision <exact-40-or-64-hex-revision> \
  --engine-version <reported-runtime-version> \
  --producer-repository owner/integration-repository \
  --producer-revision <exact-40-or-64-hex-revision> \
  --coordinate-frame right-handed-z-up-world-v1 \
  --time-step-s 0.008333333333333333 \
  --topology-sha256 <sha256-of-mesh-particles-or-scene-topology> \
  --material-model neo-hookean \
  --observation-end-frame-exclusive 12 \
  --parameterization-json parameters.json \
  --producer-artifact configs/scene.json=<sha256>
```

Materialize and independently validate the self-contained bundle:

```bash
python -m bayesian_phystwin.cli.external_physics_backend materialize \
  raw-rollout.npz runtime.json output-directory

python -m bayesian_phystwin.cli.external_physics_backend validate \
  output-directory
```

The bundle contains the copied raw archive, copied runtime manifest,
deterministic `physical-prediction.npz`, a content-addressed artifact manifest,
and `SHA256SUMS`. Validation rederives every physical array byte for byte and
rejects changed file rosters, digests, counts, units, entity order, query order,
frame-zero identity, information boundaries, or profile descriptors.

The runtime contract requires:

- exact engine and producer source revisions;
- the selected self-contained profile descriptor;
- topology and ordered frame-zero entity digests;
- finite JSON material and solver parameters;
- explicit coordinate, position, and time units;
- an explicit causal observation cutoff;
- no future observations, target outcomes, or outcome-based backend selection;
- an independently simulated zero-action replay; and
- the known action bound to the driven replay.

## Source-only backend qualification

A valid runtime bundle is not automatically a qualified physical model.
`PhysicsBackendQualificationV1` binds one exact candidate runtime and one exact
incumbent runtime to a frozen source-only protocol, independent source groups,
measured diagnostics, thresholds, and fallback evidence.

A passing record requires:

- valid units, coordinate frame, persistent entity order, and query identity;
- byte-reproducible reruns;
- zero-action equilibrium drift below a frozen threshold;
- rigid-transform equivariance error below a frozen threshold;
- time-step-refinement sensitivity below a frozen threshold;
- unchanged topology and entity identity;
- zero declared physical-sanity violations;
- finite-difference Jacobian agreement below a frozen relative-error threshold;
- source-query parity to the registered incumbent below a frozen RMSE threshold;
- byte-identical exact fallback for unsupported or rejected cases;
- a protocol frozen before source outcomes were inspected; and
- no target outcome use.

The record is content-addressed and retains all failure reasons. A failed record
remains useful source evidence but cannot authorize the runtime:

```python
from bayesian_phystwin.physics_backend_qualification_v1 import (
    load_physics_backend_qualification_v1,
    require_qualified_backend_runtime,
)
from bayesian_phystwin.physics_backend_registry_v1 import profile_from_mapping

qualification = load_physics_backend_qualification_v1(
    "physics-backend-qualification.json"
)
require_qualified_backend_runtime(
    profile_from_mapping(runtime_manifest["backend_profile"]),
    runtime_manifest["runtime_id"],
    qualification,
)
```

Qualification is still source-side mechanism evidence. Independent-object
accuracy, calibrated uncertainty, and downstream Causal4D benefit require their
own frozen confirmation protocols.

## Third-party profiles

Additional engines can register a profile without modifying
Bayesian-PhysTwin. Entry-point loading is opt-in because it imports third-party
Python code:

```toml
[project.entry-points."bayesian_phystwin.physics_backends.v1"]
my-backend = "my_package.backend_profile:PROFILE"
```

`PROFILE` may be a `PhysicsBackendProfileV1`, its exact JSON mapping, a sequence
of profiles, or a zero-argument callable returning one of those forms. Resolve
plugin profiles with `--include-plugins`. Plugin identifiers cannot override a
built-in, and the selected descriptor is embedded in the runtime so later
validation does not depend on the plugin remaining installed.

## Advancement gate

Before a qualified runtime can replace or join the incumbent on a target
protocol, require all of the following on already-open source development data:

1. a passing `PhysicsBackendQualificationV1` bound to the exact runtime;
2. a fixed parameter prior or source-only posterior with non-degenerate spread;
3. held-out source-prefix improvement or demonstrable complementarity to the
   incumbent under the same actions and metrics;
4. calibration checks using proper multivariate scores, not mean trajectory
   error alone;
5. a frozen guard whose rejection path is a byte-identical incumbent artifact;
6. no target outcome or future observation in fitting, backend choice, or
   acceptance; and
7. a separately hashed prediction before independent future outcomes are
   opened.

Genesis MPM and JAX-FEM now have concrete producer boundaries. The next
claim-relevant step is to run both on one already-open Deform360 or PokeFlex
source object, generate qualification records, and compare them under the same
frozen query/action and incumbent contract. Warp FEM should follow when custom
GPU-differentiable mechanics are the bottleneck; PhysX should follow when
operational contact throughput is the bottleneck. SOFA, MuJoCo,
PositionBasedDynamics, and Drake remain complementary reference paths.
