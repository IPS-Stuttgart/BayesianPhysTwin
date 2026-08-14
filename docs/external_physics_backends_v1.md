# External physics backend profiles v1

## Purpose

Bayesian-PhysTwin now has one simulator-neutral adapter for external deformable
physics engines. The adapter does not import an engine runtime and does not add
heavy dependencies to the base package. Instead, an engine-specific producer
exports persistent entity trajectories and a strict runtime manifest. The
adapter projects stable entity indices into the existing six-array
`physical_rollout_v1` contract.

This keeps the downstream boundary unchanged: Prob4D remains responsible for
observation uncertainty, Bayesian-PhysTwin evaluates and guards a physical
candidate, and Causal4D consumes only the selected belief. A new backend cannot
silently alter those responsibilities.

## Prioritized built-in profiles

The built-ins cover complementary mechanics, inference, and deployment roles
rather than aliases for the same spring model:

1. `genesis-mpm-v1` — broad differentiable MPM and coupled contact. This
   remains the first concrete producer target; it must preserve material-particle
   identity.
2. `jax-fem-v1` — differentiable FEM material identification and ensembles.
   Contact and time integration remain producer-declared.
3. `warp-fem-v1` — GPU-differentiable custom constitutive and contact
   experiments. Warp is a toolkit, so the producer must bind the exact
   formulation.
4. `physx-fem-v1` — high-throughput GPU surface/volume deformable reference.
   It has a GPU requirement and no assumed native parameter gradients.
5. `sofa-fem-v1` — mature contact-rich FEM and soft-robotics reference. The
   producer must bind the exact component/plugin graph and constitutive model.
6. `mujoco-flex-v1` — fast controls-oriented flexible-body/contact baseline.
   Fidelity is task-dependent and must pass the same source gate.
7. `position-based-dynamics-v1` — fast rope, rod, cloth, and soft-body
   XPBD/PBD baseline. Operational stability is not evidence of force accuracy.
8. `drake-fem-v1` — robotics systems-and-controls integration candidate.
   Deformable support is experimental and must be revision-pinned.

The profile order is a development priority, not an empirical ranking. None of
the eight is claim-bearing until it passes the source-only advancement gate
below. Exact engine and producer revisions are stored per runtime artifact, so
the profile catalog never substitutes a floating package version for
provenance.

The existing dedicated Newton MPM compatibility path remains separate while it
is migrated to this generic producer boundary. Research systems without a
stable, versioned producer should enter through the plugin mechanism rather
than expanding the core dependency surface.

## Raw producer archive

An external producer writes one no-pickle NPZ with exactly four members:

- `driven_entity_positions_m`: floating `(T,E,3)` positions;
- `zero_action_entity_positions_m`: the same shape, dtype, and exact frame-zero
  entity order;
- `query_entity_indices`: a unique integer vector selecting persistent entities;
- `action_support`: one floating value in `[0,1]` per selected entity.

`T` must be at least two. Positions are in metres in a producer-declared frame.
The same entity index must denote the same material particle, mesh node,
mechanical state, flex vertex, or PBD particle at every frame. Remeshing or
particle resampling is allowed only if the producer first establishes a stable
material correspondence and exports that correspondence as the entity order.

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

Build a content-addressed runtime manifest after an engine producer has written
its raw archive:

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

Materialize and then independently validate the self-contained bundle:

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
- an engine-independent profile descriptor embedded in the artifact;
- a topology digest and a frame-zero entity-order digest;
- finite JSON material parameters;
- an explicit causal observation cutoff;
- no future observations, target outcomes, or outcome-based backend selection;
- an independently simulated zero-action replay; and
- the known action bound to the driven replay.

## Third-party profiles

Additional engines can register a profile without modifying
Bayesian-PhysTwin. Entry-point loading is opt-in because it imports third-party
Python code:

```toml
[project.entry-points."bayesian_phystwin.physics_backends.v1"]
my-backend = "my_package.backend_profile:PROFILE"
```

`PROFILE` may be a `PhysicsBackendProfileV1`, its exact JSON mapping, a sequence
of profiles, or a zero-argument callable returning one of those forms. List or
resolve plugin profiles with `--include-plugins`. Plugin identifiers cannot
override built-ins, and the selected descriptor is embedded in the runtime so
bundle validation does not later depend on the plugin remaining installed.

## Advancement gate

A profile is only an available producer boundary. Before it can replace or join
the incumbent physical model on a target protocol, require all of the following
on already-open, source-only development data:

1. frame-zero query alignment in metres and in the registered coordinate frame;
2. repeat-run and zero-action noise-floor measurements;
3. a fixed parameter prior or source-only posterior with non-degenerate spread;
4. held-out source-prefix improvement or demonstrable complementarity to the
   incumbent under the same action and metrics;
5. calibration checks using proper multivariate scores, not mean trajectory
   error alone;
6. a frozen guard whose rejection path is a byte-identical incumbent artifact;
7. no target outcome or future observation in fitting, profile choice, or
   acceptance; and
8. a separately hashed prediction before independent future outcomes are
   opened.

The producer implementation order remains Genesis MPM on one already-open
Deform360 or PokeFlex development object, followed by JAX-FEM for differentiable
parameter inference. Warp FEM is the next choice when custom GPU-differentiable
mechanics are the bottleneck; PhysX is the next choice when operational contact
throughput is the bottleneck. SOFA, MuJoCo, PositionBasedDynamics, and Drake
provide complementary reference paths and should advance only after the first
two producers establish the end-to-end evidence protocol.
