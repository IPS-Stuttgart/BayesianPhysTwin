# Genesis MPM native smoke v1

## Scope

`run_genesis_mpm_native_smoke.py` closes the native-execution/provenance gap for
`genesis-mpm-v1`, the second active backend-qualification candidate beside
JAX-FEM. It is deliberately a **synthetic smoke**, not a source-value or target
benchmark.

The smoke keeps the existing simulator-neutral boundary:

- Genesis owns the native MPM state and time stepping;
- `GenesisMPMEntityReplayV1` exposes only persistent material-particle positions;
- `produce_material_trajectory_backend` records fresh driven and zero-action
  worlds and publishes the existing six-array physical rollout; and
- Prob4D and Causal4D continue to consume that portable rollout without a
  Genesis-specific branch.

Passing this smoke does not establish simulator fidelity, parameter
identifiability, calibrated uncertainty, fresh-object transfer, Causal4D benefit,
deployment safety, or state of the art.

## Pinned native identity

The harness pins:

- repository: `Genesis-Embodied-AI/genesis-world`;
- revision: `0796d27667087d0087fe09d903f8aadf7fa9adeb`;
- package version: `genesis-world==1.3.3`; and
- exact Git-blob identities for the installed Genesis entry point, MPM entity,
  MPM solver, elastic material, and solver-options sources.

The smoke refuses to run when the installed package version or any pinned source
blob differs. It also binds the exact BayesianPhysTwin Git revision plus SHA-256
hashes of the smoke script, the Genesis replay adapter, and the pinned native
source files into the material-runtime provenance.

## Native problem

Each replay constructs a fresh CPU Genesis scene with:

- one elastic MPM box;
- zero gravity and no external contact object;
- fixed MPM bounds and grid density;
- `gs.materials.MPM.Elastic` with the corotational model;
- a frozen time step, substep count, density, Young's modulus, and Poisson ratio;
- 64-bit precision and deterministic algorithms; and
- a fixed random seed reset before every fresh scene construction.

The driven arm assigns a known uniform positive x velocity immediately before
each `scene.step()`. The matched zero-action arm explicitly assigns zero
velocity. The purpose is to establish that native MPM particle state can be
captured, advanced, and transported through the strict material-trajectory
contract while the zero-action world stays stationary.

Uniform translation is intentional here. A deformation/contact case belongs in
the subsequent source-physics qualification because this smoke should localize
software/provenance failures rather than mix them with constitutive-model or
contact-fidelity failures.

## Replay adapter

`GenesisMPMEntityReplayV1` has no Genesis import. A caller supplies a built scene
and MPM entity. The adapter uses the public surfaces:

- `entity.get_particles_pos()` for material state;
- optional `envs_idx=[env_index]` for exact selection from batched scenes; and
- `scene.step()` for one registered output step.

It synchronizes common tensor facades through `block_until_ready`, `detach`,
`cpu`, and `numpy` when those methods exist, returns an owning contiguous host
copy, and rejects multi-environment ambiguity, shape drift, non-floating state,
non-finite values, and disagreement with `entity.n_particles` when that public
count is exposed.

## Running the smoke

Use a clean BayesianPhysTwin checkout at the revision you want to attest and an
environment containing the pinned Genesis revision. For example:

```bash
python scripts/remote/run_genesis_mpm_native_smoke.py \
  --output-dir outputs/genesis-mpm-native-smoke-v1
```

The defaults are intentionally small and synthetic. The script validates all
arguments before importing Genesis, refuses to overwrite an existing output,
and executes the complete portable backend twice from independently initialized
Genesis runtimes.

## Passing checks

A successful run must establish all of the following:

1. the installed Genesis version and selected source files match the pinned Git
   identities;
2. each portable material-trajectory bundle is complete and self-validating;
3. the two complete portable bundles are byte-identical;
4. the zero-action material point has at most `1e-8 m` drift; and
5. the driven arm has a non-degenerate displacement above the frozen synthetic
   response floor.

The result is content-addressed as `genesis-mpm-native-smoke.json`. It retains the
engine/source identity, BayesianPhysTwin producer revision, runtime settings,
problem definition, numerical checks, artifact/runtime IDs, and SHA-256 hashes of
both portable runs.

## Evidence boundary and next step

This harness is infrastructure for the `native-smoke-passed` stage only. The
next claim-bearing work for Genesis MPM remains issue #664's frozen source-physics
qualification: equilibrium drift on source units, rigid-transform equivariance,
time-step refinement, persistent particle identity, physical-sanity checks,
Jacobian agreement where gradients are claimed, source-query parity, and exact
incumbent fallback.

Only after that source-physics record passes should Genesis enter the separately
frozen source-value comparison and Prob4D/Causal4D downstream-benefit gates.
