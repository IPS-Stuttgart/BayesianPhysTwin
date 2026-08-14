# External Lagrangian backend bridge v1

## Scope

This bridge extends the simulator boundary beyond the official PhysTwin/Warp,
MatPhys/Warp, and Newton implicit-MPM paths without adding either external
runtime to the default package. It admits two explicit producer profiles:

| Profile | Intended role | Persistent identity |
|---|---|---|
| `jax-fem-quasistatic-v1` | Differentiable, strain-native FEM for linear/nonlinear elasticity, hyperelasticity, and material-parameter inverse problems | Mesh-node index |
| `genesis-world-mpm-v1` | Broader dynamic/contact mechanics through Genesis World's MPM solver | Material-particle index |

The distinction matters. JAX-FEM is the higher-priority backend for the
repository's strain-inference use case. Genesis World is the broader dynamics
candidate for contact, large deformation, and future coupled-solver studies.
Neither profile replaces the existing backends or changes any frozen evidence.

The implementation is an **export bridge**, not a bundled simulator wrapper.
The external producer runs in its own pinned environment and exports a strict
trajectory archive plus a content-addressed runtime manifest. Bayesian
PhysTwin then validates and maps that export into `physical_rollout_v1`.

## Why an export boundary first

Both upstream stacks have dependency closures and supported-Python ranges that
are substantially larger or narrower than Bayesian PhysTwin's NumPy-only core.
Keeping execution outside the base wheel gives three useful properties:

1. Python 3.14 and CPU-only users can still install and validate artifacts.
2. GPL-licensed JAX-FEM is neither redistributed nor made a package dependency.
3. A solver/runtime upgrade cannot silently reinterpret an existing result;
   the exact repository revision, runtime version, source artifacts, and raw
   trajectory hash are bound into the artifact identity.

A direct in-process smoke can be added later as a separately versioned producer.
It must emit this same export contract and must not weaken its custody checks.

## Raw trajectory archive

The producer writes a no-pickle NPZ with exactly four arrays:

- `driven_point_positions_m`: `(T, P, 3)` absolute positions in metres;
- `zero_action_point_positions_m`: `(T, P, 3)` matched zero-action positions;
- `material_query_indices`: `(N,)` unique persistent mesh-node or particle
  indices; and
- `action_support`: `(N,)` finite values in `[0, 1]`.

All position and support arrays use one dtype, either `float32` or `float64`.
Frame zero must be byte-exact between driven and zero-action trajectories.
Index `p` must refer to the same Lagrangian point in every frame; resampling,
remeshing that changes point identity, nearest-neighbour reassociation, or
frame-wise sorting is not admitted by v1.

The bridge maps the selected points to:

- `prediction_m` and `driven_readout_m`: selected driven positions;
- `zero_action_readout_m`: selected zero-action positions;
- `persistence_m`: exact repetition of frame-zero positions;
- `action_support`: the producer-declared support; and
- `frame_zero_points_m`: selected frame-zero positions.

This is the same six-array contract already consumed by the downstream
Bayesian-PhysTwin belief and Prob4D/Causal4D integration paths.

## Runtime manifest

The JSON runtime manifest is strict: missing and unknown fields are rejected,
duplicate JSON keys and non-finite values are rejected, and `runtime_id` is the
SHA-256 identity of every field except itself.

Common fields bind:

- the admitted profile, exact upstream `owner/name` and 40/64-character revision;
- engine, Python, device, coordinate-frame, unit, and step-axis identities;
- frame, point, and query counts;
- profile-specific solver metadata;
- a non-empty path-to-SHA-256 map of source scripts/configuration/mesh files;
- the information boundary; and
- the raw NPZ SHA-256.

`jax-fem-quasistatic-v1` additionally requires:

- `element_type`;
- `constitutive_model`;
- `nonlinear_solver`;
- `differentiation_mode: "jax-autodiff"`; and
- `precision`.

Its step axis is `load-step` with dimensionless units. The frames may be a
quasi-static load continuation rather than physical time.

`genesis-world-mpm-v1` additionally requires:

- `solver: "mpm"`;
- `material_model`;
- `compute_backend`;
- positive `particle_size_m` and `substeps`;
- finite `gravity_m_s2`;
- a boolean `differentiable`; and
- `precision`.

Its step axis is physical `time` in seconds.

The information boundary permits either a fully synthetic export or a
source-only export. Future observations and outcomes must remain closed, and
the known action must be declared as used. A source-only export must bind that
the source dataset payload was read; a synthetic export must bind that it was
not.

## Commands

List the admitted profiles and their required metadata:

```bash
bpt experiment run materialize-lagrangian-backend profiles
```

Materialize an exported rollout:

```bash
bpt experiment run materialize-lagrangian-backend materialize \
  lagrangian-rollout.npz \
  lagrangian-runtime.json \
  output/lagrangian-backend
```

Validate a published bundle independently:

```bash
bpt experiment run materialize-lagrangian-backend validate \
  output/lagrangian-backend
```

The publication contains:

```text
lagrangian-backend.json
physical-prediction.npz
SHA256SUMS
provenance/lagrangian-rollout.npz
provenance/lagrangian-runtime.json
```

Publication is deterministic for identical input bytes. Validation recomputes
all file hashes and content identities, reloads the no-pickle raw archive,
reconstructs every physical array, checks material identity and precision, and
requires the exact root/provenance roster.

## Upstream revisions used to design v1

The profile contracts were checked against these upstream heads on 2026-08-14:

- JAX-FEM: `deepmodeling/jax-fem@82c6993c16704e38611f9cb91a5b70f1c690daee`;
- Genesis World: `Genesis-Embodied-AI/genesis-world@06a5f2518c254f7ef2cc8757a7f84ed96eb68232`
  (`genesis-world==1.3.3`).

These are documentation references, not hidden defaults. Every produced
artifact must state and bind the revision it actually used.

## Scientific and licensing boundary

JAX-FEM is GPL-3.0 and Genesis World is Apache-2.0. Neither project, its source,
its binaries, nor generated checkpoints are distributed by Bayesian PhysTwin.
Users install and run each producer separately under its upstream terms.

Passing this bridge proves structural and provenance compatibility only. It
does not establish upstream runtime correctness, differentiability, gradient
accuracy, physical competence, target transfer, calibration, deployment
safety, or state of the art. Those require separately registered experiments
with source/target custody and exact fallbacks.
