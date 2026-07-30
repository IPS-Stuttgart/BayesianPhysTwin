# RGBench Codim-IPC competence v5

## Purpose

This is a target-free numerical qualification of a stronger cloth backbone. It
does not compare against RGBench point clouds and cannot support an accuracy or
state-of-the-art claim.

The predecessor LibuIPC arm was finite and contract-preserving, but three
isolated full-horizon CUDA replays differed by 17.956 mm endpoint RMSE and its
soft controls missed their final targets by as much as 41.177 mm. Those errors
are too large relative to RGBench's published 28.99 mm GarmentDynamics result.

Codim-IPC is tested because it supplies:

- an implicit metric shell model parameterized by density, Young's modulus,
  Poisson ratio, and thickness;
- projected Dirichlet boundary conditions instead of virtual attachment
  springs;
- a CPU path that can be restricted to one thread for exact replay tests.

## New interface

Upstream Codim-IPC does not expose arbitrary node-index trajectories through
Python. The bound patch
`third_party/patches/codim_ipc_rgbbench_v5.patch` adds four narrow operations:

1. initialize Dirichlet controls from exact graph-node identities;
2. replace their metric targets at every step;
3. retrieve node positions without parsing rounded render files;
4. measure the maximum realized pin-target error.

The patch does not alter the shell energy, contact model, Newton solve, or line
search. The protocol binds the upstream commit, patch digest, and patched source
digest.

## Frozen gate

The source garment is `green_tshirt/fling/01`, using only its released mesh,
material metadata, known pin identities, and actuator trajectories. Two
independent 0.1 s replays run with:

- Codim-IPC commit `9c6cbe3`;
- Eigen's CPU linear solver;
- `OMP_NUM_THREADS=1`;
- 10 ms steps;
- exact moving Dirichlet controls;
- no collision in this first mechanics-only gate.

The arm advances only if:

- final arrays are byte-identical and finite;
- the vertex contract is unchanged;
- maximum pin error is at most 0.1 nm;
- mean cloth displacement is at least 0.01 mm.

The motion check prevents a deterministic no-op from passing. The pin threshold
distinguishes projected controls from another soft-attachment approximation.

## Information boundary

Allowed:

- source mesh and material metadata;
- source actuator trajectories and pin identities;
- replay equality, pin error, and displacement magnitude.

Forbidden:

- segmented point-cloud names or coordinates;
- source, calibration, or target accuracy outcomes.

Passing this gate authorizes only a separately frozen, longer target-free
qualification. It does not authorize opening source accuracy.
