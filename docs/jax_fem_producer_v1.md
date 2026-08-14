# JAX-FEM producer v1

## Purpose

`bayesian_phystwin.jax_fem_producer_v1` adapts two fresh JAX-FEM solve
sequences to the strict external-physics archive and runtime manifest used by
Bayesian-PhysTwin. The module imports only NumPy and Bayesian-PhysTwin. It does
not import JAX or JAX-FEM and therefore does not add either package, an accelerator
runtime, or a nonlinear solver stack to the base installation.

JAX-FEM represents a displacement problem on persistent mesh nodes. Its usual
solve path constructs a `Mesh`, instantiates a `Problem`, and returns a list of
solution fields from `solver(problem)`. The first finite-element object stores
its reference node coordinates in `problem.fes[0].points`. The producer keeps
that engine-specific setup outside Bayesian-PhysTwin and consumes only a small
wrapper surface:

- `get_reference_points_m()` returns fixed node coordinates with shape `(N,3)`;
- `solve()` advances one load or time step and returns either one displacement
  array or the sequence returned by JAX-FEM's solver; and
- separate driven and zero-action callbacks configure the next solve before it
  is executed.

## Displacement and replay contract

The selected solution field must be a floating `(N,3)` nodal displacement in
metres relative to the fixed reference points. Frame zero is the reference mesh.
For transition `k`, the producer calls the relevant control callback and then
`solve()`. Thus an archive with `T` frames contains the reference state and
`T-1` solved transitions.

The caller supplies one `replay_factory`. It is invoked exactly twice and must
return distinct wrappers for the driven and zero-action arms. Both wrappers must
have bit-identical reference points, node order, mesh connectivity, boundary
conditions, material parameters, solver settings, and initial internal state.
A stateful dynamic wrapper may update its previous-solution history after each
solve, but `get_reference_points_m()` must remain unchanged.

The producer fails closed when:

- the reference points are not finite floating `(N,3)` arrays;
- a 2-D field has not been explicitly embedded into the registered 3-D frame;
- the two fresh wrappers do not expose bit-identical reference points;
- the reference mesh changes during a solve sequence;
- the selected displacement has the wrong node count, vector dimension, or
  dtype, or contains non-finite values;
- absolute positions overflow or become non-finite;
- a solution-field index is invalid;
- query indices or action support violate the generic external contract; or
- an output path exists, traverses a symbolic link, or loses a publication
  race.

The output NPZ is deterministic, no-pickle, timestamp independent, and
published without clobbering an existing artifact.

## Minimal JAX-FEM wrapper

A producer repository can keep all JAX-FEM imports and problem construction in
one wrapper. The exact boundary-condition and material implementation remains
experiment-specific:

```python
from pathlib import Path

from jax_fem.solver import solver

from bayesian_phystwin.jax_fem_producer_v1 import (
    produce_jax_fem_backend,
)


class Replay:
    def __init__(self, problem, control_sequence):
        self.problem = problem
        self.control_sequence = control_sequence
        self.step_index = 0

    def get_reference_points_m(self):
        return self.problem.fes[0].points

    def set_control(self, transition_index, *, zero_action):
        control = (
            ZERO_ACTION_CONTROL
            if zero_action
            else self.control_sequence[transition_index]
        )
        # Update only source-frozen loads, Dirichlet values, or internal
        # variables. The mesh and node order must remain fixed.
        self.problem.set_params(control)
        self.step_index = transition_index

    def solve(self):
        solution_fields = solver(self.problem)
        # A transient wrapper may update previous-step internal state here,
        # after retaining the displacement returned for this frame.
        return solution_fields


def replay_factory():
    problem = build_frozen_problem_and_mesh()
    return Replay(problem, FROZEN_KNOWN_ACTION_SEQUENCE)


def driven_control(k, replay):
    replay.set_control(k, zero_action=False)


def zero_action_control(k, replay):
    replay.set_control(k, zero_action=True)


result = produce_jax_fem_backend(
    raw_rollout_path=Path("raw-rollout.npz"),
    runtime_manifest_path=Path("runtime.json"),
    replay_factory=replay_factory,
    driven_control=driven_control,
    zero_action_control=zero_action_control,
    frame_count=len(FROZEN_KNOWN_ACTION_SEQUENCE) + 1,
    query_entity_indices=QUERY_NODE_INDICES,
    action_support=QUERY_ACTION_SUPPORT,
    solution_index=0,
    engine_revision=EXACT_JAX_FEM_GIT_REVISION,
    engine_version=REPORTED_JAX_FEM_VERSION,
    producer_repository="owner/jax-fem-producer",
    producer_revision=EXACT_PRODUCER_GIT_REVISION,
    coordinate_frame="right-handed-z-up-world-v1",
    time_step_s=OBSERVATION_FRAME_INTERVAL_S,
    topology_sha256=MESH_CONNECTIVITY_AND_NODE_ORDER_SHA256,
    material_model="compressible-neo-hookean",
    observation_end_frame_exclusive=OBSERVATION_END_FRAME,
    parameterization={
        "young_modulus_pa": 50000.0,
        "poisson_ratio": 0.3,
        "element_type": "TET4",
        "quadrature_order": 2,
        "nonlinear_solver": "newton",
    },
    producer_artifacts={
        "configs/problem.json": PROBLEM_CONFIG_SHA256,
        "assets/reference-mesh.vtu": REFERENCE_MESH_SHA256,
    },
)
```

`solution_index=0` matches the common one-variable JAX-FEM solve result. A
multi-field problem may select another index, but the selected field must still
be the 3-D nodal displacement. Scalar pressure, temperature, or damage fields
cannot be silently interpreted as geometry.

## Quasi-static and transient use

For quasi-static inference, each callback sets the next load state and each
`solve()` returns the corresponding equilibrium displacement. `time_step_s`
then records the physical observation-frame interval represented by adjacent
states; it is not a claim that Newton iterations model transient dynamics.

For a transient formulation, the wrapper may retain previous solutions and
update JAX-FEM internal variables between calls, as in a normal time-stepping
loop. The two replay instances must start from identical history and execute the
same integration settings. The zero-action arm must be independently solved,
not obtained by subtracting or editing the driven result.

## Three-dimensional registration

The portable physical rollout is three-dimensional. A native 2-D JAX-FEM
problem must be embedded explicitly by the producer wrapper into a frozen 3-D
coordinate frame. The embedding transform, plane convention, units, and source
mesh digest must be recorded in producer artifacts. The core adapter does not
pad a missing coordinate because silent padding would make frame and unit errors
difficult to detect.

## Provenance requirements

Bind at least:

- the exact JAX-FEM source revision and reported package version;
- the exact producer revision and JAX/JAXLIB versions used by that producer;
- reference mesh coordinates, connectivity, element type, and node-order digest;
- boundary-condition functions or their frozen generated values;
- material model, parameters, quadrature, linear/nonlinear solver settings,
  tolerances, and precision mode;
- transient integration and internal-state update rules, when applicable;
- the source-only query-node correspondence and `action_support` construction;
  and
- every transform from dataset/action coordinates to the registered FEM frame.

`topology_sha256` should bind connectivity and persistent node ordering, not
only the number of nodes. The runtime separately binds ordered frame-zero node
positions through `entity_identity_sha256`.

## Downstream materialization and qualification

The generic commands materialize and validate the result:

```bash
python -m bayesian_phystwin.cli.external_physics_backend materialize \
  raw-rollout.npz runtime.json jax-fem-bundle

python -m bayesian_phystwin.cli.external_physics_backend validate \
  jax-fem-bundle
```

A valid bundle establishes custody and contract conformance only. The exact
runtime must also pass `PhysicsBackendQualificationV1`, including deterministic
replay, equilibrium, equivariance, time-step refinement, finite-difference
Jacobian, source-query parity, and exact fallback gates. Target-facing use still
requires a frozen source-only guard and separately hashed predictions before
future outcomes are opened.
