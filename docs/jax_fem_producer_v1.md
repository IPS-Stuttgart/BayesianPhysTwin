# Fresh-solve JAX-FEM producer v1

## Purpose

`jax_fem_producer_v1` turns matched JAX-FEM solve sequences into the existing
strict `lagrangian-export-v1` bundle. JAX and JAX-FEM remain optional external
producer dependencies: BayesianPhysTwin imports neither package and receives
only fixed-reference mesh-node trajectories plus exact runtime provenance.

The producer does not add another artifact family. Its output is the same
six-array `physical_rollout_v1` bundle consumed by BayesianPhysTwin, Prob4D, and
Causal4D. Existing downstream consumers therefore require no JAX-FEM branch.

JAX-FEM is the first-ranked external material backend because it provides a
strain-native differentiable FEM family for constitutive-parameter inference,
material ensembles, and sensitivity studies. That ranking records implementation
priority, not evidence that JAX-FEM is already more accurate than the incumbent.

## Replay wrapper

A producer-side wrapper implements two methods:

```python
class JaxFemReplayV1(Protocol):
    def get_reference_points_m(self) -> object: ...
    def solve(self) -> object: ...
```

The wrapper must satisfy all of these conditions:

- `get_reference_points_m()` returns floating `(P, 3)` mesh-node positions;
- positions are in metres in `right-handed-z-up-world-v1`;
- node index `p` denotes the same reference-mesh node throughout the replay;
- the reference mesh, node order, shape, and dtype stay fixed;
- `solve()` returns one selected total nodal-displacement field from that
  fixed reference mesh; and
- the selected displacement has shape `(P, 3)`.

For a multi-field JAX-FEM problem, the producer-side wrapper must select the
geometry displacement before returning. This keeps field selection in the exact
wrapper or configuration source hashed by `source_artifacts`; the portable
runtime does not carry an unbound field-index parameter.

JAX-like arrays are synchronized with `block_until_ready()` before host capture.
The producer also supports common `detach()`, `cpu()`, and `numpy()` facades and
copies every captured value into contiguous NumPy memory.

For a small integration, use the callback wrapper:

```python
from bayesian_phystwin.jax_fem_producer_v1 import CallbackJaxFemReplayV1

replay = CallbackJaxFemReplayV1(
    reference_points_callback=lambda: problem.points,
    solve_callback=lambda: solve_current_load_step()[DISPLACEMENT_FIELD],
    context=problem,
)
```

The callback or wrapper is responsible for presenting metres in the canonical
world frame. It must not expose visual vertices, quadrature points, reordered
nodes, or an updated/deformed mesh as the fixed reference.

## Fresh driven and zero-action arms

On a successful publication, `produce_jax_fem_backend` calls `replay_factory`
exactly twice. Each call must construct a fresh JAX-FEM problem,
boundary-condition state, nonlinear-solver state, and wrapper. Reusing one
object is rejected.

The execution order is fixed:

1. capture the fixed reference mesh as frame zero;
2. invoke the arm-specific control for transition `k`;
3. call `solve()`;
4. synchronize and capture the selected total displacement;
5. verify that the reference mesh is still byte-exact; and
6. publish `reference + displacement` as frame `k + 1`.

The first arm receives `driven_control`; the second receives
`zero_action_control`. Both arms must start from exactly equal reference points.
The producer rejects unequal fresh references, reference drift, changed shape or
dtype, non-finite values, duplicate or out-of-range material queries, invalid
action support, and an unselected list or tuple of solution fields.

The object-level freshness check cannot detect two different Python wrappers
that alias one hidden solver cache. The exact wrapper, scene, mesh, boundary
conditions, and resolver inputs must therefore be included in
`source_artifacts`, and qualification must include repeated fresh executions.

## Publication example

```python
from bayesian_phystwin.jax_fem_producer_v1 import (
    CallbackJaxFemReplayV1,
    produce_jax_fem_backend,
)


def replay_factory() -> CallbackJaxFemReplayV1:
    problem, solve = build_fresh_problem_and_solver()
    return CallbackJaxFemReplayV1(
        reference_points_callback=lambda: problem.points_m,
        solve_callback=lambda: solve()[DISPLACEMENT_FIELD],
        context=problem,
    )


LOAD_STEPS = 7


def driven_control(k, replay):
    replay.context.set_load_fraction((k + 1) / LOAD_STEPS)


def zero_action_control(k, replay):
    del k
    replay.context.set_zero_load()


artifact = produce_jax_fem_backend(
    output_dir="output/jax-fem-quasistatic-v1",
    replay_factory=replay_factory,
    driven_control=driven_control,
    zero_action_control=zero_action_control,
    frame_count=8,
    material_query_indices=query_indices,
    action_support=action_support,
    engine_revision="<exact 40- or 64-character JAX-FEM revision>",
    engine_version="<installed JAX-FEM version>",
    source_artifacts={
        "producer/run_jax_fem.py": "<sha256>",
        "producer/problem.py": "<sha256>",
        "scene/mesh.vtk": "<sha256>",
        "scene/config.json": "<sha256>",
        "environment/requirements.lock": "<sha256>",
    },
    device="gpu:0",
    load_step_size=1.0 / LOAD_STEPS,
    element_type="HEX8",
    constitutive_model="neo-hookean",
    nonlinear_solver="newton",
    source_kind="synthetic",
)
```

The output directory contains only the admitted Lagrangian bundle:

```text
lagrangian-backend.json
physical-prediction.npz
SHA256SUMS
provenance/lagrangian-rollout.npz
provenance/lagrangian-runtime.json
```

The runtime manifest binds the exact JAX-FEM revision and version, Python and
device identity, load-step size, element and constitutive model, nonlinear
solver, precision, source-artifact hashes, information boundary, and raw rollout
digest. The producer also inserts the SHA-256 of its own installed source module
under `bayesian_phystwin/jax_fem_producer_v1.py`; callers cannot override that
entry.

`source_kind="synthetic"` asserts that no dataset payload was opened.
`source_kind="source-only"` records already-open source/development payload
access while keeping future observations and all outcome-based selection closed.
Neither option authorizes target or confirmation access.

## Integration with Prob4D and Causal4D

The materializer selects the registered mesh-node queries and emits the existing
portable members:

- `prediction_m` and `driven_readout_m` from the driven JAX-FEM trajectory;
- `zero_action_readout_m` from the independent zero-action trajectory;
- exact frame-zero persistence;
- the registered action-support vector; and
- fixed frame-zero material points.

Prob4D may therefore evaluate calibration, shared covariance, and guarded
non-harm using the same physical-query boundary as every other backend.
Causal4D may consume the same candidate under its existing
abduction-intervention-prediction and exact-fallback rules. Neither repository
needs to import JAX, JAX-FEM, or producer-native topology.

## Advancement boundary

A successful bundle establishes execution order, fixed reference-node identity,
units, frame convention, source custody, deterministic portable materialization,
and downstream compatibility. It does not establish:

- physical fidelity or material-parameter identifiability;
- correctness of autodiff gradients;
- numerical convergence or mesh independence;
- calibrated predictive uncertainty;
- improvement over the current PhysTwin/Warp path;
- fresh-object or fresh-session transfer;
- Causal4D intervention benefit;
- deployment safety; or
- state of the art.

Before target-facing use, one exact JAX-FEM runtime must pass the common
source-only material-backend competence protocol. At minimum, freeze and test
repeated-run and zero-action floors, rigid-transform equivariance, load-step and
mesh refinement, parameter sensitivity, finite-difference agreement where
gradients are claimed, source-query parity, grouped predictive score and width,
runtime/failures, harmful accepted updates, worst-group regret, and byte-exact
incumbent fallback.
