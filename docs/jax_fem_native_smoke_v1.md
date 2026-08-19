# JAX-FEM native smoke v1

## Scope

This is the first native-execution gate for the registered
`jax-fem-quasistatic-v1` material backend. It advances one of the two frozen
candidates in issue #664 without admitting another backend family.

The smoke is deliberately synthetic. Passing it establishes only that the
pinned JAX-FEM implementation actually executed, preserved the registered
material-query contract, produced a nonzero action response, held the
zero-action arm fixed, and reproduced the complete portable backend bundle byte
for byte. It does **not** establish source-value improvement, calibration,
fresh-object transfer, Causal4D benefit, deployment safety, or state of the art.

## Pinned upstream identity

The harness accepts exactly:

- repository: `https://github.com/deepmodeling/jax-fem`;
- revision: `82c6993c16704e38611f9cb91a5b70f1c690daee`;
- package version: `0.0.12`.

A package-version check alone is not accepted as revision evidence. Before any
solve, the harness recomputes Git blob identities for three installed JAX-FEM
source files and requires exact equality with the pinned upstream tree:

| Installed source | Required Git blob SHA-1 |
| --- | --- |
| `jax_fem/problem.py` | `8a20d24fc2e98aa33d4bd76e543f00c471740551` |
| `jax_fem/solver.py` | `f0f64cb629e202f2d179710b745ea4d682f1ace2` |
| `jax_fem/generate_mesh.py` | `bd564c8f4a049ae28bc3592e21d9547a5f509629` |

The SHA-256 values of those installed files are also bound into each portable
runtime manifest as producer source artifacts.

## Native problem

The smoke uses one 40 mm x 10 mm x 10 mm `HEX8` element with persistent mesh
node identity. The left face is fixed in all three coordinates. The driven arm
ramps the right-face x displacement to 4 mm over five frames; the zero-action
arm keeps it at zero. The constitutive law is small-strain isotropic linear
elasticity with `E = 100 kPa` and `nu = 0.3`.

Each solve constructs a fresh JAX-FEM `Problem` and calls JAX-FEM's native
`solver()`. The existing `produce_jax_fem_backend()` boundary then materializes
the result through `lagrangian-export-v1` and the common six-array
`physical_rollout_v1` contract. Prob4D and Causal4D therefore require no
backend-specific branch.

## Environment

Use an isolated environment that can import JAX and the dependencies required
by the pinned JAX-FEM checkout. Install JAX-FEM from the exact Git revision,
not merely from a same-version wheel. For example, after preparing the JAX
runtime and the upstream dependencies:

```bash
python -m pip install -e .
python -m pip install \
  "jax-fem @ git+https://github.com/deepmodeling/jax-fem.git@82c6993c16704e38611f9cb91a5b70f1c690daee"
```

The upstream revision imports `gmsh`, `meshio`, and the solver stack from its
normal package modules, so missing native dependencies fail closed rather than
silently substituting another implementation.

## Run

Choose a new output directory; the command refuses to overwrite an existing
path:

```bash
PYTHONPATH=src python scripts/remote/run_jax_fem_native_smoke.py \
  --output-dir results/sota/diagnostics/jax_fem_native_smoke_v1
```

The harness runs the complete driven/zero-action replay twice. It fails unless:

1. the installed JAX-FEM source matches the pinned Git blobs;
2. JAX reports at least one native execution device;
3. both runs materialize valid portable backend bundles;
4. every portable member is byte-identical across runs;
5. the prescribed 4 mm terminal displacement is reproduced within the frozen
   synthetic tolerance;
6. the driven-minus-zero response is non-degenerate; and
7. maximum zero-action drift is at most `1e-10 m`.

The output contains `run-a/`, `run-b/`, and
`jax-fem-native-smoke.json`. The JSON record binds the upstream source hashes,
JAX version and devices, physical problem, numerical checks, portable artifact
IDs, and SHA-256 hashes. It also records that no dataset payload, future
observation, or target outcome was read.

## Completed qualification path

The retained native smoke was used to freeze the JAX-FEM source-physics
qualification. That gate passed. The separately frozen full-horizon value arm
then failed its outcome-blind physical gate before prefix observations were
opened. JAX-FEM is therefore `source-physics-qualified`, not
`source-value-qualified`; the exact incumbent fallback remains selected. See
[`jax_fem_zebra_source_value_v1_result.md`](jax_fem_zebra_source_value_v1_result.md).

Genesis MPM remains the second selected candidate. New backend families remain
outside the main registry while the admission freeze is active.
