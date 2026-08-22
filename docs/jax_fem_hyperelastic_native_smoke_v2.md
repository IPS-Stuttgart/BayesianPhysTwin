# JAX-FEM stable Neo-Hookean native smoke v2

`scripts/remote/run_jax_fem_hyperelastic_native_smoke_v2.py` is the native
finite-deformation gate for the `jax-fem-stable-neo-hookean-v2` variant of the
canonical `jax-fem-quasistatic-v1` backend family. The historical profile name
is retained for portable-contract compatibility; it does not mean that v2 uses
the rejected v1 small-strain formulation.

The v2 runner holds the upstream engine fixed at JAX-FEM revision
`82c6993c16704e38611f9cb91a5b70f1c690daee` and changes only the registered
physical formulation. It uses the singularity-free stable Neo-Hookean energy
of Smith, de Goes, and Kim (2018), JAX automatic differentiation for first
Piola stress and the consistent tangent, Newton line search, warm starts, and
fixed continuation substeps. Every continuation state is rejected if a
tetrahedral orientation probe is non-finite or non-positive.

Passing this smoke establishes exact native execution, finite-deformation
response, constitutive objectivity, orientation preservation on the synthetic
problem, deterministic portable publication, and exact runtime provenance. It
does not establish source value, calibration, fresh-object value, or downstream
benefit.

## Frozen runtime

- JAX-FEM `0.0.12`, revision
  `82c6993c16704e38611f9cb91a5b70f1c690daee`;
- Python `3.12.13`, JAX/JAXLIB `0.4.38`, NumPy `2.2.6`, SciPy `1.15.2`;
- petsc4py `3.23.7`, Gmsh `4.13.1`, and meshio `5.3.5`; and
- Git-blob and SHA-256 identities for the six installed JAX-FEM modules that
  implement basis evaluation, finite elements, mesh construction, problem
  assembly, and nonlinear solution.

## Synthetic finite-deformation problem

The mesh has 27 persistent nodes and eight `HEX8` elements. The minimum-x face
is fixed. The maximum-x face undergoes a 60-degree twist and 5 mm axial
extension, leaving the middle nine nodes to be solved by JAX-FEM. The seven
published frames use two fixed continuation substeps per load increment.

The gate requires:

1. two complete portable publications to be byte-identical;
2. zero-action drift at most `1e-10 m`;
3. driven response greater than `5 mm`;
4. all 48 tetrahedral orientation probes to remain above the frozen determinant
   threshold throughout continuation; and
5. normalized stress at rest and under a rigid rotation to remain below
   `1e-10`.

The output root must not exist. The result records
`future_outcomes_read=false` and `dataset_payload_read=false`.
