# SOFA Neo-Hookean FEM native smoke v1

`scripts/remote/run_sofa_fem_native_smoke.py` is the native-execution gate for
the registered `sofa-fem-v1` backend. It uses SOFA's own scene graph, implicit
integrator, sparse direct solver, tetrahedral topology, and Neo-Hookean FEM force
field. It does not substitute a NumPy solver for SOFA.

Passing this synthetic gate establishes native execution, fixed material
identity, constitutive sensitivity, finite non-inverted deformation, and exact
runtime provenance. It does not establish source value, fresh-object value,
calibration, or downstream Causal4D benefit.

## Frozen runtime

- SOFA `v26.06.00`, revision
  `7c18e95d5c5f2839079892c69e7d89a313c79603`.
- Official Linux CPython 3.10 archive
  `SOFA_v26.06.00_Linux_Python3.10.zip`, SHA-256
  `129211fd01781bdd5ba3f28f1c3617a2f3792a71b62dc609cf866eec4ac745e2`.
- The runner verifies `git-info.txt`, the hyperelastic FEM and direct-solver
  libraries, SofaPython3, and its Core and Simulation bindings byte for byte.

The process must be launched with `SOFA_ROOT`, `SOFA_PLUGIN_PATH`,
`PYTHONPATH`, and `LD_LIBRARY_PATH` pointing into that exact extracted archive.
The runner rejects mixed or substituted installations.

## Frozen synthetic scene

The scene contains 27 persistent nodes and 48 positively oriented tetrahedra.
The nine minimum-x nodes are fixed. A `ConstantForceField` drives the nine
maximum-x nodes while the matched zero-action replay receives zero force. The
constitutive law is `TetrahedronHyperelasticityFEMForceField` with the
`NeoHookean` material, parameterized from the registered Young's modulus and
Poisson ratio.

The smoke requires:

1. two complete portable bundles to be byte-identical;
2. zero-action drift at most `1e-12 m`;
3. driven response greater than `1e-4 m`;
4. all final tetrahedral deformation determinants to remain in `(0.5, 2.0)`;
5. half-modulus response to exceed double-modulus response by at least 25
   percent; and
6. fixed topology, finite state, exact source revision, and exact binary hashes.

## Isolated execution

After extracting the official archive:

```bash
export SOFA_ROOT=/absolute/path/SOFA_v26.06.00_Linux
export SOFA_PLUGIN_PATH="$SOFA_ROOT/plugins"
export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib"

python3.10 scripts/remote/run_sofa_fem_native_smoke.py \
  --distribution-archive /absolute/path/SOFA_v26.06.00_Linux_Python3.10.zip \
  --sofa-root "$SOFA_ROOT" \
  --output-dir /new/ordinary/output/directory
```

The output directory must not exist. The completed directory contains two
independent portable bundles and `sofa-fem-native-smoke.json`.
