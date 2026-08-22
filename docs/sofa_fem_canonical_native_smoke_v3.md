# SOFA FEM canonical-gauge native smoke v3

**Registered, not yet executed.** This synthetic smoke qualifies the new
`sofa-stable-neo-hookean-canonical-gauge-keyed-dirichlet-v3` backend variant.
It does not replace or rerun the frozen negative v2 source gate.

## Correction boundary

SOFA v2 passed every source-physics check except rigid-coordinate equivariance
on the lift mesh. A global pose perturbation below `2.3e-16 m` at the registered
contact boundary was amplified to `7.2 um`. No source object outcome was opened.

V3 fixes this generic interoperability defect before native execution. It
centers the material geometry, constructs a deterministic right-handed
principal-axis frame, and rounds canonical metric coordinates to a fixed
`10 pm` lattice. Geometries whose principal axes are not identifiable are
rejected. SOFA still receives the same tetrahedral topology, stable Neo-Hookean
law, keyed moving Dirichlet constraints, time step, damping, density, and
material parameters.

The quantized canonical boundary is separately required to remain within
`20 pm` of the original world-frame boundary. Native SOFA projection retains
its existing `1 pm` gate.

## Synthetic gate

The committed runner executes one irregular five-node, two-tetrahedron source
scene twice, then executes its fixed globally rotated and translated copy. It
requires:

1. exact repeated trajectory and determinant arrays;
2. identical gauge, scene, and schedule identities under global pose;
3. world-frame equivariance error at most `1e-12 m`;
4. native attachment error at most `1e-12 m`;
5. world-boundary approximation error at most `2e-11 m`;
6. deformation determinants in `[0.5, 2.0]`; and
7. exact SOFA distribution, binary, source, and clean-Git provenance.

The smoke reads no dataset, source outcome, target, held-out, DLO4/DLO5, or
held-v8 artifact. A pass is necessary but not sufficient to register a new
source-physics protocol.

## Frozen invocation

The first and only execution uses the already hash-verified SOFA distribution
and an absent output root:

```bash
set -o pipefail
ROOT=/tmp/sofa-v26.06-py310/SOFA_v26.06.00_Linux
export SOFA_ROOT="$ROOT"
export SOFA_PLUGIN_PATH="$ROOT/plugins"
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/plugins/SofaPython3/lib"
export PYTHONPATH="$PWD/src:$ROOT/plugins/SofaPython3/lib/python3/site-packages"
/usr/bin/python3.10 scripts/remote/run_sofa_fem_canonical_native_smoke_v3.py \
  --distribution-archive /tmp/SOFA_v26.06.00_Linux_Python3.10.zip \
  --sofa-root "$ROOT" \
  --repo-root "$PWD" \
  --output-dir /tmp/bpt-sofa-canonical-native-smoke-v3-v1 \
  2>&1 | tee /tmp/bpt-sofa-canonical-native-smoke-v3-v1.log
```
