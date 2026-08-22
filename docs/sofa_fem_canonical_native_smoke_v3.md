# SOFA FEM canonical-gauge native smoke v3

**Executed exactly once and passed.** This smoke qualifies the synthetic
native-execution and canonicalization boundary of the new
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

## Frozen result

The sole execution used implementation revision
`122760f754ce3eb1037930d01da677dc711ce16f` and produced smoke ID
`daf9282116be7c126c2b01191ed57a11602a4a446ee4d2edde8ecaf28dd57795`.
The committed receipt has SHA-256
`1785b151adc66bd6b52850336d7ed1c633746a378cb7a466ec23b36a8d9ba442`;
the external trajectory archive has SHA-256
`5ed08ef4b2070fa473e1249881ff89bc7e32598a555c27689bc0429f6579c63a`.

Repeated native trajectories were byte-identical. The rigid-pose
equivariance error was `3.5321681408794624e-17 m`, native attachment error was
`5.421010862427522e-20 m`, and the canonical world-boundary approximation was
`5.936135615772196e-12 m`. Deformation determinants remained between
`0.9909963061451169` and `1.0025660769164813`. Every dataset, source-outcome,
future-outcome, target, and held-out access flag remained false.

The compact receipt is stored at
`results/sota/diagnostics/sofa_fem_canonical_native_smoke_v3/result.json`.
The trajectory archive and execution log remain outside Git.

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
