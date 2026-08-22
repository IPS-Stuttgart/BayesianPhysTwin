# SOFA FEM keyed-Dirichlet source qualification v2

**Registered, not yet executed.** The frozen protocol is
`configs/sota/sofa_fem_zebra_source_physics_v2.json`. Its result must remain
unknown until the committed runner is executed once from a clean revision.

## Question

Can the exact pinned SOFA runtime replay two already-open PhysTwin source
actions with a stable finite-deformation constitutive law and genuine native
moving Dirichlet constraints, before any source outcome is read?

This is a numerical and custody gate. A pass authorizes only a separately
frozen source-value experiment. It is not evidence of predictive improvement,
uncertainty calibration, target performance, or state of the art.

## Corrected native boundary

The earlier `AttachProjectiveConstraint` experiment was rejected because its
position projection is symmetric even when `twoWay=false`; it does not provide
the intended one-way moving boundary. V2 instead creates one native
`LinearMovementProjectiveConstraint` for each attached material vertex. The
complete substep key schedule is derived from the registered rigid-patch action
before simulation starts. No target mechanical state and no post-step position
correction are used.

The exact runtime is SOFA `26.06.00`, revision
`7c18e95d5c5f2839079892c69e7d89a313c79603`, from the official Linux CPython
3.10 archive at SHA-256
`129211fd01781bdd5ba3f28f1c3617a2f3792a71b62dc609cf866eec4ac745e2`.
The runtime loader independently checks the archive, ABI, environment, reported
version, required plugins, and installed binary hashes.

## Frozen sources

The gate uses `double_lift_zebra` and `double_stretch_zebra`. It may read only
frame-zero material geometry, the known controller trajectory, registered
attachment weights, and the already prepared target-blind tetrahedral/contact
archives. Existing incumbent files are never decoded; they may be hashed and
copied as opaque bytes solely to prove exact fallback.

The prepared archives bind 4,607 nodes and 23,659 tetrahedra for lift, and 4,208
nodes and 21,307 tetrahedra for stretch. Their contact patches contain 40/67
and 7/9 attached vertices, respectively. The loader reconstructs the rigid
contact projection from source inputs and requires `1e-14` absolute parity with
the byte-exact prepared archive. This admits only last-bit SVD differences
between the preparation and pinned CPython 3.10 BLAS runtimes.

Source object outcomes, incumbent prediction arrays, targets, held-out data,
future scoring inputs, DLO4/DLO5, and held-v8 remain closed.

## Frozen probes

Each group runs seven native scenes over the first source interval:

1. the 32-substep base replay;
2. an exact independent repeat;
3. a zero-action equilibrium replay;
4. a 64-substep refinement;
5. fixed 25 kPa and 500 kPa Young's-modulus sanity probes; and
6. a fixed global rigid-coordinate transform.

All scenes use the native SOFA stable Neo-Hookean tetrahedral force field,
`E=100 kPa`, `nu=0.3`, `rho=1000 kg/m^3`, and Rayleigh stiffness/mass damping
of `0.1/0.1`, except for the two explicitly registered modulus probes.

Qualification requires exact repeat arrays, at most `1e-12 m` zero drift,
at most `1e-6 m` rigid-coordinate error, at most `10 um` absolute and 2 percent
relative 32/64-substep error, at least `0.1 mm` action response, measurable but
bounded modulus sensitivity, at most `1e-12 m` attachment error, at most
`75 mm` node displacement, and deformation determinants in `[0.5, 2.0]`.

## One-shot command

The two group roots must contain the exact files named by the protocol. The
output directory must not exist.

```bash
ROOT=/tmp/sofa-v26.06-py310/SOFA_v26.06.00_Linux
export SOFA_ROOT="$ROOT"
export SOFA_PLUGIN_PATH="$ROOT/plugins"
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/plugins/SofaPython3/lib"
export PYTHONPATH="$PWD/src:$ROOT/plugins/SofaPython3/lib/python3/site-packages"

/usr/bin/python3.10 scripts/remote/run_sofa_fem_source_qualification_v2.py \
  --protocol configs/sota/sofa_fem_zebra_source_physics_v2.json \
  --repo-root "$PWD" \
  --distribution-archive /tmp/SOFA_v26.06.00_Linux_Python3.10.zip \
  --sofa-root "$ROOT" \
  --group-root double_lift_zebra=/absolute/source/double_lift_zebra \
  --group-root double_stretch_zebra=/absolute/source/double_stretch_zebra \
  --output-dir /new/ordinary/output/directory
```

Any failed gate freezes a negative qualification and keeps source-value scoring
closed. No automatic retry, parameter change, replacement, or outcome-guided
revision is permitted.
