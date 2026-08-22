# SOFA FEM canonical-gauge source qualification v3

**Registered, not yet executed.** This is one transparently versioned,
source-independent correction to the frozen negative v2 numerical gate. V2
remains negative and must not be rerun, relaxed, or replaced in place.

## Question

Does the pose-canonical SOFA variant preserve the native keyed-Dirichlet,
stable-Neo-Hookean source physics while removing the coordinate-conditioning
failure observed by v2, before any source outcome is read?

A pass authorizes only a separately frozen source-value gate. It is not
predictive, calibration, target, DEFORM, Causal4D, or state-of-the-art
evidence.

## Frozen ancestry

The protocol binds the v2 protocol at SHA-256
`76f2934082fec366b3a11c0c62d0f62802864dfde1e134f8c4143d9a285a8117`
and its negative result at SHA-256
`1508bd4f6f043825a8ad720a346e9cae0904da883e12ace4a2ba7e48a806084b`.
It also binds the passing v3 synthetic native smoke at SHA-256
`1785b151adc66bd6b52850336d7ed1c633746a378cb7a466ec23b36a8d9ba442`.
All three artifacts are hash-verified before native source execution.

The v3 protocol SHA-256 is
`4a9a72210787314e727b742795bb8c35af99ee6e75419d73435db2f1083eea73`.

## Correction boundary

V3 centers each registered material mesh, constructs a deterministic
right-handed principal-axis frame, and rounds canonical metric coordinates to
a fixed `10 pm` lattice. Geometries without a relative principal-axis eigengap
of at least `1e-6` fail closed. The solver output is reconstructed in the
original world frame.

This changes only the coordinate gauge presented to the native solver. The
tetrahedral topology, isotropic stable Neo-Hookean law, keyed moving Dirichlet
constraints, time step, density, damping, material values, source roster, and
opaque exact fallback remain those of v2.

## Frozen sources and probes

The gate reuses the exact target-blind inputs for `double_lift_zebra` and
`double_stretch_zebra`: 4,607/4,208 nodes, 23,659/21,307 tetrahedra, and the
same 40/67 and 7/9 contact-patch rosters. It may read frame-zero geometry, the
known controller trajectory, attachment weights, prepared tetrahedral/contact
archives, and incumbent bytes solely for hash verification and byte-exact
fallback copying. Incumbent arrays are never decoded.

Each group runs the same seven native scenes as v2: base, exact repeat,
zero-action, half-step refinement, low/high fixed modulus probes, and one fixed
global rigid-coordinate transform.

Every v2 physical criterion is retained. Coordinate-sensitive checks are split
so native parity remains at the v2 `1 pm` tolerance while the declared lattice
approximation is charged separately to a `20 pm` world-coordinate bound. V3
requires:

1. exact canonical gauge, scene, and schedule identity under rigid pose;
2. native zero-action drift and native frame-zero parity at most `1e-12 m`;
3. world-frame rigid equivariance error at most `1e-12 m`;
4. maximum world point and attachment approximation at most `2e-11 m`; and
5. the unchanged determinant interval `[0.5, 2.0]` and exact fallback.

Source object outcomes, incumbent prediction arrays, targets, held-out data,
future scoring, DLO4/DLO5, and held-v8 remain closed.

## One-shot command

The registered output root and log are absent. The first execution is the only
permitted execution of this v3 protocol.

```bash
set -o pipefail
ROOT=/tmp/sofa-v26.06-py310/SOFA_v26.06.00_Linux
SOURCE=/tmp/bpt-sofa-source-qualification-inputs-v2-490a452f
export SOFA_ROOT="$ROOT"
export SOFA_PLUGIN_PATH="$ROOT/plugins"
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/plugins/SofaPython3/lib"
export PYTHONPATH="$PWD/src:$ROOT/plugins/SofaPython3/lib/python3/site-packages"
/usr/bin/python3.10 scripts/remote/run_sofa_fem_source_qualification_v3.py \
  --protocol configs/sota/sofa_fem_zebra_source_physics_v3.json \
  --repo-root "$PWD" \
  --distribution-archive /tmp/SOFA_v26.06.00_Linux_Python3.10.zip \
  --sofa-root "$ROOT" \
  --group-root double_lift_zebra="$SOURCE/double_lift_zebra" \
  --group-root double_stretch_zebra="$SOURCE/double_stretch_zebra" \
  --output-dir /tmp/bpt-sofa-source-qualification-v3-v1 \
  2>&1 | tee /tmp/bpt-sofa-source-qualification-v3-v1.log
```

Any failed gate freezes a negative v3 result and leaves source-value scoring
closed. No automatic retry, threshold change, source replacement, or
outcome-guided revision is permitted.
