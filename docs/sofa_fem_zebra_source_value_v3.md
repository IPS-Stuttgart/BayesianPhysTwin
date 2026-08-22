# SOFA FEM canonical-gauge source-value gate v3

**Status:** registered and target-blind; native prediction has not been
executed and no source outcome partition has been opened.

## Question

Does the source-qualified pose-canonical SOFA FEM v3 arm add predictive value
on the two registered PhysTwin source actions without changing its method in
response to object-motion outcomes?

This is an already-open source-value gate. It is not fresh-object, calibration,
target, DEFORM, Causal4D, or state-of-the-art evidence. A pass may authorize a
separately registered untouched evaluation. A failure retains each incumbent
archive byte-for-byte.

## Frozen ancestry

The protocol at SHA-256
`236e32b86c0fc45352f8abcc22408f5e84ce3f07cb6a6dae9b53a53104da607d`
binds the exact passing v3 source-physics protocol, result, qualification, and
runtime identities. It reuses the registered `double_lift_zebra` and
`double_stretch_zebra` source inputs, prepared tetrahedral/contact archives,
incumbent hashes, and separately bound prefix/future outcome hashes.

The incumbent bytes may be hash-verified and copied for exact fallback, but
their prediction arrays are never decoded. Outcome roots are not arguments to
prediction or the pre-prefix physical gate. MatPhys, target, reserve,
DLO4/DLO5, and held-v8 artifacts are outside the protocol.

## Frozen candidate

The arm is the equal-weight ensemble of the source-qualification probes at
Young's moduli `[25, 100, 500] kPa`, fixed Poisson ratio `0.3`, 32 native
substeps per 30 Hz interval, and the same pose-canonical stable-Neo-Hookean
runtime that passed v3 qualification. These values were fixed from the
source-independent low/base/high probes, not selected from motion outcomes.

The ensemble mean is the point prediction. The three members define an
equal-event 3D marginal energy score. Prefix scoring uses the final third of
the already-open prefix as validation and reports the first two thirds only as
fit diagnostics.

## Information order

1. Generate and seal every full-horizon native member, ensemble mean, runtime
   identity, source hash, and physical diagnostic from frame-zero geometry and
   the known controller trajectory only.
2. Re-derive and seal the pre-prefix physical gate without accepting any
   outcome-root argument.
3. Open prefix observations exactly once only after a cryptographically bound,
   passing pre-prefix receipt exists.
4. Freeze the source decision. Open future observations exactly once only if
   the prefix value gate authorizes them.
5. Preserve the first result, positive or negative, without retry, parameter
   change, source replacement, or outcome-guided revision.

The pre-prefix gate requires all three members in both groups, contact error at
most `0.02 m`, native attachment error at most `1e-12 m`, world attachment and
point approximation errors at most `2e-11 m`, node displacement at most
`0.35 m`, stored-frame determinants in `[0.5, 2.0]`, and native continuation
determinants above the fixed `0.35` hard floor.

## Value gate

On the prefix validation partition, equal-group point and energy ratios versus
persistence must each be at most `0.95`; the worst-group point ratio must be at
most `1.0`. Equal-group identity and Chamfer ratios versus the incumbent must
each be at most `1.05`. Final ensemble spread must lie in `[1e-5, 0.1] m`.

These are registered thresholds. They must not be changed after prediction or
after any source outcome is opened.

## Registered execution

The first command is the sole authorized native prediction once this
registration is committed from a clean revision. The output root must be
absent before launch and must not be replaced or reused for a duplicate.

```bash
set -o pipefail
ROOT=/tmp/sofa-v26.06-py310/SOFA_v26.06.00_Linux
SOURCE=/tmp/bpt-sofa-source-qualification-inputs-v2-490a452f
export SOFA_ROOT="$ROOT"
export SOFA_PLUGIN_PATH="$ROOT/plugins"
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/plugins/SofaPython3/lib"
export PYTHONPATH="$PWD/src:$ROOT/plugins/SofaPython3/lib/python3/site-packages"
/usr/bin/python3.10 scripts/remote/run_sofa_fem_source_value_v3.py predict \
  --protocol configs/sota/sofa_fem_zebra_source_value_v3.json \
  --physics-protocol configs/sota/sofa_fem_zebra_source_physics_v3.json \
  --physics-result \
    results/sota/diagnostics/sofa_fem_zebra_source_physics_v3/result.json \
  --qualification \
    results/sota/diagnostics/sofa_fem_zebra_source_physics_v3/material-backend-qualification.json \
  --repo-root "$PWD" \
  --distribution-archive /tmp/SOFA_v26.06.00_Linux_Python3.10.zip \
  --sofa-root "$ROOT" \
  --group-root double_lift_zebra="$SOURCE/double_lift_zebra" \
  --group-root double_stretch_zebra="$SOURCE/double_stretch_zebra" \
  --output-dir /tmp/bpt-sofa-source-value-v3-grid-v1 \
  2>&1 | tee /tmp/bpt-sofa-source-value-v3-grid-v1.log
```

Only after successful prediction may the outcome-free pre-prefix gate run:

```bash
/usr/bin/python3.10 scripts/remote/run_sofa_fem_source_value_v3.py \
  finalize-pre-prefix \
  --protocol configs/sota/sofa_fem_zebra_source_value_v3.json \
  --group-root double_lift_zebra="$SOURCE/double_lift_zebra" \
  --group-root double_stretch_zebra="$SOURCE/double_stretch_zebra" \
  --grid-dir /tmp/bpt-sofa-source-value-v3-grid-v1 \
  --output-dir /tmp/bpt-sofa-source-value-v3-pre-prefix-v1
```

Prefix and future commands remain unauthorized until each preceding sealed
gate explicitly permits the next information boundary.
