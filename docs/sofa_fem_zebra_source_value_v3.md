# SOFA FEM canonical-gauge source-value gate v3

**Status:** frozen source-physical rejection; no source outcome partition was
opened and no retry is authorized.

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

The original detached wrapper was terminated by the command orchestrator before
the generator created its output root. Both logs remained byte-empty, no source
group input was loaded, and no native replay began. Its compact receipt is
[`launch-interruption-v1.json`](../results/sota/diagnostics/sofa_fem_zebra_source_value_v3/launch-interruption-v1.json).
The original lock and logs remain preserved and must not be deleted or reused.

The command below was the sole managed recovery. The protocol, implementation,
ensemble, gates, and source roster remained byte-identical to the interrupted
registration. Its output root and log now exist and must not be replaced,
deleted, or used for another launch.

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
  --output-dir /tmp/bpt-sofa-source-value-v3-grid-managed-v2 \
  2>&1 | tee /tmp/bpt-sofa-source-value-v3-grid-managed-v2.log
```

Only after successful prediction may the outcome-free pre-prefix gate run:

```bash
/usr/bin/python3.10 scripts/remote/run_sofa_fem_source_value_v3.py \
  finalize-pre-prefix \
  --protocol configs/sota/sofa_fem_zebra_source_value_v3.json \
  --group-root double_lift_zebra="$SOURCE/double_lift_zebra" \
  --group-root double_stretch_zebra="$SOURCE/double_stretch_zebra" \
  --grid-dir /tmp/bpt-sofa-source-value-v3-grid-managed-v2 \
  --output-dir /tmp/bpt-sofa-source-value-v3-pre-prefix-managed-v2
```

## Frozen result

All three `double_lift_zebra` members and their equal-weight ensemble mean
sealed. The first `double_stretch_zebra` member, fixed at `25 kPa`, then failed
closed at native step 1094 when its minimum continuation determinant reached
`0.34743295104684863`, below the predeclared hard floor `0.35`. No stretch
archive or prediction grid was published.

This is a source-physical rejection, not a missing dependency or scheduler
failure. The pre-prefix gate was therefore not run. Prefix and future outcomes
remained unopened, so source-value scoring and an untouched evaluation are not
authorized. The exact incumbent fallback remains retained without retry,
threshold change, or parameter change.

The compact receipt is
[`failure.json`](../results/sota/diagnostics/sofa_fem_zebra_source_value_v3/failure.json).
It binds the exact implementation, qualification, distribution archive, native
log, and four partial archive hashes without publishing their payloads.
