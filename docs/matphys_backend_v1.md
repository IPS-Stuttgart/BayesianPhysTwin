# MatPhys physical-backend adapter v1

## Scope

Bayesian-PhysTwin can now consume a MatPhys material proposal as a distinct,
opt-in physical backend artifact. MatPhys is not a replacement simulator in
this stack: it proposes `spring_Y`, and the official PhysTwin Warp simulator
still produces the action-conditioned trajectory.

This guarded v1 consumer deliberately starts after proposal generation and Warp
replay. The separate
[`matphys_official_producer_v1`](matphys_official_producer_v1.md) interface now
binds the official checkpoint, complete material pipeline, spring field, and
fresh fixed-identity Warp replays before producing the candidate and
zero-strength archives consumed here. The frozen scalar-stiffness Deform360
runner is not silently repurposed as a per-spring MatPhys producer.

The adapter is additive. It does not change the frozen Deform360 v6.1
candidate, its `B0` physical fallback, or any Causal4D claim.

## Required inputs

The materializer accepts three compatible six-array physical archives:

1. the incumbent physical prediction;
2. the Warp replay using a nonzero MatPhys spring proposal; and
3. a zero-strength identity replay from the same MatPhys/Warp path.

It also requires a content-addressed proposal manifest and a disjoint
validation-prefix gate. The proposal manifest binds the exact MatPhys and
PhysTwin revisions, checkpoint, spring field, target evidence boundary,
source objects, and target-object exclusion. The gate binds its source
artifacts and the SHA-256 identity of all three evaluated archives, and rejects
validation that overlaps proposal fitting or crosses the future boundary.

The physical archive contains:

- `prediction_m`;
- `persistence_m`;
- `driven_readout_m`;
- `zero_action_readout_m`;
- `action_support`; and
- `frame_zero_points_m`.

The proposal may change the physical trajectory, but it may not change frame
zero, exact persistence, graph-node order, dtype, shape, or action-support
contract.

## Guard and fallback

The MatPhys proposal is selected only if all of the following hold:

- its zero-strength identity replay is below the frozen coordinate-RMSE limit;
- its balanced validation-prefix score improves by the frozen minimum;
- neither validation Chamfer distance nor validation track error exceeds the
  frozen regression allowance; and
- all target-object exclusion and future-blind provenance checks pass.

When any selection check fails, the output archive is a byte-for-byte copy of
the incumbent archive. Input failures fail closed instead of manufacturing a
fallback.

## Command

```bash
bpt experiment run materialize-matphys-backend materialize \
  matphys-proposal.json \
  matphys-gate.json \
  incumbent-physical.npz \
  matphys-candidate.npz \
  matphys-identity-replay.npz \
  output-directory
```

The same command exposes `proposal`, `gate`, and `validate` operations. Run it
with `--help` for their exact arguments. The legacy remote script delegates to
this registered experimental command.

The output directory is staged privately, self-validated, and atomically
published. Its `physical-prediction.npz` is directly compatible with the generic
Bayesian-PhysTwin physical-array consumer. `matphys-backend.json` records the
selection, identities, causal boundary, and exact-fallback proof.

## Evidence boundary

Earlier object-disjoint PhysTwin-22 experiments already showed why the guard
is mandatory. Selecting among MatPhys spring proposals improved the incumbent
from 10.849/19.482 mm to 10.242/19.059 mm CD/track, but an unguarded full
proposal regressed to 12.281/24.868 mm and the guarded result still missed the
published rounded 8/15 mm reference. Those opened results motivate this
interface; they do not validate a new Deform360 deployment.

A new evaluation should first run on already-open development objects, then
freeze one source-only protocol on genuinely fresh public objects. It must
compare incumbent, MatPhys/Warp, and guarded selection with the same causal
prefix, action, metrics, calibration audit, and exact fallback. The frozen
Deform360 v6.1 source scorer must not be retrofitted with this backend.

The currently preserved LOO22 workspace is a result summary rather than a
portable six-array replay bundle. The guarded consumer and official producer
now have contract and synthetic end-to-end coverage, but still have no newly
executed real-data score. A fresh preregistered evaluation is not justified by
implementation alone. First reproduce one already-open development case in
both the published per-case parity regime and the target-excluded causal regime,
including an exact zero-strength identity replay; only then lock a source-only
public-data panel.
