# Deform360 Bias-Aware Prospective V2

## Status

V2 completed its calibration phase. The target-free support gate passed, but the
fresh accuracy gate failed: four objects received candidate updates and all four
were harmful on at least one co-primary metric. Target access remains forbidden,
and no reserved target object, media, or future was opened. See
`docs/deform360_bias_aware_prospective_v2_result.md` for the frozen result.

V2 repaired only the target-free support failure of V1. The source-v4 estimator,
all candidate thresholds, calibration requirements, and the 12-object reserved
target cohort remained unchanged throughout execution.

The canonical protocol is
`configs/sota/deform360_bias_aware_guarded_belief_prospective_v2.json` with
config SHA-256
`67e1157fa04283f1376855a7ac60f85a4de02434612592ffe8b4ef1e4607ebe4`.

## Why V2 exists

V1 stopped before outcomes because only five of nine calibration objects produced
automatic physical twins. Its frozen gate required at least seven, including at
least two per deformation stratum. The result did not test predictive accuracy and
left every calibration future and all target data sealed.

A subsequently frozen integration path gives failed frame-zero reconstructions a
`persistence_only` disposition. Candidate and baseline are then bit-identical, no
physical graph or Warp rollout is invented, and the case cannot count as an
accuracy or calibration sample.

V2 retains the nine V1 calibration objects and adds three fresh filament objects:

| Object | Episode | Selection evidence |
| --- | ---: | --- |
| `078-fishing-line` | 4 | SHA-256 rank over frozen names |
| `161-tube` | 4 | SHA-256 rank over frozen names |
| `088-snake` | 1 | SHA-256 rank over frozen names |

Before lock, both GPU servers reported zero matching directories for these three
objects. `072-cotton-clohesline` was excluded because an earlier source preflight
had already created matching paths.

## Unchanged support gate

The complete 12-case calibration disposition is sealed before any new future is
opened. V2 proceeds only if:

- at least seven calibration objects have automatic physical twins;
- at least two automatic twins exist in every stratum;
- at least two of the three fresh filament objects have automatic twins;
- at least five new eligible object groups remain for the regret certificate;
- source plus calibration provide at least nine eligible groups and exact 90%
  finite-sample rank;
- no failed object is replaced.

Fallback ties do not manufacture support. If the gate fails, calibration futures
and all target data remain sealed.

## Accuracy gate

If support passes, only the direct source-group regret certificate may be refit.
The method family, features, rank, observation model, thresholds, and covariance
model stay frozen. Target access requires:

- a regret upper bound below -0.005 mm;
- negative object-balanced regret on both co-primary metrics;
- zero harmful accepted calibration objects.

The target evaluation then uses the unchanged V1 cohort and object-clustered
statistics. It is an online causal-prefix comparison against the selected raw
physical/persistence backbone, not official open-loop Deform360 Table-4 parity.

## Information order

```text
commit and push V2 lock
-> download only three fresh calibration objects
-> stage causal prefixes
-> seal automatic-twin or persistence-only disposition for all calibration cases
-> evaluate support without futures
-> open calibration futures only if support passes
-> freeze or reject the direct regret certificate
-> touch the reserved targets only if every calibration gate passes
```

The intended positive claim, if every gate passes, is limited to prospective
improvement over the strong selected Bayesian/physical backbone across fresh
public deformable objects. No state-of-the-art claim is licensed by the protocol
alone.

## Additive execution lock

The v1 prediction implementation remains checksum-frozen. Before any fresh
RGB prefix is decoded, the additive v2 protocol-identity adapter and the exact
reused remote stages are bound by
`configs/sota/deform360_bias_aware_guarded_belief_prospective_v2_execution.json`.
The adapter installs no outcome loader and authorizes only target-free
prediction construction for the three fresh calibration objects. Each stage
is launched through `run_deform360_bias_aware_v2_stage.py` from a clean Git
checkout; process exit restores the original v1 bindings.

## Exact camera-search acceleration

The first fresh prediction attempt exposed a target-free runtime problem before
any final prediction or outcome was opened: the frozen selector enumerates every
8-camera subset, which means 10,518,300 subsets for 32 cameras and 30,260,340
subsets for 36 cameras. The attempt completed and sealed the fishing-line
measurement and uncertainty intermediates, but was stopped during cycle replay;
no candidate prediction was produced.

The replacement changes only how the same lexicographic camera objective is
searched. A checksum-addressed native depth-first solver returns the exact optimum
and preserves the exhaustive selector's first-in-order tie rule. Before reuse, it
must:

- match the frozen selector on randomized and tied synthetic inputs;
- reproduce the selected cameras, centers, and score already sealed by the
  original fishing-line measurement;
- run behind a new target-free execution lock committed before restart;
- leave the observation model, tracker, candidate, thresholds, and cohort intact.

The interrupted intermediates are retained as an aborted execution record and are
not mixed with the restarted prediction cohort. This is an execution-equivalence
amendment, not a method or gate change.
