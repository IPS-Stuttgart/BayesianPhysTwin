# Deform360 Bias-Aware Prospective V2

## Status

V2 is locked before download or media access for its three additional
calibration objects. It repairs only the target-free support failure of V1. The
source-v4 estimator, all candidate thresholds, calibration requirements, and the
12-object reserved target cohort are unchanged.

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
