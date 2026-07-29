# Deform360 Causal-Response V12 Synthetic Controls

## Decision

The frozen implementation controls pass:

| Control | Result | Locked requirement |
| --- | ---: | ---: |
| Persistent nonrigid response detected | 12 / 12 | 12 / 12 |
| Placebo response admitted | 0 / 12 | at most 0 / 12 |
| Rejected placebo used exact fallback | 12 / 12 | 12 / 12 |
| Mean positive-control future RMSE improvement | 11.30% | at least 10% |

Every positive-control trial improved future RMSE. Mean RMSE changed from
1.688 mm for the unchanged baseline to 1.497 mm after the admitted update.

## Controls

The positive arm injects a persistent nonrigid graph response. The placebo arm
alternates between a rigid common-mode translation and a cross-panel
inconsistent nonrigid response. All arms pass through the production V12
admission, measurement, robust RBF update, and exact-fallback code paths.

The synthetic mechanism is fixed before prediction. Synthetic future values
are instantiated and scored only after the prediction has been formed. No real
object observation, identity, metric, target artifact, or held-v8 artifact or
process is accessed.

## Interpretation

This is an implementation control, not evidence that V12 improves real
Deform360 forecasts. It shows that the locked gate can detect the modeled
causal response, rejects the two declared nuisance/placebo classes, and
preserves the selected baseline exactly on rejection. The false-positive and
power properties outside these synthetic families remain unknown.

Fresh-object selection remains prohibited until the independently supplied
held-v8 all-attempt hash-only exclusion scope is available and the complete
exclusion union can be bound into the source lock.

## Evidence

- Result artifact:
  `results/sota/deform360_causal_response_synthetic_v12/summary.json`
- Canonical artifact SHA-256:
  `ed15430d4ba1bff9866d0ab9e9a858a37fc799469a5b736e2b95f0b6246a761a`
- File SHA-256:
  `74a45a2c81d222865236a39ec8d252eb6a2aee99370e816c14435f30b25f21d3`

