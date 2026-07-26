# Penguin CoTracker3 Bias-Aware Source Study v1

## Status

This is an already-authorized, same-object source-development experiment on
penguin episodes 1, 3, 4, 6, 7, and 9. It does not access held-v8, the sealed
penguin episodes, or a fresh target object. It cannot establish object-level
transfer, calibration, state of the art, or non-regression on new objects.

The study asks one narrower question:

> Can exact-prefix CoTracker3 identities make the existing bias-aware Bayesian
> state update improve the selected reusable physical response?

## Prediction Boundary

The `predict` operation consumes only:

- the already selected candidate-03 physical response;
- calibrated camera geometry;
- frame-zero mask and depth arrays;
- exact RGB prefixes ending at frames 19, 38, and 57.

For update frame `u`, CoTracker3 receives exactly frames `[0, u]`. Sixteen
frame-zero material identities are projected into five deterministically
selected cameras and queried directly. No full-window `vel.h5`, future
`pcd_clean`, source metric, tactile stream, or held artifact is accepted.

Every episode prediction is written before any source PCD outcome is opened.
The separate `seal` operation requires all six predictions and hashes their
archives and reports. Only then may `evaluate` read the authorized source PCDs.

## Observation And Update

Sparse tracks are triangulated with the shared deterministic multiview RANSAC
contract. Two-view observations are permitted because hard three-view
admission was itself a prospective failure surface, but their metric variance
is conservatively inflated. Leave-one-view disagreement can only increase
variance, and duplicated coherent support cannot reduce it below the fixed
metric floor.

Prior reliability uses only view redundancy and reprojection geometry. It does
not use the residual against the physical state. The innovation enters once
through the existing Student-t bias-aware likelihood.

The state update is otherwise the frozen source-v4 method:

- rank-4 causal physical-response support;
- explicit shared spatial and camera-bias nuisance terms;
- removal of state directions confounded with the declared bias basis;
- at least 0.5 mm physical and observed response;
- at least 0.40 robust physical agreement;
- persistent readout of accepted response-constrained state coefficients;
- exact selected-physical fallback on every failure or rejection.

## Source Evaluation

The primary outcomes are hidden-identity RMSE and hidden symmetric Chamfer
after the update frames. The sixteen queried identities are excluded from both
metrics. A six-fold leave-one-episode-out guard is fit on the other five
source episodes, with each episode treated as one finite-sample group.

The source accuracy gate requires all of:

1. at least 5% aggregate improvement in both primary metrics;
2. at least 5% late-horizon improvement in both primary metrics;
3. no episode-metric degradation above 10%;
4. at least one held-episode candidate admitted by the guard.

This gate does not include a calibration claim. Six episodes from one physical
object do not provide object-level exchangeability or 90% finite-sample
resolution. A positive result would justify a separate fresh-object protocol
with an independent calibration panel; a negative result rejects this feeder
for scaling and leaves the selected physical response unchanged.

## Commands

```bash
python scripts/remote/run_deform360_penguin_cotracker_bias_source.py predict \
  --staged-root /path/to/staged-v5 \
  --response-root /path/to/selected-backbone-candidate03 \
  --output-root /path/to/predictions \
  --cotracker-source /path/to/co-tracker \
  --cotracker-checkpoint /path/to/scaled_offline.pth \
  --device cuda:0

python scripts/remote/run_deform360_penguin_cotracker_bias_source.py seal \
  --prediction-root /path/to/predictions

python scripts/remote/run_deform360_penguin_cotracker_bias_source.py evaluate \
  --prediction-root /path/to/predictions \
  --staged-root /path/to/staged-v5 \
  --output /path/to/result.json
```
