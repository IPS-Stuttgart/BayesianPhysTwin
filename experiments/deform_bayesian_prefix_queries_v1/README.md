# Bayesian prefix-query feasibility v1

This research-only implementation starts with the nine-recording validation
headroom screen. Failure is terminal for this setting: the larger selector
pipeline is conditional, and no source-test loader is exposed here.

The [execution receipt](execution_receipt.json) records the completed source
screen's terminal stop. The eight-case evaluation and selector tournament were
not authorized. Full results and interpretation belong to the private paper
repository under `experiments/gaussian-processes/dlo3-prefix-query-headroom`.

## Scientific boundary

Can a source-trained correlated posterior select useful point histories after
changing simulator? The historical 2.98% unchanged DEFORM-to-PyElastica mean
improvement motivates the test but does not establish Bayesian value.
Neither covariance-based sensor selection nor active discrepancy learning is new:
[Krause et al.](https://jmlr.org/papers/v9/krause08a.html),
[Yang et al.](https://arxiv.org/abs/2502.05372).

`gp.py` reuses the finite-rank GP algebra from the unchanged trial at
`1b0622c823d6d927ab9d4863911caab67e98ebaf`. Material nodes are pooled here;
the existing causal features contain material arc coordinates. Canonical
coordinate blocks remain independent, but node/time covariance is retained.
Inference is Gaussian for the specified finite-rank model, not automatically
calibrated physical uncertainty. Recorded 3D coordinates are not camera cues.

## Information and calibration

The checksum-bound source split is 39 fit, nine calibration/validation and eight
source-test recordings. This stage exports only the first 48. The registered
seed-42 update-6400 checkpoint is frozen, not retrained. Its historical training
exposure remains a limitation. Native replay receives the first two states and
future clamped motion; future free-node values are replaced by the second
initial state even in otherwise unused input fields. Original clipping and
coordinate conventions are retained. Dataset reads are whitelisted; the eight
source-test paths are explicitly denied. No mixed archive download is needed.

The screen uses DEFORM validation replays, not PyElastica test outcomes. The
shared forecast is the physical rollout plus GP residual mean at shrinkage
0.25. Latent covariance is scaled by 0.25 squared. Observation noise is fitted
residual variance times one of 0.25/1/4/16, with a 1e-12 m^2 floor. Select the
multiplier by joint NLL on the permitted validation prefixes only. Reusing
these nine recordings for headroom is a descriptive development screen, not
held-out calibration evidence or a distribution-free coverage claim.

At endpoints 50/150/250, one query reveals the ten past samples endpoint-46 to
endpoint-1 at stride five. Nodes 2/4/6/8 and 3/5/7/9 alternate as query and
never-queried scoring panels. The primary continuation is 100 frames. Every
one of the six two-point forecasts uses the same conditioning engine and is
sealed before scoring. Query scoring receives no measured values or future
truth. Invalid covariance returns exact open-loop fallback and a failure record.

The fixed pair is optimal on validation separately per panel but shared across
all recordings/endpoints. The oracle chooses actual minimum future MSE per
context and is scoring-only. Aggregation is square root of the equally weighted
recording/context/panel coordinate MSE. Require at least 10% relative RMSE
headroom. Failure means stop; no architecture search or source-test opening.

## Reproduction

Stage an exact clean Git archive, record its SHA-256, and pass its full commit.
Both operators additionally record their executed Python-file hashes. Use a new
output root; never overwrite or automatically retry a technical failure.

```bash
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts/remote"
python -m experiments.deform_bayesian_prefix_queries_v1.prepare \
  --revision "$COMMIT" --output "$ROOT/cache" --device cuda:0 --threads 4
python -m experiments.deform_bayesian_prefix_queries_v1.run \
  --revision "$COMMIT" --cache "$ROOT/cache" --output "$ROOT/headroom"
```

The exporter uses the retained native DEFORM environment, whose registered
backend requires CUDA. The first CPU invocation failed before any replay and
is retained as a setup failure; an explicitly authorized CUDA export uses a
separate output directory without changing the checkpoint or method. The GP stage needs
only NumPy/SciPy, not a GPU. The method seal binds model/calibration/cache/source;
the prediction seal binds all 324 validation candidates before future scoring.

## Conditional larger study

A verified passing screen permits implementing and freezing the full comparator
panel: random, spatial, physical-motion, fixed-pair, cluster-sandwich, trajectory
bootstrap, cross-fitted ridge/kernel utilities, and diagonal/scrambled/matched
distribution controls. Those comparisons are not claimed as implemented or run
by this headroom stage. A headroom pass alone does not authorize opening the
eight cases before comparator and prediction-custody implementation is complete.

The eight-case gate remains 5% RMSE improvement, 6/8 wins, worst ratio <=1.10,
non-degraded late error, no-query improvement, and no independent control beating
the posterior. Secondary horizons cannot rescue it. A pass motivates a separate
replication protocol, not confirmation or SOTA. No held-v8, DLO4/DLO5, official
evaluation, Prob4D, or robot execution belongs to this experiment.
