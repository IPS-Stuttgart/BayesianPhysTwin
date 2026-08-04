# Deform360 Public-Data Evaluation v1

## Purpose

This workflow evaluates frozen BayesianPhysTwin endpoint-discrepancy dynamics on
the public Deform360 data already mounted on `workstation2` at
`/home/github-runner/.cache/datasets/deform360`. It is a read-only external-data
diagnostic. It does not download data, alter the cache, refit method parameters,
or claim parity with the official Deform360 world-model benchmark.

The workflow is `.github/workflows/deform360-public-evaluation.yml`. Every run
uploads the dataset inventory, environment identity, focused tests, complete JSON
result, concise Markdown summary, and dependency lock evidence.

## Supported released representations

The evaluator intentionally discovers data by validated array contracts instead
of depending on one historical private staging tree.

### Fixed-identity trajectories

An NPZ archive is eligible when it contains a finite array with shape
`(frames, tracks, 3)`. Keys declaring positions, particles, control points,
tracks, or trajectories receive priority. A matching validity array is consumed
when present.

For every rolling one-step prediction, only frames up to the current frame are
passed to the estimator. The hidden next frame is used only for scoring. The
three arms are:

1. exact persistence;
2. last observed residual;
3. the frozen 15-component Bayesian endpoint model average.

Identity RMSE and correspondence-free symmetric Chamfer RMSE are reported. The
raw model-average 90% ellipsoid coverage and effective component count are also
reported as diagnostics; the covariance is not described as calibrated.

### Packed visual hulls

Archives with `frame_indices`, `point_offsets`, and `points_world_m` are scored
without assuming point identity. The model observes only the causal sequence of
cloud centroids and predicts a global translation of the current cloud. Methods
are compared by centroid error and symmetric Chamfer RMSE.

This arm tests global motion handling only. It is not evidence that the method
predicts local deformation or a corrected latent physical state.

## Aggregation

Metrics are first averaged within each archive, then across archives. This keeps
long episodes from dominating shorter episodes. Object and archive counts are
reported separately. Discovery order is deterministic, and large arrays are
subsampled by evenly spaced indices under the selected workflow profile.

The profiles are:

| Profile | Archives | Frames/archive | Tracks/archive |
| --- | ---: | ---: | ---: |
| `smoke` | 8 | 32 | 512 |
| `standard` | 64 | 96 | 2,048 |
| `full` | 256 | 256 | 4,096 |

## Interpretation boundary

A positive result would establish that the frozen endpoint estimator transfers
to released Deform360 trajectory artifacts under a rolling causal-prefix
protocol. It would not establish official Table-4 parity, superiority over
Deform360 world-model baselines, tactile benefit, or physical-state correction.
Those require a separately locked adapter to the official benchmark inputs,
action conditioning, and the official evaluation targets.

A negative result remains useful because persistence and last residual are
strong, transparent comparators. It should trigger mechanism analysis rather
than target-dependent tuning on the same evaluated objects.
