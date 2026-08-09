# Deform360 Prob4D source calibration

## Purpose

This stage uses released real-world Deform360 measurements. It requires no new
recording, robot execution, contact-registration review, or human scientific
approval. The inputs are the locked calibration objects' causal multiview RGB
prefixes and an official Deform360 metric-prefix reconstruction generated from
those same public recordings.

The MotionCrafter producer intentionally writes exploratory Prob4D outputs.
They are not claim-bearing until point uncertainty, cross-window Sim(3) gauge
covariance, and the first-window metric prior have been calibrated. This stage
fits those artifacts without opening any confirmation object or future frame.

It does not authorize confirmation access. A separate registered observability,
transfer, and coverage gate must pass before the twelve confirmation objects can
be opened.

## Statistical units

Dense overlapping pixels are correlated and are not treated as independent
replications. The input bundle assigns every point row to a declared
camera/overlap dependence cluster. The fitter reduces each cluster to one
ratio-equivalent calibration row before invoking Prob4D's group-balanced point
calibrator. Each physical object then receives equal mass.

Gauge covariance is calibrated in scale, rotation, and translation blocks. The
fitter upper-winsorizes normalized residual ratios within each physical object,
then averages the resulting factors across objects. Consequently, an object
with more pixels, cameras, windows, or frames cannot dominate merely by
contributing more rows.

The source-support contract requires at least eight distinct physical objects
and at least four objects in each of two strata. A case must have at least two
independently recorded camera prediction manifests. Every valid point and gauge
row must lie inside that case's causal half-open frame range.

## Sample bundle

The JSON manifest uses schema
`bayesian-phystwin.deform360-prob4d-calibration-samples` version 2. Its
content-addressed `bundle_id` binds:

- the locked selection, visual-provider specification, and metric-prior policy;
- the exact Deform360, Prob4D, and MotionCrafter revisions;
- the successful visual-production result;
- every object, episode, stratum, camera, prediction manifest, and causal range;
- every metric-reference source and calibration digest; and
- one checksummed NPZ containing the numeric samples.

Every path in `source_artifacts` is relative to the sample-manifest directory.
The loader confines, opens, and re-hashes those ordinary files; a digest-only
claim without the corresponding portable source artifact is rejected.

The NPZ has these exact members:

| Member | Shape | Meaning |
| --- | --- | --- |
| `point_errors_m` | `(N, 3)` | Metric point residuals in metres. |
| `point_ray_directions` | `(N, 3)` | Observation-ray directions. |
| `point_parallel_variance_m2` | `(N,)` | Uncalibrated along-ray variance. |
| `point_lateral_variance_m2` | `(N,)` | Uncalibrated per-axis lateral variance. |
| `point_case_index` | `(N,)` | Index of the physical-object case. |
| `point_frame_id` | `(N,)` | Causal source frame. |
| `point_correlation_cluster_index` | `(N,)` | Global dependence-cluster identity. |
| `point_valid` | `(N,)` | Target-free validity mask. |
| `gauge_errors` | `(G, 7)` | Sim(3) residuals in log-scale, rotation-vector, translation order. |
| `gauge_covariance` | `(G, 7, 7)` | Uncalibrated correlated gauge covariance. |
| `gauge_case_index` | `(G,)` | Physical-object case for each gauge row. |
| `gauge_frame_id` | `(G,)` | Causal source frame for each gauge row. |
| `anchor_global_from_local` | `(P, 7)` | First-causal-frame metric Sim(3) estimate for each prediction stream. |
| `anchor_covariance` | `(P, 7, 7)` | Per-stream metric-anchor covariance. |
| `anchor_prediction_index` | `(P,)` | Exact flattened prediction-stream index for every anchor. |

Here `C` is the number of physical-object cases and `P` is the number of
successful camera prediction streams. Each camera has its own MotionCrafter
local gauge, so each `metric_references` entry is paired by `job_id` and
`camera_id` with one prediction manifest and one anchor. Reusing one object-level
anchor across cameras is invalid even when all cameras share the same metric
world frame.

The input constructor is expected to use the official Deform360 reconstruction
pipeline on the permitted prefix only. That constructor must publish its own
source artifacts and hashes; this fitter does not infer metric scale from
MotionCrafter predictions.

## Commands

Validate the bundle without importing Prob4D or reading prediction arrays:

```bash
python scripts/science/fit_deform360_prob4d_source_calibration.py validate \
  --samples /path/to/source-calibration-samples.json \
  --selection protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-spec protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json \
  --metric-prior-policy protocols/locks/deform360_official_hub_visuotactile_v1_metric_frame_prior_policy.json \
  --prediction-root /path/to/protected/calibration-predictions
```

Fit from a clean checkout at the exact frozen Prob4D revision:

```bash
python scripts/science/fit_deform360_prob4d_source_calibration.py fit \
  --samples /path/to/source-calibration-samples.json \
  --selection protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-spec protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json \
  --metric-prior-policy protocols/locks/deform360_official_hub_visuotactile_v1_metric_frame_prior_policy.json \
  --prediction-root /path/to/protected/calibration-predictions \
  --prob4d-checkout /path/to/exact/prob4d-checkout \
  --output-dir /path/to/new/source-calibration-result
```

The output directory is created once and contains content-addressed Prob4D point
and gauge calibration JSON files, one content-addressed metric anchor per source
case, a compact source-only result, and `SHA256SUMS`. It records
`confirmation_access_authorized=false` and `calibration_gate_evaluated=false`.

## Claim boundary

Passing this fitter means only that the public source residuals were converted
into internally valid, object-balanced Prob4D calibration artifacts. It does
not establish transfer, coverage, physical-query improvement, tactile benefit,
confirmation success, or state of the art.
