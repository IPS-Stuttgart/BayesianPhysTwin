# Deform360 Prob4D sample materializer

## Purpose

This stage turns released Deform360 measurements into the source-only residual
rows required by the Prob4D covariance fit. It requires no new capture and no
human registration approval. It consumes:

- integrity-bound MotionCrafter windows from the sole frozen calibration
  producer;
- sparse metric point grids obtained by projecting the released synchronized
  robot/taxel geometry through the matching released camera calibration; and
- the locked selection, visual-provider, and metric-prior contracts.

The output is the version-2 sample bundle consumed by
`fit_deform360_prob4d_source_calibration.py`. It does not evaluate a source gate,
authorize confirmation, or make a prediction-benefit claim.

## Metric-prefix archive

Every successful camera/job stream has one non-pickled NPZ with exactly:

| Member | Shape | Meaning |
| --- | --- | --- |
| `frame_indices` | `(T,)` | Exact released source-frame IDs. |
| `points_world_m` | `(T,H,W,3)` | Sparse released robot geometry in the Deform360 world frame. |
| `valid_mask` | `(T,H,W)` | Finite supported metric pixels. |

`frame_indices` must equal the complete registered causal half-open range. A
shortened, shifted, reordered, or future-crossing archive fails before any
residual is formed. The metric calibration bytes used to construct the grid are
bound separately for provenance.

The released `rendered_depth.h5` product is deliberately excluded: it is a
privileged full-sequence reconstruction and is not causal predictive evidence.
The robot metric grid reads no RGB pixels, tactile values, or rendered depth;
only registered prefix robot values contribute. It uses the same
cover-resize/crop pixel convention as the
frozen MotionCrafter producer. Projected robot geometry is a scale/frame gauge,
not an object-state target; mismatched or occluded rows remain calibration
residuals rather than receiving residual-dependent prior confidence.

The content-addressed metric-prefix plan uses schema
`bayesian-phystwin.deform360-prob4d-metric-prefix-plan`. Version 1 names every
successful production job exactly once. Version 2 separately binds the frozen
target-free camera-eligibility policy and partitions every successful
production job into an included stream or a retained visibility exclusion.
Each included stream binds `job_id`, `camera_id`, the sealed prediction
manifest, metric-prefix NPZ, and metric-calibration file. The included and
excluded sets must be disjoint and their union must equal the production
result's successful-job set. There is no replacement and no view selection
from camera images, prediction residuals, calibration outcomes, or future
data.

The registered batch constructor applies that rule to the complete frozen
visual-production roster. It has three terminal states:

- `all-streams-supported` emits the plan consumed below;
- `support-negatives-retained` records cameras where released robot geometry is
  outside the fixed prefix and emits no plan; and
- `technical-failures-retained` records a hashed software-failure detail and
  emits no plan.

Unsupported or failed cameras are never replaced. This distinction prevents a
technical failure from being presented as a model result and prevents
post-observation camera selection from changing the source cohort.

Under the separately locked version-2 policy, target-free visibility negatives
may produce `target-free-visible-streams-supported` only when all ten objects
retain at least two supported streams, at least 90% of all frozen streams are
supported, and no technical failure occurs. Every excluded stream remains in
`excluded_streams` with its original production identity and the single
allowed reason. Falling below any threshold produces
`camera-eligibility-gate-failed` and no plan.

## Residual construction

For each overlapping MotionCrafter window, the materializer fits a local-to-
metric Sim(3) transform on the window's first causal frame. The fit
uses frame/spatial-tile cluster-robust covariance and requires at least eight
independent tiles. Point residuals are then computed only on later causal frames
from that window. The uncalibrated Prob4D depth/disagreement model supplies the
parallel and lateral covariance before the source-only fit scales it.

Adjacent decoded windows are aligned with Prob4D's strict unknown-correlation
policy. Their estimated relative gauge is compared with the relative transform
derived from the two independent metric fits. Robot pose, camera calibration,
visibility, and association error are not subtracted from the residual, so the
resulting covariance scale is conservative.

Dense point rows share one effective correlation key across cameras and overlap
windows for each physical object, frame, and coarse image tile. Duplicating a
camera or decoded window can increase raw rows but cannot create a new effective
calibration cluster. Metric anchors remain stream-specific because every camera
prediction has its own local MotionCrafter gauge.

## Command

```bash
python scripts/science/materialize_deform360_prob4d_calibration_samples.py \
  --plan metric-prefix-plan.json \
  --production-result production/visual-production-result.json \
  --production-root production \
  --prediction-root /durable/calibration-visual-production \
  --metric-root /durable/deform360-metric-prefix \
  --selection protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-spec protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json \
  --metric-prior-policy protocols/locks/deform360_official_hub_prob4d_robot_metric_gauge_v1.json \
  --camera-eligibility-policy protocols/locks/deform360_official_hub_prob4d_camera_eligibility_v2.json \
  --prob4d-checkout /path/to/exact/prob4d \
  --prob4d-revision 25d90ef7f78ba4307f4555cb636d666004e1bf66 \
  --processing-revision d8522a4403b766aeb387510c04e89032a56fdf35 \
  --output-dir source-calibration-samples
```

Publication is atomic and no-overwrite. The output includes `samples.json`,
`samples.npz`, copied metric source/calibration bytes, and `SHA256SUMS`. Before
rename, the result is reloaded through the strict source-calibration consumer.

The per-camera public metric input is generated first with:

```bash
python scripts/science/materialize_deform360_robot_metric_prefix.py \
  --prepared-source-inventory prepared-source-inventory.json \
  --processed-root /protected/calibration-processed/aligned \
  --object-id OBJECT_ID \
  --camera-id CAMERA_ID \
  --processing-revision d8522a4403b766aeb387510c04e89032a56fdf35 \
  --target-height 320 \
  --target-width 640 \
  --output-dir /durable/metric-prefix/OBJECT_ID/CAMERA_ID
```

For the frozen production, generate every registered stream and the plan in one
atomic, no-overwrite publication:

```bash
python scripts/science/materialize_deform360_prob4d_metric_batch.py \
  --prepared-source-inventory prepared-source-inventory.json \
  --production-result production/visual-production-result.json \
  --production-root /durable/calibration-visual-production \
  --prediction-root /durable/calibration-visual-production \
  --processed-root /durable/calibration-processed/aligned \
  --selection protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-spec protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json \
  --metric-prior-policy protocols/locks/deform360_official_hub_prob4d_robot_metric_gauge_v1.json \
  --camera-eligibility-policy protocols/locks/deform360_official_hub_prob4d_camera_eligibility_v2.json \
  --processing-revision d8522a4403b766aeb387510c04e89032a56fdf35 \
  --implementation-revision "$(git rev-parse HEAD)" \
  --output-dir /durable/deform360-prob4d-metric-batch
```

The batch recursively hashes every metric member and publishes
`metric-batch-result.json`. With the optional version-2 policy,
`metric-prefix-plan.json` exists only when the locked target-free visibility
thresholds pass; without it, the byte-compatible version-1 all-stream rule
applies. The constructor reads released robot poses and camera calibration; it
performs no new recording and requires no human approval.

This path is governed by
`protocols/locks/deform360_official_hub_prob4d_robot_metric_gauge_v1.json`.
The older metric-frame-prior lock names a different reconstruction input and is
not silently reinterpreted as robot/taxel evidence.

## Claim boundary

This materializer establishes only causal, source-only residual construction
from released real measurements. It does not establish calibrated uncertainty,
transfer, confirmation benefit, official Deform360 benchmark parity, deployment
safety, or state of the art. Confirmation payloads and reserved future frames
are outside its interface.
