# Deform360 calibration visual-production plan

The successful official-Hub calibration-source run fixes the exact ten physical
objects, calibrated camera streams, and action-only source windows. The original
visual-provider lock fixes Prob4D, MotionCrafter, model snapshots, resolution,
window geometry, covariance retention, and the first-frame metric-prior policy.
It does not by itself turn those two artifacts into an executable per-view work
list.

`Deform360CalibrationVisualProductionPlanV1` closes that gap before visual
inference. It is a target-blind execution contract, not a result artifact.

## Inputs

The builder independently revalidates the complete successful source chain:

- calibration-source protocol;
- Stage-0 protocol and immutable 10+12 object selection;
- target-blind visual-provider lock;
- exact-file calibration plan and download manifest;
- strict successful terminal run record; and
- prepared calibration-source result.

Every planned object must be one of the ten frozen calibration objects and must
have status `source_prepared`. The twelve confirmation objects are never emitted
into the plan; only a digest of their frozen identity set is retained so the
execution boundary can be checked without publishing target identities.

## Camera policy

The plan uses every camera admitted by official calibration-source preparation.
For each object, camera IDs must be unique and are sorted lexicographically. No
camera is selected or dropped using prediction quality, target metrics, contact
outcomes, or a later visual result.

The source paths are derived rather than discovered recursively:

```text
<object-id>/episode_0000/<camera-id>/undistorted.mp4
<object-id>/episode_0000/<camera-id>/aligned_timestamps.txt
```

The corresponding output path is fixed as:

```text
objects/<object-id>/episode_<physical-episode-id>/views/<camera-id>
```

All paths are safe POSIX-relative paths, and output collisions fail closed.

## Frame policy

The prepared result supplies three nested, half-open source ranges for every
object:

- 81 selected action-window frames;
- the first 76 prediction frames; and
- the first 58 prefix frames.

The plan preserves the original frame indices. A later runner can therefore read
the original undistorted video directly with explicit `frame_start` and
`frame_stop`; no implicit recropping or re-encoding is required.

## Seed policy

The provider lock declares `per-object-derived-seed-v1`. This plan makes that
choice executable and independently reproducible.

One 32-bit object seed is the first four bytes of the SHA-256 digest of canonical
JSON containing:

```text
schema
provider root seed
visual-provider lock ID
object ID
physical episode ID
```

Each camera receives a deterministic substream seed derived from canonical JSON
containing the object seed and camera ID. Object and view seed collisions across
the complete ten-object plan are rejected.

A MotionCrafter invocation uses the camera substream as its root seed and the
Prob4D `derived-per-call` policy for baseline and window calls. Thus stochastic
lineage is explicit without relying on process order or worker scheduling.

## Dependence groups

Every view records two content-addressed dependence groups:

1. the shared frozen model set across all calls; and
2. the shared object/episode scene across views of one physical object.

These identities are intended for provider-v2 lineage and dependence-aware
fusion. They do not assert that camera streams are statistically independent.

## CLI

Build the plan only after a strict successful source record exists:

```bash
python scripts/science/build_deform360_calibration_visual_production_plan.py \
  build \
  --source-protocol protocols/deform360_official_hub_calibration_source_v1.json \
  --stage0-protocol protocols/deform360_official_hub_visuotactile_v1.json \
  --selection-lock <selection-lock.json> \
  --visual-provider-lock <visual-provider-lock.json> \
  --calibration-source-plan <calibration-source-plan.json> \
  --calibration-source-download <calibration-source-download.json> \
  --calibration-source-run-record <execution-manifest.json> \
  --calibration-source-result <calibration-source-result.json> \
  --implementation-revision <exact-40-character-BPT-commit> \
  --output <visual-production-plan.json>
```

Revalidation is independent and content-addressed:

```bash
python scripts/science/build_deform360_calibration_visual_production_plan.py \
  validate <visual-production-plan.json>
```

Publication is atomic and non-replacing by default.

## Information boundary

The plan records that calibration payloads were opened by the already completed
source stage. It simultaneously requires:

```text
confirmation_payloads_opened=false
target_outcomes_used=false
replacement_allowed=false
```

Building or validating this plan reads no video, tactile, robot, prediction, or
target-outcome payload. It does not run MotionCrafter, fit Prob4D, produce an
observability case, authorize confirmation access, or establish provider
competence. Its only claim is that the next calibration-only visual computation
has one deterministic, reviewable work list.
