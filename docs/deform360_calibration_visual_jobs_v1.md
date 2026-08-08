# Deform360 calibration visual jobs v1

## Purpose

The official-Hub calibration-source execution now provides ten synchronized,
action-windowed RGB/tactile/robot objects. Before MotionCrafter or Prob4D runs,
the exact camera inputs, causal frame limits, window schedule, and stochastic
seeds must be frozen without selecting cameras from target outcomes.

`Deform360CalibrationVisualJobsV1` is that bridge. It plans every aligned camera
for every source-supported calibration object. It does not select a preferred
camera subset and it does not run inference.

## Required source evidence

The builder accepts:

- the exact Stage-0 protocol and 10+12 selection lock;
- the exact target-blind visual-provider lock;
- one successful calibration-source terminal record;
- its exact prepared-source result;
- the runner-local processed calibration root; and
- the exact BayesianPhysTwin implementation revision.

The terminal record must report a successful 8/10 plus 4/5-per-stratum source
gate and a verified closed confirmation boundary. The exact selection,
visual-provider, and prepared-result file identities must match that record.

For each source-prepared object, the builder verifies the result-bound
`alignment.json`, the complete sorted camera list, and ordinary non-symlinked
`undistorted.mp4` files. Exact video sizes and SHA-256 values enter the manifest;
local absolute paths do not.

## Causal schedule

The action-only staging contract is unchanged:

- selected window: 81 frames;
- prediction span: 76 frames;
- causal provider prefix: 58 frames;
- evaluation-only future: frames 58 through 75 relative to the selected start;
- cutoff convention: exclusive.

MotionCrafter and Prob4D may read only the 58-frame prefix. With the locked
25-frame window and 8-frame overlap, each camera has exactly these relative
windows:

```text
[0, 25)
[17, 42)
[33, 58)
```

The final window is explicitly anchored at `prefix_stop - window_size`, matching
`Prob4D@25d90ef7f78ba4307f4555cb636d666004e1bf66`.

## Stochastic schedule

The provider lock declares `per-object-derived-seed-v1`. The manifest derives one
32-bit object seed as the first four bytes of SHA-256 over canonical JSON:

```json
{
  "schema": "bayesian-phystwin.deform360-per-object-derived-seed-v1",
  "root_seed": 20260805,
  "object_id": "<physical object>"
}
```

Within an object, the exact Prob4D `derived-per-call` algorithm derives separate
seeds for:

- the disjoint baseline;
- the latent-linear baseline; and
- each independently decoded overlap window.

The schedule creates deterministic common random numbers across cameras of one
physical object while preventing seed reuse across products and objects. This is
a reproducibility device, not a claim of statistical independence.

## Claim-bearing command

```bash
python scripts/science/build_deform360_calibration_visual_jobs.py \
  --stage0-protocol \
    protocols/deform360_official_hub_visuotactile_v1.json \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json \
  --calibration-source-run-record /sealed/execution-manifest.json \
  --calibration-source-result /sealed/calibration-source-result.json \
  --processed-root \
    /mnt/lexar4tb/datasets/deform360_official_hub_visuotactile_v1/calibration-processed/aligned \
  --implementation-revision "$(git rev-parse HEAD)" \
  --output /sealed/deform360-calibration-visual-jobs-v1.json
```

The command publishes atomically without replacement. Exit code `0` means the
source-supported object gate remains satisfied. Exit code `2` is a contract or
publication failure; exit code `3` is reserved for a valid retained-failure
manifest with insufficient object support.

## Next execution stage

A runner must consume this manifest, verify every source-video byte identity,
check out the exact Prob4D and MotionCrafter revisions, resolve the exact model
set, and run the declared camera jobs without accessing the evaluation-only
future. Prediction outputs remain calibration-only until the complete
observability, calibration-bundle, and confirmation-authorization gates pass.

## Information and claim boundary

The manifest acknowledges hashing calibration camera payload bytes, but reports
that camera frames were not decoded, prediction outputs were not opened,
calibration target metrics were not computed, confirmation payloads were not
opened, target outcomes were not used, and replacement was not allowed.

A valid manifest establishes source-byte, frame-range, camera-accounting, and
stochastic-schedule provenance only. It does not establish provider competence,
physical-query benefit, tactile benefit, uncertainty calibration, deployment
safety, Causal4D benefit, or state of the art.
