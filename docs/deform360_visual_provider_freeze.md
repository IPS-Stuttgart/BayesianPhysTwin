# Deform360 visual-provider freeze

## Purpose

The official-Hub Deform360 study has a metadata-only object/episode selection,
but selected calibration payloads may not be opened until one exact visual
producer is frozen. This step resolves that producer without reading a Deform360
dataset.

The reviewed command is:

```bash
python scripts/science/build_deform360_visual_provider_freeze.py \
  --spec protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json \
  --metric-prior-policy \
    protocols/locks/deform360_official_hub_visuotactile_v1_metric_frame_prior_policy.json \
  --prob4d-checkout /exact/Prob4D \
  --motioncrafter-checkout /exact/MotionCrafter \
  --cache-dir /exact/huggingface/hub/cache \
  --output-dir /new/output/directory
```

It accepts no dataset, confirmation, geometry, tactile, robot, or target path.

## Frozen choices

The preflight specification fixes:

- `IPS-Stuttgart/Prob4D` at
  `25d90ef7f78ba4307f4555cb636d666004e1bf66`;
- `TencentARC/MotionCrafter` code at
  `9cb4e9679f5f34e249945544052464ef46324bc2`;
- provider API 2 with analytic Sim(3) composition Jacobians and canonical
  covariance roots;
- the deterministic MotionCrafter model family;
- a root seed of `20260805` with per-object derived seeds;
- 25-frame windows, overlap 8, resolution 320 by 640, and `float32` storage;
- the complete joint gauge covariance with rank cap 64 and at least 99.9%
  retained covariance trace;
- the first retained causal frame as the mandatory per-object metric-frame prior;
  and
- no additional metric anchors for the primary arm.

The metric-frame policy is itself content-addressed. It fixes the information
source and frame rule, while each object's numerical prior remains a later
calibration or observation artifact. This avoids pretending that the metric
initialization is a monocular no-anchor method.

## Cache resolution

Every model source must already exist as a complete Hugging Face snapshot. The
preflight recognizes:

1. MotionCrafter UNet;
2. MotionCrafter geometry/motion VAE;
3. the nested image VAE; and
4. the Stable Video Diffusion base pipeline.

Snapshot directory names must be exact 40- or 64-character revisions. A missing,
incomplete, or ambiguous cache fails closed. The UNet and geometry/motion VAE
must resolve to the same MotionCrafter model revision.

The script then uses Prob4D's `PinnedMotionCrafterModelSet` to derive the portable
model-set identity. It emits a provider-v2 manifest and an independently
validated exploratory pre-calibration attestation. Calibration IDs are
deliberately absent at this stage; they are bound by the later Stage-1
calibration execution seal.

## Output

A successful run atomically publishes:

```text
provider-manifest.json
provider-attestation.json
motioncrafter-model-set.json
visual-provider-lock.json
summary.json
SHA256SUMS
```

When a committed bundle is present, `--expected-bundle-dir` requires byte-exact
agreement for every substantive JSON file. This turns the self-hosted run from a
candidate generator into an independent reproduction of the reviewed lock.

The workflow
`.github/workflows/deform360-visual-provider-freeze.yml` runs hosted contract
tests and the data-free preflight on `workstation2`. It has read-only repository
permissions, checks out exact source revisions, records environment identity,
and uploads only compact producer evidence.

## Information and claim boundary

This preflight does not open any selected Deform360 camera, tactile, robot,
reconstruction, depth, tracking, point-cloud, control-point, calibration,
confirmation, or target payload. It supplies the last producer-identity
prerequisite before the registered 10-object calibration execution. It is not
evidence of provider competence, tactile benefit, physical-query improvement,
predictive calibration, Causal4D benefit, deployment safety, or state of the art.
