# Official-Hub MotionCrafter dependency amendment v2

## Status

This is a pre-inference runtime-dependency amendment for the ten-object
official-Hub Deform360 calibration cohort. It does not change the selected
objects, camera panels, causal windows, untouched futures, seeds, provider
products, or evaluation policy.

The first smoke under the v1 model-set lock stopped during model construction.
No MotionCrafter prediction completed, no provider output was admitted, no
calibration score was opened, no calibration policy was fit, and no
confirmation or target artifact was accessed.

## Failure found

At MotionCrafter revision
`1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257`, the geometry-motion VAE creates
an image VAE through a hard-coded `from_pretrained` call. The v1 model-set
manifest bound the MotionCrafter UNet, geometry VAE, and SVD base pipeline, but
did not bind this nested Stable Diffusion image VAE. Offline execution therefore
failed closed before inference.

## Amendment

Prob4D revision `8aa18c9f2aca3ef089adc3a23c06643e1c4cd79f` adds the
image VAE to the content-addressed MotionCrafter model set and intercepts the
upstream hidden load. The adapter replaces the mutable repository, cache, and
revision arguments with the frozen source below and rejects upstream call-shape
drift:

- repository: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- role: `image-vae`

The amended model-set identity is
`466f5197722a0c77e5d5e0b70b110d302484e162b6e926c84f7b610c1c4df775`.
The Bayesian-PhysTwin runner now passes all four pinned sources to
`PinnedMotionCrafterModelSet.inspect`.

## Artifact lineage

The original causal-window manifest remains immutable:

- manifest ID:
  `9fe5fdf4ae6449182d2e5064ad99417b4252dd04b76831df722b44614c2351dd`
- file SHA-256:
  `7398575f32ea8868f241da9356264d5b44b815f41425f6f259afcbbd10f336de`

The amended provider recovery lock is
`protocols/locks/deform360_official_hub_visuotactile_v2_visual_provider_recovery_v1.json`.
It binds the original provider lock as `amendment_parent` and records the
pre-inference chronology. Its artifact ID is
`ce48b2a8823b81cd697f63355b84a5cd48ee3e0024022b782417e832947240ea`.

The dependency-complete assets are:

- `protocols/locks/deform360_official_hub_visuotactile_v2_motioncrafter_model_set.json`
- `protocols/locks/deform360_official_hub_visuotactile_v2_prob4d_provider_manifest.json`
- `protocols/locks/deform360_official_hub_visuotactile_v2_visual_provider_recovery_v1.json`

The earlier v2 job manifest remains the record of the failed pre-inference
attempt. A new job manifest must bind a post-amendment Bayesian-PhysTwin commit
and the amended runner hash before inference resumes. It must use a fresh output
root; the failed attempt may not be resumed under the new dependency set.

## Claim boundary

This amendment establishes runtime completeness and provenance only. It is not
evidence of provider competence, covariance calibration, physical-twin
improvement, confirmation performance, or state-of-the-art performance.
