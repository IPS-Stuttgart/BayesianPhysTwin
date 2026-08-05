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

The earlier v2 job manifest remains the record of the first failed
pre-inference attempt. Its replacement was
`protocols/locks/deform360_official_hub_visuotactile_v3_motioncrafter_jobs.json`.
It binds:

- Bayesian-PhysTwin implementation
  `55982e89596ce8a19af977d2d9924d3f7e210809`;
- runner SHA-256
  `62fdb997ebfcf30ec2906117a02a31cf14777678a023225db70149626c417052`;
- manifest ID
  `8cf8df7629d4f2a17ec4d5dcb992a65fca638acb8420a7cca79a91c5ecb80682`.

That second smoke loaded the pinned image VAE and then stopped before inference
because the current Hugging Face resolver treated the exact selective SVD cache
as an incomplete full-repository snapshot. Completing the repository would have
downloaded roughly 31 GB of unused alternative weights. No provider prediction
or calibration score completed.

Prob4D revision `25d90ef7f78ba4307f4555cb636d666004e1bf66` now resolves an
exact-revision local snapshot only when all seven base-pipeline members consumed
by MotionCrafter are present. Otherwise it retains the exact remote revision and
fails closed when offline. The source identity remains the pinned SVD revision;
the cache path is only a runtime resolution of that source.

The current dependency-complete assets are:

- model-set ID
  `b072956636612ca1a31d1edb83bd7d1bd27b8962cb617c6e615b9b310a16de6e`;
- provider-manifest ID
  `112e1c9debe1d947b6352193497053ed8d1528b7c0a755b26888e16a7bc74ba3`;
- provider-recovery-lock ID
  `d3d68423a6c1a19e5cab5651ef4f08921652757fc565539227c06bd8e7dfcbce`.

The v2 and v3 job manifests remain immutable pre-inference failure records. The
active replacement is
`protocols/locks/deform360_official_hub_visuotactile_v4_motioncrafter_jobs.json`,
with manifest ID
`4a6b7ab8547b88a5332928c30bd52bbc1f74e91e42efbbb8ae5e453c2078fd5c`.
It binds Bayesian-PhysTwin implementation
`7085d4722bb61a93ca1c8632ffca5790f965e444` and the unchanged runner SHA-256
`62fdb997ebfcf30ec2906117a02a31cf14777678a023225db70149626c417052`.
Execution must use a fresh output root.

The v4 smoke completed one provider job, and the complete-cohort continuation
sealed a second job. The third camera job then failed during geometry-motion VAE
decoding because the shared adapter retained 8.67 GiB of unallocated CUDA cache
between independent videos. The incomplete v4 report remains an immutable
provider-runtime record: two of 30 jobs completed, calibration scores remained
sealed, no policy was fit, and no confirmation or target artifact was accessed.

The runner now performs Python garbage collection and `torch.cuda.empty_cache()`
in a `finally` block after every independent camera job, including resumed jobs
and failed jobs. This is an inter-job memory-lifecycle correction only: it does
not change model weights, source frames, windows, seeds, inference parameters,
provider products, or output validation. The next job manifest must bind the new
runner digest and use a fresh output root; v4 must not be resumed under the
amended runtime.

That replacement is
`protocols/locks/deform360_official_hub_visuotactile_v5_motioncrafter_jobs.json`,
with manifest ID
`202ac2b16e91a35538e3f61daae4017b6582a6bcef6a83ecac90674832136ac2` and
file SHA-256
`1c368d4a6ff9163c80b3831821e10193907fbd18ea2d0ab7ebc33f2e369fd8c2`.
It binds Bayesian-PhysTwin implementation
`4d662a3c48d063edaa420e9a10b94b365422c3f0` and runner SHA-256
`56a9dc023692d2bdfc73cd60d624c4a9405145691222c5c0c6545a37e3f68d22`.

The v5 smoke completed, but the complete-cohort continuation again stopped on
the third camera during VAE decoding. Explicit garbage collection and CUDA cache
release reduced the retained allocation but did not remove non-releasable
allocator state from the shared model process. Two of 30 jobs completed; scores,
policy fitting, confirmation payloads, and target outcomes remained sealed.

The next runtime isolates every independent camera job in a child process. The
parent verifies the frozen schedule and each content-addressed prediction, while
process exit provides a complete CUDA-context teardown before the next camera.
Already complete predictions are hash-verified directly, and partial jobs resume
only inside their own isolated worker. This changes orchestration only and must
be bound by a new job manifest and fresh output root; v5 remains immutable.

The process-isolated schedule is
`protocols/locks/deform360_official_hub_visuotactile_v6_motioncrafter_jobs.json`,
with manifest ID
`9726e7ae12d442956ff81376fe52cdc2f8360fdcd3e5cccbc12543ca584b30f9` and
file SHA-256
`b9302a27d779a6de619baffc04e624eee629a226a140b90278fa9dd06b213fe2`.
It binds Bayesian-PhysTwin implementation
`5b7f1c60546b814c1c34e56db397e4a0877dd36f` and runner SHA-256
`2d3fe74d7d76ff1cf5766781879056ddfabaa1e5e1842ae776fdd01514846e65`.

## Claim boundary

This amendment establishes runtime completeness and provenance only. It is not
evidence of provider competence, covariance calibration, physical-twin
improvement, confirmation performance, or state-of-the-art performance.
