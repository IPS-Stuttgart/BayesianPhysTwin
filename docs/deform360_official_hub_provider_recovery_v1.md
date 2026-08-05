# Deform360 official-Hub provider recovery v1

## Why this is a recovery lock

The calibration payload was acquired before the visual-provider and finite-group
amendments were committed. The immutable Stage 1 correction records that order.
Consequently, `Deform360VisualProviderLockV1`, which requires unopened selected
payloads, is invalid for this campaign and is intentionally left unchanged.

The recovery lock makes a narrower, still useful promise: the exact visual
producer, causal processing window, and all output-affecting settings are frozen
before any calibration score, provider comparison, calibration-policy fit,
confirmation payload, or target outcome is opened. Confirmation therefore remains
prospective, while calibration-method selection must be described honestly as
post-payload and pre-score.

## Frozen producer

The content-addressed lock is
`protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_recovery_v1.json`.
It binds:

- Prob4D provider API 2 at revision
  `364f216c14f7770c1b360bb1b836b11ecf0c18b8`;
- deterministic MotionCrafter at revision
  `1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257`;
- immutable MotionCrafter UNet/VAE revision
  `fc7b18d5657184607bf4501b02d64ada7540b4e3` and Stable Video Diffusion
  revision `9e43909513c6714f1bc78bcb44d96e733cd242aa`;
- derived-per-call seeds rooted at `20260805`, five inference steps, guidance
  scale 1, 320x640 images, float32 storage, and low-memory inference;
- decoded-uniform overlap fusion, fixed-grid stride-4 sampling, sequential full
  joint gauge covariance, canonical covariance roots, analytic composition
  Jacobians, and no pointwise covariance fallback;
- one first-frame metric prior under the separately content-addressed Sim(3)
  policy, with no later metric anchors; and
- rank-64 gauge covariance retaining at least 99.9% of covariance trace.

The claim-bearing provider attestation depends on gauge and point covariance
calibration IDs. It is therefore generated during calibration and must be bound
by the pre-confirmation calibration bundle; it is not fabricated in this earlier
lock.

Stage 1 retained 32 calibrated cameras per object. The transitive execution lock
`deform360_official_hub_visuotactile_v1_visual_execution_lock_v1.json` also binds
a three-camera policy before any image value or score is used: select the camera
triple with the largest minimum spherical baseline, then largest total baseline,
then lexicographically smallest name tuple. Cross-view factors retain a shared
camera-bias nuisance and use equal-weight covariance intersection because their
correlation is unknown. The later calibration bundle must bind this execution-lock
ID, not merely the lower-level provider-recovery ID.

## Frozen causal window

The event clock exactly follows the official processing rule: over taxel rows
0-11, a gripper is active when its paired sensors jointly contain at least two
strictly positive processed taxels. The first active frame is the contact start.
Search stops at that frame; later tactile values cannot alter the boundary.

For contact frame `t_c`:

```text
causal cutoff (exclusive) = t_c + 6
observed source interval  = [cutoff - 42, cutoff)
future evaluation         = [cutoff, cutoff + 24)
```

The 42 observed frames produce two independently decoded 25-frame MotionCrafter
windows with eight overlapping frames. The complete official-processing view is
therefore 66 contiguous frames. An episode with insufficient history, no detected
contact, or insufficient future is retained as a technical failure without
replacement or boundary adjustment.

The first-contact index is a shared scheduling variable for every arm. Tactile
magnitudes beyond that one event bit enter only tactile-labelled methods. This
keeps visual-only and visuotactile comparisons on identical observed and future
frames, while making the shared scheduling information explicit.

## Remaining boundary

This lock is not evidence that Prob4D is competent on Deform360, that tactile
anchors improve the physical update, that raw covariance is calibrated, or that
the method beats a state of the art. Calibration must still process all ten
locked objects, retain failures, fit the registered artifacts, and seal the exact
calibration bundle before confirmation data can be accessed.
