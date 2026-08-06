# PokeFlex instance-shrinkage fresh12 v2 protocol

## Question

The first prospective public-take experiment showed that a globally conservative
online correction scale of 0.125 improves the released PokeFlex checkpoint by
1.043% on an object-balanced basis without an object regression. A post-open
diagnostic found modest additional scale headroom, but no larger uniform scale
was safe across objects.

This protocol tests the next narrow hypothesis: can one correction multiplier
calibrated on an already-open interaction of a physical object transfer to a
different, previously unexamined interaction of the same object?

## Frozen method

- The observation, association, action-local correction field, and exact
  unsupported-frame fallback are unchanged.
- The base correction scale remains 0.125.
- Each object's multiplier is selected once from `{0.5, 1.0, 1.5, 2.0}` using
  only its opened source interaction.
- The frozen calibration is
  `configs/sota/pokeflex_instance_scale_calibration_v2.json`, canonical SHA-256
  `74c2f5fe6b57215fdebedd18cc31cb1b4bca010aac905b1c91f185fb34b10390`.
- There is no target-outcome adaptation or online scale selector.

Each sealed prediction contains three trajectories computed before target mesh
access:

1. the released checkpoint;
2. the previously validated global 0.125 correction;
3. the source-calibrated instance correction.

Unsupported updates return the released checkpoint bit for bit in both
correction arms.

## Freshness and custody

The 12 target take IDs are selected deterministically from the 20 public takes
remaining after excluding the original exposure union and the first prospective
fresh12 cohort. They are new interactions, not new physical objects. The frozen
freshness audit is
`configs/sota/pokeflex_instance_fresh12_exclusion_audit_v2.json`, canonical
SHA-256
`b9afe9cb4fe3f1e6b07a919ecd7fa204093308993b2b3cbf33887862d58c348e`.

Prediction and scoring remain separate. All 12 predictions must be sealed from
histories ending at frame `f-1` before any scored frame-`f` mesh is decoded.
There is no replacement of a failed take.

## Gates

The instance arm must pass independently against both references:

- positive object-balanced CD-UL1 improvement;
- 97.5% object-bootstrap upper bound on the paired difference below zero;
- no per-object regression.

The released checkpoint comparison is the primary transfer test. Advancement
over the global 0.125 arm establishes whether instance calibration adds value.
The published 6.498 mm PokeFlex aggregate is contextual only because this cohort
is not the paper's 18-take validation split.

## Claim boundary

A positive result supports source-calibrated, object-specific update magnitude
for repeated interactions of known objects. It does not establish transfer to
unseen objects, reproduce the published aggregate, or authorize target-adaptive
scale selection. A failed advancement gate retains the globally conservative
0.125 method as the supported result.
