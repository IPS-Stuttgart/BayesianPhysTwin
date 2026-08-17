# PokeFlex all-object robust-scale public-13 retrospective result

## Outcome

The source-locked all-object scale map improves the strongest prior PokeFlex
configuration when replayed on all thirteen publicly materializable
official-validation takes.

| Method | Frame-balanced CD-UL1 (mm) | Object-balanced CD-UL1 (mm) |
| --- | ---: | ---: |
| Released Kinect checkpoint | 6.56942 | 6.79037 |
| Global correction, scale 0.125 | 6.49932 | 6.71972 |
| V3 repeated-action robust scale | 6.44785 | 6.66390 |
| V4 all-object robust scale | **6.40344** | **6.62464** |

V4 improves object-balanced CD-UL1 by **2.44%** relative to the released
checkpoint, **1.42%** relative to the global correction, and **0.59%** relative
to V3. Relative to the checkpoint, it records 12 object wins, one exact tie,
and no losses. The 97.5% paired object-bootstrap upper bounds for V4 minus the
checkpoint and V4 minus the global correction are `-0.11048 mm` and
`-0.05503 mm`, respectively.

The archived checkpoint and global frame scores reproduce exactly, with maximum
drift `0.0 mm`. V4 is better than the global correction on ten objects, ties on
two, and is slightly worse on one. The exception is `3dPrintedBunny_T1`, where
V4 scores `3.63318 mm` against `3.63022 mm` for the global arm, while still
improving the `3.68103 mm` released-checkpoint result.

The V4 frame-balanced value of `6.40344 mm` is numerically `0.09456 mm` below
the published `6.498 mm` Kinect reference. This is useful evidence of headroom,
not a direct state-of-the-art comparison, because the cohorts differ.

## Information Boundary

The V4 multipliers were frozen from repeated source actions before this replay.
No parameter was selected from these thirteen target outcomes. Nevertheless,
all thirteen outcomes had already been opened by prior methods, so this analysis
is retrospective and cannot independently confirm transfer.

The five officially listed takes absent from the licensed public release remain
unopened and unreplaced:

- `Pillow_T8`
- `3dPrintedCylinder_T7`
- `3dPrintedHeart_T14`
- `Sponge_T10`
- `3dPrintedPizza_T13`

No future or missing official-take artifact and no Deform360 held-v8 artifact,
process, identity, query, score, barrier, or outcome was accessed.

The authorized claim is:

> On all thirteen publicly materializable official PokeFlex validation takes,
> the source-locked all-object robust scale map improves the released checkpoint
> and the prior V3 robust configuration in retrospective replay, with no
> object-level loss relative to the checkpoint.

Calling this an independent confirmation, a complete official-18 evaluation, or
a direct defeat of the published `6.498 mm` result is not authorized.

## Provenance

- Implementation revision:
  `6af891b882782e5c5f099dd1610c49f772866445`
- V4 calibration canonical SHA-256:
  `e94eeb9bdd2cc69e245b0bd48d843e5f64cb039e1eb02841e4a784cbe4dbc880`
- V4 calibration file SHA-256:
  `00cdf5732f5dbf7eb0f899ebbb536260d9e66c0a151b41eec81ffaaef4aaf110`
- Archived V3 result file SHA-256:
  `619c46726aab0f7e81d2e943bd44820e521c9fe6285906add28af87203c15ebd`
- Retrospective result file SHA-256:
  `9d6a3ce6e4d606485dcecfb12418199dc4bd3bbf43236e2d42f3f25f94a98a0e`

The result contains 13 objects, 970 scored frames, and 835 frames with supported
updates. Every archived checkpoint and global frame score was re-evaluated before
the V4 aggregate was accepted.

## Decision

Retain V4 as the preferred PokeFlex guarded-update configuration. Freeze an
official-18 V4 protocol now, while the five unavailable outcomes remain unseen.
The direct comparison should run only after the PokeFlex authors supply the exact
five takes or a checksummed processed validation bundle. Until then, report V4 as
the strongest retrospective public-subset result and keep the published-reference
comparison explicitly numerical rather than confirmatory.
