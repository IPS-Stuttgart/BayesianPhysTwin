# PokeFlex action-robust public official-subset result

## Outcome

The frozen repeated-action robust scale rule passed every registered gate on the
thirteen publicly materializable PokeFlex official-validation takes.

| Method | Frame-balanced CD-UL1 (mm) | Object-balanced CD-UL1 (mm) |
| --- | ---: | ---: |
| Released Kinect checkpoint | 6.56942 | 6.79037 |
| Global correction, scale 0.125 | 6.49932 | 6.71972 |
| Repeated-action robust scale | **6.44785** | **6.66390** |

Relative to the released checkpoint, the robust arm improves object-balanced
CD-UL1 by **1.86%**, with 12 wins and one exact fallback tie. The 97.5% paired
object-bootstrap upper bound for candidate minus checkpoint is `-0.08028 mm`.

Relative to the global correction, it improves by **0.83%**, with six wins and
seven ties. The corresponding bootstrap upper bound is `-0.02037 mm`. No object
regresses. Five ties are required global fallbacks for objects absent from the
source scale calibration; the remaining ties use a calibrated multiplier of one
or have no supported update.

The archived global arm reproduces exactly at `6.4993172114797195 mm`, satisfying
the `1e-9 mm` drift control. The robust frame-balanced value is numerically below
the published `6.498 mm` Kinect value.

## Claim boundary

This is a strong public-benchmark result, but not a direct reproduction or defeat
of the published full split:

- the public archive materializes 13 of the 18 officially listed validation takes;
- five missing takes receive no replacements;
- all 13 outcomes had been opened previously under the fixed-scale method;
- the robust-scale evaluation is therefore retrospective;
- the published `6.498 mm` number aggregates a different 18-object cohort.

The authorized claim is:

> On all thirteen publicly materializable official PokeFlex validation takes, the
> independently source-calibrated repeated-action robust correction improves both
> the released checkpoint and the frozen global correction under the registered
> evaluator, without an object-level regression.

Calling this a prospective confirmation or full official-split state of the art is
not authorized.

## Execution

Predictions were generated from causal observations through `f-1` on
`gpuserver4090` at clean commit
`a9e38c287d31330cd16b7148dfc4a96f888befe5`. All thirteen seals passed before
scoring. The 803,068,398-byte prediction and barrier payload was transferred over
the direct server LAN to `gpuserver6000`; source and destination canonical tar
digests both equal
`2cbe15a92d0cae9493903bb43157509f8e95532147ede4e1e374fe37ff05f130`.
No payload traversed the jump server.

## Evidence

- Protocol canonical SHA-256:
  `fe3199f72822ff384ce5e304c0afa85c7913f973055aaae06b88d62bdbc49349`
- Barrier canonical SHA-256:
  `f855ba6fd08931d5c0bf039ee49194a1046284f65101f2c920cb32b6c1be1da2`
- Target result file SHA-256:
  `619c46726aab0f7e81d2e943bd44820e521c9fe6285906add28af87203c15ebd`
- Summary canonical SHA-256:
  `a7797cb5b318cb54e84c3cee16f33206a39fed81d5de0090409e2dc4ca00e6cf`
- Provenance canonical SHA-256:
  `a6ab77204215e1e7691cdf34399b28d6db753e278392796e3d83c02b29541835`

The full raw result, compact summary, barrier, provenance, and prediction seals are
stored in
`results/sota/pokeflex_action_robust_official13_public_v1/`.

## Recommendation

Use this as the strongest reproducible public official-subset evidence in the
Bayesian-PhysTwin paper. The next direct-comparison upgrade requires either the
five missing official takes from the PokeFlex authors or a new independently
locked public cohort shared by both methods. Do not tune this rule further on the
opened thirteen takes.
