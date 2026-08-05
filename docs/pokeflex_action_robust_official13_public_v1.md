# PokeFlex action-robust public official-subset protocol

## Purpose

This protocol asks whether the repeated-action robust correction scale that passed
the fresh six-object transfer experiment also improves the thirteen exact official
validation takes available in the public PokeFlex archive.

It is the strongest executable comparison to the official benchmark without
private data. It is not the full published split: five officially listed takes are
absent from the public archive and are neither substituted nor imputed.

## Evidence boundary

All thirteen target outcomes were opened previously for the fixed global-scale
experiment. This evaluation is therefore retrospective for the robust scale rule.
No take is represented as prospectively untouched.

The new execution still enforces a complete prediction barrier. All thirteen
prediction archives must be generated from observations through frame `f-1`,
sealed at one clean implementation revision, and validated before this rerun reads
any target mesh. That barrier protects the new computation from target adaptation;
it does not erase the historical exposure.

The published Kinect value of `6.498 mm` covers eighteen objects. A public-subset
score below that number is useful contextual evidence but does not authorize a
direct full-split or state-of-the-art claim.

## Frozen method

The physical prior, graph-registration field, causal observation window, and base
scale are unchanged. The candidate uses the source-calibrated two-action maximin
multiplier for each object represented in that calibration. An object absent from
the calibration receives multiplier one, exactly reproducing the global `0.125`
scale for that object.

The thirteen-object multiplier map is:

| Object | Multiplier | Effective scale | Source status |
| --- | ---: | ---: | --- |
| MemoryFoam | 1 | 0.125 | global fallback |
| PlushVolleyball | 1 | 0.125 | calibrated |
| FoamHalfSphere | 2 | 0.25 | calibrated |
| 3dPrintedBunny | 1 | 0.125 | global fallback |
| 3dPrintedPyramid | 1 | 0.125 | calibrated |
| FoamDice | 1 | 0.125 | global fallback |
| PlushMoon | 4 | 0.5 | calibrated |
| PlushOctopus | 1 | 0.125 | global fallback |
| PlushDice | 4 | 0.5 | calibrated |
| PlushTurtle | 4 | 0.5 | calibrated |
| Beanbag | 4 | 0.5 | calibrated |
| FoamCylinder | 3 | 0.375 | calibrated |
| ToiletPaperRoll | 1 | 0.125 | global fallback |

## Controls and gates

The rerun carries three references:

1. The released PokeFlex checkpoint.
2. The unchanged global `0.125` correction reconstructed from the same prediction
   bundle.
3. The archived public-13 global score of
   `6.4993172114797195 mm`, which must reproduce within `1e-9 mm` before the new
   candidate is interpreted.

The robust arm must improve over both paired physical references with a negative
97.5% object-bootstrap upper bound and no per-object regression. It must also be
numerically below `6.498 mm` on the public subset. The last condition is an
advancement threshold only; it does not make the incomparable published aggregate
gating.

## Claim if the gates pass

The authorized claim is:

> On the thirteen publicly materializable official PokeFlex validation takes, the
> independently source-calibrated repeated-action robust correction improves both
> the released checkpoint and the previously validated global correction under the
> registered evaluator.

The phrases "official eighteen-object reproduction", "prospective confirmation",
and "published-split state of the art" remain unauthorized.

## Registered artifacts

- Protocol: `configs/sota/pokeflex_action_robust_official13_public_v1.json`
- Protocol canonical SHA-256:
  `fe3199f72822ff384ce5e304c0afa85c7913f973055aaae06b88d62bdbc49349`
- Scale calibration: `configs/sota/pokeflex_action_robust_scale_v3.json`
- Scale-calibration canonical SHA-256:
  `78d3c74e4246ec6b69cbcfe113ed04324bf1a9f49d543194df8a7a87d7f09157`
