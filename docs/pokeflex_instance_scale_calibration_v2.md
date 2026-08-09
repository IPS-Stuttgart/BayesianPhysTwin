# PokeFlex Instance Scale Calibration V2

## Question

Can one completed interaction calibrate the magnitude of the Bayesian
observation correction for a later interaction of the same physical object?

The fresh12 scale audit showed that the correction direction transfers broadly,
but its useful magnitude varies by object. A globally larger scale improves the
mean while introducing material regressions. This protocol therefore changes
only one scalar per physical object and leaves the correction field, causal
inputs, support gate, released checkpoint, and exact fallback unchanged.

## Source calibration

The opened fresh12 interactions are now development data. For each physical
object, the calibration chooses the source-action minimum from multipliers
`0.5`, `1.0`, `1.5`, and `2.0` of the globally validated `0.125` scale. The bank
was capped before the next target cohort was selected. Exact score ties prefer
the multiplier closest to one. If the source interaction had no admitted
update, the multiplier remains one.

This is an empirical-Bayes hyperparameter update: an earlier interaction of the
same object supplies the instance-level scale, while the globally validated
scale remains the prior/default. It is not an online target-outcome selector.

| Physical object | Source take | Multiplier | Effective scale |
| --- | --- | ---: | ---: |
| `3dPrintedCylinder` | `T5` | 2.0 | 0.2500 |
| `3dPrintedPizza` | `T6` | 0.5 | 0.0625 |
| `3dPrintedPyramid` | `T4` | 0.5 | 0.0625 |
| `Beanbag` | `T3` | 2.0 | 0.2500 |
| `FoamCylinder` | `T3` | 2.0 | 0.2500 |
| `FoamHalfSphere` | `T5` | 2.0 | 0.2500 |
| `Pillow` | `T7` | 2.0 | 0.2500 |
| `PlushDice` | `T1` | 2.0 | 0.2500 |
| `PlushMoon` | `T5` | 2.0 | 0.2500 |
| `PlushTurtle` | `T5` | 2.0 | 0.2500 |
| `PlushVolleyball` | `T3` | 1.0 | 0.1250 |
| `Sponge` | `T4` | 1.5 | 0.1875 |

The source calibration has canonical digest
`74c2f5fe6b57215fdebedd18cc31cb1b4bca010aac905b1c91f185fb34b10390`
and file SHA-256
`bfde9f3572b694d4dffe008b889d45dccea888162886a307fd3b96cfd6b475f3`.

## Prospective boundary

The next evaluation must use another untouched take of every object. The scale
mapping is frozen before target archive extraction. Each frame continues to use
only Kinect and robot history through `f-1`; target-frame depth, RGB, force, and
mesh geometry remain forbidden prediction inputs. Unsupported updates return
the released checkpoint prediction byte for byte.

The previous fresh12 outcomes may justify this method, but they cannot support
its transfer claim. Only the next all-case sealed evaluation can do that.
