# Deform360 tactile contact-geometry smoke v1

## Result

The frozen source-only geometry gate passed for the same calibration smoke case
used by the causal robot-prefix experiment. Across the six permitted contact
frames `[144, 150)`, the extractor found 46 active taxels. All tactile values
after the cutoff remained untouched, as did calibration prediction scores,
confirmation payloads, target outcomes, and held-v8 artifacts.

| Quantity | Result |
|---|---:|
| Active frames | 6 / 6 |
| Active taxels | 46 |
| `tactilel_left` / `tactilel_right` | 2 / 0 |
| `tactiler_left` / `tactiler_right` | 4 / 40 |
| Assignment separation, minimum | 91.3 mm |
| Assignment separation, median | 108.7 mm |
| Assignment separation, maximum | 128.9 mm |
| Geometry admission | pass |

This is a contact-geometry feasibility result, not a tactile measurement update
or a prediction result. In particular, 40 of the 46 active taxels belong to
`tactiler_right`, so silently choosing the wrong gripper assignment would move
most of the available evidence by about 9--13 cm.

## Assignment uncertainty

The public sensor names identify the `tactilel` and `tactiler` groups, but the
released interface does not bind those groups to recovered marker-ID grippers
0 and 1. The artifact therefore preserves two hypotheses:

| Hypothesis | Mapping | Prior probability |
|---|---|---:|
| `direct` | `tactilel` to gripper 0, `tactiler` to gripper 1 | 0.5 |
| `swapped` | `tactilel` to gripper 1, `tactiler` to gripper 0 | 0.5 |

Both hypotheses remain in the artifact. No hard assignment was selected from a
state residual or future prediction outcome.

## Frozen execution

- Lock ID: `9f3fb26568d4bf9269ad35ce792ebd8739cd397d82f806b63b265f54f42879f9`
- Geometry implementation: `af100b02c8a71e9c61ea8c48b8e73241236916a4`
- Runtime revision: `97b19a897a75a554dab27a0212665512dcf2f477`
- Upstream Deform360 revision: `d8522a4403b766aeb387510c04e89032a56fdf35`
- Parent robot-prefix artifact: `555943b0614e211e9c4d0fb5e2928c5499d695f5d6a020b6d26c2646d536297e`
- Contact-geometry artifact: `4a00b4297b6ec8a61da6f1fbffcfe4c61787a3a1fcb90e70f0204eb665e76dc8`
- NPZ SHA-256: `992fc413d85211ca8d47e910149f20701048ab70886c5bd45c068cc3794e4fc0`
- Manifest SHA-256: `e9ea31591baf60bf232db207c42cb36e612c65b3a341e7d6c46c2f9a11e902c0`

An independent second execution reproduced both the NPZ and manifest byte for
byte.

## Decision

The result authorizes development of a source-only association and covariance
feasibility gate. It does not authorize treating taxels as object displacement
anchors, opening calibration scores, or making a prediction claim. Before any
scoring, the next frozen method must associate the gripper-mounted taxels with
object hypotheses without using future outcomes, preserve the gripper-assignment
mixture, and calibrate metric covariance for the resulting sparse evidence.
