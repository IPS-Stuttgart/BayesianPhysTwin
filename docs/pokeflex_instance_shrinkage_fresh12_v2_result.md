# PokeFlex Instance-Shrinkage Fresh12 V2 Result

## Outcome

The source-calibrated object-specific scale improves the released PokeFlex
checkpoint on the prospectively selected second 12-take cohort, but it does not
pass the locked advancement gate against the previously validated global scale.

| Frozen arm | Object-balanced CD-UL1 | Versus checkpoint | Wins/ties |
| --- | ---: | ---: | ---: |
| Released checkpoint | 6.266 mm | reference | - |
| Global scale 0.125 | 6.196 mm | 1.117% better | 11/1 |
| Object-specific scale | 6.158 mm | 1.723% better | 11/1 |

The object-specific arm is 0.612% better than the global arm in the aggregate,
with a 97.5% paired object-bootstrap upper bound of -0.00791 mm for candidate
minus global. However, `3dPrintedPyramid` regresses by 1.140% relative to the
global arm. The preregistered gate permits no per-object regression, so the
instance-specific method is rejected even though its aggregate change is
positive. The fixed map must not be retuned from this opened cohort.

| Public take | Checkpoint | Global | Instance | Instance vs checkpoint | Instance vs global | Supported/scored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3dPrintedCylinder_T1` | 3.487 | 3.355 | 3.264 | +6.390% | +2.722% | 92/97 |
| `3dPrintedPizza_T4` | 2.038 | 2.029 | 2.027 | +0.546% | +0.104% | 53/58 |
| `3dPrintedPyramid_T5` | 6.641 | 6.510 | 6.584 | +0.862% | -1.140% | 62/67 |
| `Beanbag_T1` | 5.966 | 5.896 | 5.836 | +2.188% | +1.022% | 51/56 |
| `FoamCylinder_T6` | 4.224 | 4.174 | 4.137 | +2.058% | +0.889% | 36/39 |
| `FoamHalfSphere_T6` | 2.387 | 2.314 | 2.278 | +4.552% | +1.528% | 76/81 |
| `Pillow_T3` | 8.329 | 8.251 | 8.207 | +1.462% | +0.537% | 80/86 |
| `PlushDice_T6` | 12.215 | 12.077 | 11.917 | +2.439% | +1.324% | 65/73 |
| `PlushMoon_T4` | 8.053 | 8.023 | 7.998 | +0.688% | +0.309% | 77/82 |
| `PlushTurtle_T6` | 7.045 | 6.997 | 6.953 | +1.302% | +0.625% | 58/63 |
| `PlushVolleyball_T6` | 7.051 | 7.051 | 7.051 | +0.000% | +0.000% | 0/77 |
| `Sponge_T5` | 7.762 | 7.681 | 7.650 | +1.436% | +0.398% | 91/96 |

Across the cohort, 741 of 875 scored frames admit the correction. The
volleyball take has no admitted update and is an exact fallback for both
correction arms.

## Interpretation

The experiment strengthens the narrow evidence for conservative online belief
updates: both frozen correction arms transfer over a second set of interactions
and improve every object on which they are admitted relative to the released
checkpoint. It does not establish that source-calibrated object-specific
shrinkage is preferable to one conservative global scale. The only authorized
conclusion for that comparison is a positive aggregate trend that failed the
registered no-regression gate.

This failure points to the missing ingredient without authorizing target-side
tuning: scale adaptation needs an online, baseline-relative regret guard or a
calibrated posterior over correction magnitude. A fixed object lookup table is
not robust enough. Any successor must be calibrated on source data and tested
on a new cohort; these 12 outcomes are now diagnostic-only.

The published 6.498 mm PokeFlex value remains contextual. This cohort differs
from the paper's 18-take validation split, so neither the 6.158 mm instance
number nor the 6.196 mm global number is a direct table-SOTA comparison.
Jaccard is non-gating because none of the released target meshes satisfy the
registered volumetric Boolean contract.

## Custody

The protocol and implementation were frozen at commit
`d51eca193ca1762b95c0802a1a428c09d036d92f`. The 12 opaque stages comprise
5,583 files and 5,416,488,098 bytes. They were transferred directly from
`gpuserver6000` to `gpuserver4090` over the server LAN and rehashed at the
destination; the jump server was not in the data path.

All predictions completed before target access. The barrier canonical digest is
`8feb255b279fb4749ce8acabeb67eef7f74edf8bdb4625ccf351bf02180a5693`
and its file SHA-256 is
`243ee31e5f1f8fd94d8d2fcb601f9ccb4a2afde1fd4527f017578d037d0a5e51`.
The target result file SHA-256 is
`defc3c8cf78ccdea0c5b4d03fcb5d3e983dff1a63c4c08b29111a004bfdb0d3b`.

## Decision

Do not advance the fixed instance-scale map and do not tune it on this cohort.
Retain the global 0.125 correction as the validated PokeFlex arm. The next
credible method experiment is a source-calibrated, baseline-relative guarded
scale update with exact fallback, followed by genuinely independent testing.
