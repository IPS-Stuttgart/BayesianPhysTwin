# PokeFlex baseline-relative guard public paired v2 result

## Outcome

The frozen guard improves the object-balanced mean but fails the complete
registered transfer gate. `CD_UL1` falls from 5.5693 mm to 5.5486 mm, a 0.3704%
improvement. The 97.5% object-bootstrap upper bound for candidate-minus-baseline
error is -0.00532 mm. Ten objects contain at least one scored admitted update.

The breadth result is 9 wins, 2 exact fallback ties, and 1 loss. The single
loss, `Sponge_T3`, is 0.0141%, but the frozen gate permits no regression and
requires at least 10 wins. `paired_transfer_passed` is therefore false.

| Take | Checkpoint (mm) | Guarded (mm) | Improvement | Admitted/scored |
| --- | ---: | ---: | ---: | ---: |
| `3dPrintedCylinder_T3` | 5.986 | 5.906 | 1.334% | 31/84 |
| `3dPrintedPizza_T5` | 2.138 | 2.133 | 0.220% | 9/66 |
| `3dPrintedPyramid_T1` | 2.400 | 2.362 | 1.567% | 21/62 |
| `Beanbag_T4` | 6.668 | 6.666 | 0.017% | 1/60 |
| `FoamCylinder_T7` | 5.559 | 5.473 | 1.541% | 28/39 |
| `FoamHalfSphere_T4` | 2.474 | 2.450 | 0.995% | 26/97 |
| `Pillow_T6` | 9.310 | 9.310 | 0.000% | 0/86 |
| `PlushDice_T5` | 6.817 | 6.805 | 0.175% | 9/70 |
| `PlushMoon_T3` | 7.556 | 7.555 | 0.004% | 8/80 |
| `PlushTurtle_T1` | 7.679 | 7.676 | 0.029% | 3/76 |
| `PlushVolleyball_T5` | 6.749 | 6.749 | 0.000% | 0/83 |
| `Sponge_T3` | 3.497 | 3.497 | -0.014% | 1/85 |

The candidate mean is numerically below the published 6.498 mm Kinect number,
but that number uses an unavailable internal split. It remains non-gating
cross-split context and does not support an official-table SOTA claim.

## Custody

The first target-free smoke run at `22b0a57e` exposed a staging schema error
before producing a prediction archive or seal: `volucam` calibration had been
paired with `realsense` depth. No target mesh was opened. Amendment `1185d9fb`
stages matching released Kinect depth and calibration and passed all 151 native
PokeFlex tests. The original tagged attempt remains unchanged.

All 12 amended predictions were then sealed at `1185d9fb`. They contain 801 raw
supported frames, 150 guard admissions, zero fallback mismatches, and zero
future-mesh reads. The complete barrier has canonical SHA-256
`26aaed737db34e65796b855977afdfdc432146aed43cfd93da6cab29b88e2a8e`.
Only after it validated were target meshes opened and scored once.

Both payload transfers used the direct LAN between `gpuserver6000` and
`gpuserver4090`; the jump server did not relay data. The target result SHA-256
is `aa2680cbe0d7a6c9e342c9093ff4045e25a6952191fc9033376516d468329685`.

## Interpretation

This is stronger evidence than v1 that a causal Bayesian state correction can
improve the released physical checkpoint on average: the mean and paired
bootstrap criteria transfer to a second fresh-take cohort. It is still a
negative confirmation under the declared uniform-safety claim. The calibrated
guard admitted one harmful scored frame and abstained completely on two
objects, leaving only nine object wins.

These 12 outcomes are now open development evidence. They must not be used to
alter the claim on this cohort. A further confirmation needs genuinely fresh
physical objects or another dataset; selecting another take from the same
opened PokeFlex objects would not provide independent object-level evidence.
