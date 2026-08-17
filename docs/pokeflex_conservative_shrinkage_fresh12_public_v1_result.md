# PokeFlex Fresh-Take Checkpoint Comparison V1 Result

## Outcome

The unchanged source-selected Bayesian correction transfers positively on all
eligible members of the prospectively selected 12-take public cohort.
Object-balanced CD-UL1 falls from 4.940 mm for the released PokeFlex checkpoint
to 4.888 mm, a 1.043% improvement. Eleven objects improve and the object without
admitted updates uses exact checkpoint fallback and ties. The 97.5% paired
object-bootstrap upper bound for candidate minus baseline is -0.0329 mm. Every
locked target gate therefore passes.

| Public take | Checkpoint (mm) | Candidate (mm) | Improvement | Supported/scored |
| --- | ---: | ---: | ---: | ---: |
| `3dPrintedCylinder_T5` | 3.193 | 3.124 | 2.155% | 127/134 |
| `3dPrintedPizza_T6` | 2.266 | 2.261 | 0.218% | 59/64 |
| `3dPrintedPyramid_T4` | 2.012 | 1.986 | 1.283% | 61/65 |
| `Beanbag_T3` | 5.204 | 5.132 | 1.390% | 32/36 |
| `FoamCylinder_T3` | 5.542 | 5.429 | 2.052% | 70/75 |
| `FoamHalfSphere_T5` | 2.354 | 2.285 | 2.930% | 67/72 |
| `Pillow_T7` | 7.905 | 7.857 | 0.615% | 76/81 |
| `PlushDice_T1` | 6.361 | 6.288 | 1.140% | 66/71 |
| `PlushMoon_T5` | 6.307 | 6.275 | 0.499% | 64/69 |
| `PlushTurtle_T5` | 7.772 | 7.685 | 1.127% | 76/81 |
| `PlushVolleyball_T3` | 6.698 | 6.698 | 0.000% | 0/85 |
| `Sponge_T4` | 3.662 | 3.639 | 0.642% | 83/88 |

Across the cohort, 781 of 921 scored frames admit the frozen update. Every
unsupported frame retains the released checkpoint prediction exactly.

## Claim Boundary

This is a prospective paired improvement over the released PokeFlex checkpoint
on 12 previously unexamined public takes. It is not an unseen-object result:
source development used other takes of the same physical objects. It is also
not a direct reproduction of the published 18-take table. The candidate's
4.888 mm value is numerically below the published 6.498 mm reference, but the
cohorts differ, so that number is contextual and does not establish table SOTA.

Jaccard remains non-gating under the frozen protocol because the released OBJ
targets do not reliably satisfy the volumetric Boolean contract. No surrogate
metric was introduced after outcomes were opened.

## Custody

The storage-only amendment and prediction implementation were frozen at commit
`a93e88edd1a19e0ccaf6afdf9e0c9b4ba78c7cde`. Twelve ZIP archives were opaquely
staged on `gpuserver6000`, then 5,022,873,600 bytes were transferred directly
over the server LAN to `gpuserver4090`; the jump server was not in the data
path. Every transferred member was rehashed against its canonical stage
manifest before prediction.

All 12 predictions were sealed before target scoring. The complete prediction
barrier has canonical digest
`c25c35de41a7db693b519a05701e60ae3475d89391e950430a2c6f304ed34b7b`
and file SHA-256
`e9ba6abba60c3ac0b4bfb34003cc05053f940879996704c11ecaac8f6bdbff25`.
The exact target result has SHA-256
`3355769d8994ea70c421f4c009fb180a33bf0a72b7aaa8cf4efa37aecced902f`.

## Interpretation

The result provides independent evidence for a narrow Bayesian-PhysTwin claim:
a conservative, action-supported observation update improves an existing
physical predictor across new interactions while retaining an exact fallback.
The effect size is modest, but its direction is consistent across every object
where the update is admitted. This is stronger evidence than a large gain on a
few favorable cases because the locked no-regression and paired-transfer gates
both pass.

The next decisive evaluation is the official split or another independently
locked cohort with the same metric contract. Until then, the defensible wording
is "better than the released checkpoint on the prospective fresh-take cohort,"
not "new PokeFlex state of the art."
