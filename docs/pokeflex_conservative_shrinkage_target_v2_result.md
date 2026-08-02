# PokeFlex Conservative Shrinkage Target v2 Result

## Outcome

The source-selected correction transfers prospectively on the registered public
PokeFlex `T2` cohort. Object-balanced CD_UL1 falls from 5.691 mm for the released
checkpoint to 5.631 mm, a 1.046% improvement. Seven objects improve and the one
object without the required `T_WE` stream uses exact fallback and ties. The
97.5% object-bootstrap upper bound for candidate minus baseline is -0.0271 mm.
The paired transfer gate therefore passes.

| Object | Checkpoint (mm) | Candidate (mm) | Improvement | Supported/scored |
| --- | ---: | ---: | ---: | ---: |
| 3dPrintedCylinder | 3.878 | 3.744 | 3.455% | 56/59 |
| 3dPrintedPizza | 2.083 | 2.072 | 0.548% | 55/60 |
| FoamHalfSphere | 2.975 | 2.916 | 1.971% | 83/88 |
| Pillow | 7.830 | 7.801 | 0.369% | 73/78 |
| PlushDice | 8.940 | 8.795 | 1.613% | 83/88 |
| PlushTurtle | 6.845 | 6.782 | 0.923% | 63/68 |
| PlushVolleyball | 9.257 | 9.257 | 0.000% | 0/96 |
| Sponge | 3.717 | 3.681 | 0.971% | 107/112 |

## Gate accounting

The candidate CD_UL1 of 5.631 mm is numerically 13.34% below the published
6.498 mm PokeFlex reference. The comparison uses a different, prospectively
registered public split, so it is a metric-reference result rather than an
identical-split SOTA result.

The full direct metric-reference gate fails. All 649 candidate and baseline
Jaccard evaluations raised `ValueError: Not all meshes are volumes!`, giving a
registered valid fraction of zero rather than the required one. The locked rule
therefore reports failure and does not substitute a voxel approximation.

A post-result diagnostic found that the first scored released target OBJ for
all eight objects is non-watertight and non-volumetric. Seven corresponding
prediction meshes are volumetric; PlushTurtle's prediction is also
non-volumetric. This identifies a released-target/evaluator contract mismatch,
but it does not retroactively repair the failed gate.

## Custody

Version 1 stopped after seven prediction seals when the registered Volleyball
robot stream exposed a systematic missing-`T_WE` schema. No target mesh had
been opened. Version 2 locked missing required action metadata to exact
checkpoint fallback, preserved the original cohort, and regenerated all eight
predictions at clean commit `118135a50ca9a73c12752c80fab266f661ce3b8b`.

The eight-case barrier has digest
`7b9ae64a9f4d9118bfda55e00c53efe41e9c8603b3e8b28f18bdd37ea23ce20b`.
Only after it passed were 649 force-active target meshes extracted and scored.
The exact target result has SHA-256
`eed0b72d7aa80cdf232fdbf98152cf76a675f2f80ed7adc7039d3b086829ed16`.

## Interpretation

This is credible independent evidence that conservative, action-local Bayesian
state shrinkage improves the released PokeFlex physical prior. The effect is
consistent but small. It supports the Bayesian-PhysTwin observation-update
direction; it does not establish a full PokeFlex SOTA claim because the public
split differs and the preregistered Jaccard gate failed.

Any future direct SOTA protocol must preflight the metric contract on source
meshes before selecting fresh targets. It should either reproduce the exact
official CSG environment on admissible volumetric meshes or preregister a
different published-compatible metric before outcomes are opened.
