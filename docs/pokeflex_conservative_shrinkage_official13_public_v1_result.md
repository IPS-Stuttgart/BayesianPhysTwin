# PokeFlex public official-subset result v1

## Outcome

The frozen Bayesian state update passes every registered prospective transfer gate on
the publicly materializable subset of the official PokeFlex validation split. On the ten
prospectively untouched exact takes, object-balanced CD_UL1 falls from 6.6262 mm for the
released checkpoint to 6.5569 mm, a 1.046% improvement. Nine takes improve and
`PlushVolleyball_T4` uses exact fallback and ties. The 97.5% paired object-bootstrap upper
bound for candidate minus baseline is -0.0438 mm.

Across all 13 publicly available official takes, including three disclosed development
overlaps, equal-frame CD_UL1 falls from 6.5694 to 6.4993 mm, a 1.067% improvement. Twelve
takes improve and one ties.

| Take | Checkpoint (mm) | Candidate (mm) | Improvement | Supported/scored |
| --- | ---: | ---: | ---: | ---: |
| MemoryFoam_T2 | 11.601 | 11.473 | 1.103% | 55/60 |
| PlushVolleyball_T4 | 8.029 | 8.029 | 0.000% | 0/70 |
| FoamHalfSphere_T3 | 3.711 | 3.610 | 2.721% | 99/106 |
| 3dPrintedBunny_T1 | 3.681 | 3.630 | 1.380% | 74/80 |
| 3dPrintedPyramid_T6 | 2.227 | 2.154 | 3.280% | 60/64 |
| FoamDice_T3* | 5.138 | 5.060 | 1.517% | 89/94 |
| PlushMoon_T1 | 6.694 | 6.673 | 0.304% | 69/74 |
| PlushOctopus_T6* | 4.992 | 4.951 | 0.830% | 67/72 |
| PlushDice_T8 | 5.480 | 5.439 | 0.763% | 83/92 |
| PlushTurtle_T3 | 10.274 | 10.213 | 0.595% | 52/56 |
| Beanbag_T6 | 8.213 | 8.124 | 1.081% | 50/55 |
| FoamCylinder_T1 | 6.353 | 6.225 | 2.018% | 55/60 |
| ToiletPaperRoll_T1* | 11.882 | 11.776 | 0.891% | 82/87 |

`*` marks a disclosed development-overlap control. These three takes do not enter the
prospective gate.

## Claim boundary

This result supports the following claim:

> The unchanged source-selected Bayesian state update improves the released PokeFlex
> checkpoint on ten prospectively untouched exact takes from the public official-validation
> subset, with no take-level regression and exact fallback when unsupported.

It is not a direct state-of-the-art comparison with the paper's published 6.498 mm
official-18 aggregate. Five exact official take IDs are absent from the public archive and
receive no replacements. The numerically close public-13 candidate value of 6.4993 mm is
therefore not compared statistically or descriptively as if it came from the same cohort.

Boolean-volume Jaccard is non-gating under the frozen protocol. All 970 baseline and
candidate evaluations raised `ValueError: Not all meshes are volumes!`; no mesh repair or
voxel surrogate was substituted.

## Custody and verification

All 13 predictions were generated at clean pre-outcome revision
`f794ceff8421bc584a3d7c2e41bddcb01b87c16e` and sealed before any campaign target mesh was
opened. The barrier has canonical digest
`0eb52c17606d3e9684248987cf5404d8368eab00325d2decd8304aa069ef9324` and file SHA-256
`caa72b62f99b836d386d96b668797564f26d4b4629552ca8aec45e0caa71651c`.

The sealed 510,237,568-byte prediction bundle moved directly from `gpuserver4090` to
`gpuserver6000` over their shared LAN in 52 seconds; the jump server was not in the data
path. Only after revalidation were 970 force-active target meshes extracted and scored on
CPU. Independent recomputation reproduced the aggregate exactly and revalidated the
barrier. The exact result SHA-256 is
`0bc3a16f2f91a556d624b1b45fa8aca4063736085d474ee889ff99018901dbe4`.

The focused protocol/result suite passes 19 tests and scoped Ruff is clean. A clean Linux
Git checkout passes 1,379 of 1,380 repository tests under the frozen NumPy 1.26.4 runtime;
the sole remaining test exercises a private namespace available only in NumPy 2 and passes
separately under NumPy 2.4.5.

Three nonresult infrastructure attempts remain disclosed in `execution_manifest.json`:
an invalid first staging caused by shell parsing of zero-padded frames, a partial relay
transfer, and a prediction environment missing two imports. None created a prediction or
score used by this result; each was quarantined rather than overwritten.

## Recommendation

The prospective consistency, negative bootstrap bound, and exact unsupported fallback
justify carrying this method into the Bayesian-PhysTwin paper as independent transfer
evidence. A larger preregistered run is justified only on a cohort with an authoritative
mapping for all official takes or on a newly locked benchmark where the candidate and
reference are evaluated on exactly the same cases. This result should not be stretched
into an official-18 SOTA claim.
