# PokeFlex Repeated-Action Robust Scale Fresh6 V3 Result

## Outcome

The frozen repeated-action maximin scale passes every preregistered target gate
on the six prospectively selected third interactions. It improves all six takes
relative to both the released PokeFlex checkpoint and the globally validated
`0.125` correction.

| Frozen arm | Object-balanced CD-UL1 | Versus checkpoint | Wins |
| --- | ---: | ---: | ---: |
| Released checkpoint | 5.771 mm | reference | - |
| Global scale `0.125` | 5.712 mm | 1.025% better | 5/6 |
| Repeated-action robust scale | 5.642 mm | 2.240% better | 6/6 |

The robust arm is 1.227% better than the global arm, wins all six paired
comparisons, and has a 97.5% paired object-bootstrap upper bound of
`-0.01643 mm` for candidate minus global. Against the checkpoint, the
corresponding upper bound is `-0.05560 mm`. The smallest per-object relative
improvements are positive against both references, so the locked no-regression
conditions pass.

| Public take | Checkpoint | Global | Robust | Robust vs checkpoint | Robust vs global | Multiplier | Supported/scored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3dPrintedCylinder_T6` | 3.584 | 3.483 | 3.368 | +6.018% | +3.296% | 3.0 | 110/115 |
| `3dPrintedPizza_T1` | 2.276 | 2.279 | 2.270 | +0.242% | +0.367% | 0.5 | 57/62 |
| `Beanbag_T7` | 4.687 | 4.642 | 4.637 | +1.082% | +0.118% | 4.0 | 30/35 |
| `FoamCylinder_T4` | 4.780 | 4.708 | 4.684 | +2.008% | +0.507% | 3.0 | 70/75 |
| `Pillow_T5` | 7.992 | 7.923 | 7.878 | +1.416% | +0.557% | 2.0 | 77/83 |
| `PlushDice_T7` | 11.309 | 11.238 | 11.014 | +2.606% | +1.993% | 4.0 | 58/66 |

Across the cohort, 402 of 436 scored frames admit the update. Unsupported
frames remain exact released-checkpoint fallbacks. Jaccard is non-gating: none
of the released target meshes satisfies the preregistered volumetric Boolean
contract.

## Interpretation

This is the first prospective evidence that adapting the guarded correction
magnitude from repeated actions is more reliable than either the released
checkpoint or one global correction magnitude. The result also exhibits the
intended protection mechanism. The global correction slightly regresses on
`3dPrintedPizza_T1`, while the source-maximin multiplier `0.5` improves that
take relative to both references. On the opened source panel,
`3dPrintedPyramid` had conflicting preferred scales; the rule therefore chose
multiplier one, exactly preserving the global arm instead of repeating the
earlier object-specific regression.

The claim remains narrow. The six outcomes are untouched interactions of
previously studied physical objects, not unseen-object generalization. The
experiment changes only the magnitude of an existing guarded state correction;
it does not validate a new observation model or uncertainty-calibration claim.
The cohort also differs from the published eighteen-object split, so the
published `6.498 mm` value is contextual and no direct table-SOTA claim is
authorized.

The result justifies a larger independent preregistered evaluation. That next
evaluation should use genuinely fresh physical objects or a separately locked
official split, preserve the global and checkpoint controls, and freeze the
scale-selection rule without using these six outcomes for retuning.

## Custody

The protocol and implementation were frozen at commit
`7882fc449e33f12a577ad2cfcec3d24651bfba79`. Before target access, the full
suite passed 1,784 tests with 29 skips, and the changed-file Ruff check passed.

The six opaque stages comprise 2,865 files and 2,711,727,518 bytes. They were
transferred directly from `gpuserver6000` to `gpuserver4090` over the server
LAN, rehashed at the destination, and the one-use transfer credential was
removed. The jump server was not in the payload path.

All predictions completed before target access. The barrier canonical digest is
`3db692dfadb646fda183d78ba55461f18386211a66517380172f71509515003e`
and its file SHA-256 is
`7a0fdf988edfc81ffb80675d363430d45265c717fc7f9b0dcb25a832be5434cb`.
The target result file SHA-256 is
`1e8fcae19d618d52a05762ebd039e92098b52725459bf8d320124fffcaead204`.

The compact summary canonical digest is
`3bdf93d6f939d87a4ad971d64589095553fd3df59b4016767479b664ba0fb945`
and its file SHA-256 is
`41dae21e0c5d92bb79835da92fb7fa5e479bae52b667eac4b48757d2fdb01b9b`.
The scoring-provenance canonical digest is
`6efb36c8f2ccf25efd3d667803c39b01dcc9ee88eae6b7b1c5f8329a8e8ea326`
and its file SHA-256 is
`1e3e149a8eada32750a05dd1ba28a2b4eb8688773b2f61409cb4d0d83ac8d5c4`.

## Decision

Advance the repeated-action maximin scale as the preferred PokeFlex guarded
state-update arm for further independent evaluation. Keep the global `0.125`
correction and released checkpoint as mandatory controls. Do not tune the map
or gate from this opened fresh6 panel.
