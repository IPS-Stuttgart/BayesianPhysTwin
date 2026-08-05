# PokeFlex Final-Two Action Transfer V5 Result

## Outcome

The frozen action-robust scale improves both final previously unscored public
takes relative to the released PokeFlex checkpoint, but it does **not** pass the
preregistered advancement gate over the global `0.125` correction.

| Public take | Checkpoint | Global | Robust | Robust vs checkpoint | Robust vs global |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Pillow_T4` | 7.7571 mm | 7.7434 mm | 7.7469 mm | +0.1306% | -0.0460% |
| `PlushDice_T3` | 8.3544 mm | 8.3230 mm | 8.2192 mm | +1.6186% | +1.2477% |
| Object-balanced | 8.0557 mm | 8.0332 mm | 7.9830 mm | +0.9022% | +0.6242% |

Against the checkpoint, the robust arm wins 2/2 takes and its 97.5% paired
object-bootstrap upper candidate-minus-reference difference is `-0.01013 mm`.
Against the global correction, it wins only 1/2 takes; the minimum per-object
relative improvement is `-0.04599%`, and the upper bootstrap difference is
`+0.00356 mm`. The locked zero-regression and bootstrap conditions therefore
fail. No retuning from either outcome is authorized.

The frame-balanced values are 8.0308 mm for the checkpoint, 8.0090 mm for the
global correction, and 7.9634 mm for the robust scale. They are diagnostic; the
equal-weight physical-object result drives the registered gate.

## Interpretation

The experiment strengthens the narrower claim that the guarded state correction
transfers beyond its source actions: all eight prospective v3/v5 interactions
improve over the released checkpoint. It does not support the stronger claim
that the per-object maximin scale is uniformly better than the simpler global
scale. The Pillow miss is small, but it was exactly the failure mode the
no-regression gate was designed to expose.

The result points toward baseline-relative online admission or shrinkage rather
than more aggressive fixed per-object scaling. Such a selector must be developed
on source data and tested on fresh evidence; these opened outcomes cannot be
used to tune it.

## Custody

The protocol and implementation were pushed before either archive was staged at
commit `edd799ffaa3530e5b3641a6ff391afe7e766bee2`. The clean server PokeFlex
suite passed 211 tests with 3 skips. Both archives moved directly from
`gpuserver6000` (`129.69.102.145`) to `gpuserver4090` (`129.69.102.139`), with
no jumpserver in the payload path, and rehashed to their registered SHA-256
values.

Both prediction seals reported zero future-mesh reads. The complete barrier has
canonical digest
`2e927fe1f4df38f9bd6f63130fedf8397791215a3d936b6fad9a533b6324387c`
and file SHA-256
`0f8faa083e8c93f45d4caa966cf2a3928fb60ebc4a117b2dd446c50b417bf68f`.
Only after that barrier passed were target meshes decoded and scored. The target
result file SHA-256 is
`e575b56923d68daa9119ac431d43be3f6a2e1bb3ce35c1186f011f7ec80a60c0`.

The compact summary has canonical digest
`da35b35fa2b6bbf13f079be5fa6b5705320ac351dbb8cfab06d7a27ec9687112`
and file SHA-256
`06e872cd68753bea003ea2df41baf89d715d5a8d2ead55a8df9f76d68300bdd4`.

## Claim Boundary

These are prospectively held actions of two previously studied physical objects,
not unseen-object generalization. The cohort is not the published PokeFlex
validation split, whose five unavailable legacy take identifiers remain
unmapped. The published 6.498 mm value is therefore contextual only and no
direct table-SOTA claim is made from this result.
