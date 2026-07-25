# PokeFlex Independent-Depth Regret Guard v1

## Scope

This source-only study follows the frozen PokeFlex D405 validation. The D405
pair predicted candidate regret, but the zero-margin selector improved the full
source cohort by only 3.93% and regressed on `3dPrintedHeart` and `MemoryFoam`.
The registered 5%/4-object gate failed, so `T2`, calibration objects, and target
objects remained sealed.

The question here is narrower:

> Can source-calibrated upper bounds retain only D405-supported updates whose
> regret is likely below the unchanged released checkpoint?

This is method development on already opened sources. It is not confirmation or
a PokeFlex state-of-the-art result.

## Rejected alternatives

Two direct state-update variants were killed before this guard was fit.

1. Same-time D405 selection of the existing Kinect correction bank improved
   `FoamDice_T1` by 3.41%, but regressed `3dPrintedHeart_T1` by 7.82%. The arm
   selector was stopped after those two source-only kill tests.
2. A D405-only graph displacement, with per-sensor bias terms, correlation-aware
   information mass, action support, and exact fallback, regressed
   `FoamDice_T1` by 5.15%: 24 frame wins, 68 losses, and 7 fallbacks. It was not
   run across the cohort or tuned with a scale sweep.

The evidence therefore supports D405 as a candidate-regret sensor here, not as a
standalone metric state estimator.

## Guard

The guard preserves the frozen three-radius, four-nonzero-scale candidate bank.
For each candidate it uses only causal, target-independent features:

- worst and mean D405 regret plus cross-sensor disagreement;
- candidate radius and scale;
- Kinect graph-update magnitude and its relation to prior motion;
- temporal correction alignment;
- measured force and force change;
- robust-likelihood and assignment-spread diagnostics;
- static D405 calibration quality.

A group-weighted ridge model is fit on source takes. A 90% outer, 80% within-take
cross-fitted residual quantile forms a candidate regret upper bound. The
in-support candidate with the smallest bound is proposed. A second group bound,
fit to the residual induced by this minimum selection, corrects selector
optimism. The update is accepted only when the corrected upper regret is
strictly negative; otherwise the released checkpoint vertices are returned
byte-for-byte.

No target error enters a feature. Candidate innovations are processed only in
the existing robust registration likelihood.

## Source result

Leave-one-object-out evaluation used 25 opened source artifacts from five
objects (`T1`, the frozen `T3` design smoke interval, and `T4-T6`). Aggregation is
equal frames within take, equal takes within object, then equal objects.

| Method | Object-balanced CD_UL1 | Relative change | Object wins | Object losses |
| --- | ---: | ---: | ---: | ---: |
| Released checkpoint | 4.888 mm | reference | - | - |
| Regret-guarded update | 4.810 mm | **-1.60%** | 4/5 | 0/5 |

`MemoryFoam` is an exact object-level tie because the guard abstains. Across
accepted frames, 217 improve and 19 regress, for an 8.05% false-safe rate. The
cross-object candidate upper bound covers 85.36% of supported candidate rows;
this is compatible with the deliberately non-simultaneous 80% within-take
target and must not be described as 90% per-candidate coverage.

| Object | Baseline | Guarded | Relative improvement |
| --- | ---: | ---: | ---: |
| `3dPrintedHeart` | 4.696 mm | 4.696 mm | 0.01% |
| `FoamDice` | 5.742 mm | 5.512 mm | 4.01% |
| `MemoryFoam` | 2.412 mm | 2.412 mm | 0.00% |
| `PlushOctopus` | 5.364 mm | 5.211 mm | 2.85% |
| `ToiletPaperRoll` | 6.227 mm | 6.219 mm | 0.13% |

The source gate requires at least 1% object-balanced improvement, four object
wins, no object regression, at most 10% false-safe accepted frames, and at least
80% candidate-upper coverage. All checks pass.

## Prospective replication

Only three untouched reserved development archives exist for the source object
set:

- `FoamDice_T7`;
- `FoamDice_T8`;
- `PlushOctopus_T7`.

They are locked in
`configs/sota/pokeflex_independent_depth_regret_guard_prospective_v1.json`.
No archive member or outcome was read before the lock. The replication succeeds
only with at least 1% object-balanced improvement, improvement on both objects,
and no object regression. There is no replacement.

Because these are new takes of source objects, a positive result establishes
same-object temporal transfer only. Calibration and target objects remain
sealed, and a positive result still does not establish independent-object
generalization or beat the published PokeFlex benchmark.

## Prospective result

The locked replication passed every registered gate.

| Method | Object-balanced CD_UL1 | Relative change | Object wins | Object losses |
| --- | ---: | ---: | ---: | ---: |
| Released checkpoint | 6.418 mm | reference | - | - |
| Regret-guarded update | 6.222 mm | **-3.06%** | 2/2 | 0/2 |

All three take means improve:

| Take | Baseline | Guarded | Relative improvement |
| --- | ---: | ---: | ---: |
| `FoamDice_T7` | 4.244 mm | 4.231 mm | 0.30% |
| `FoamDice_T8` | 5.407 mm | 4.985 mm | 7.81% |
| `PlushOctopus_T7` | 8.010 mm | 7.835 mm | 2.19% |

The guard accepts 87 of 241 target frames and falls back exactly on 154. Among
accepted frames, 77 improve and 10 regress. This replication supports the
source-calibrated guard as a safer same-object temporal update, while leaving
the independent-object and published-SOTA questions open.

## Independent-object calibration result

The exact source deployment was then evaluated without refitting on the four
predeclared calibration objects. It failed the joint gate: object-balanced
CD_UL1 changed from 4.817 to 4.872 mm (**+1.16%**), only two of four objects
improved, `3dPrintedPyramid` regressed by 19.36%, and 28 of 71 accepted frames
were harmful. The eight target objects remain sealed.

The positive same-object result therefore does not establish independent-object
transfer. This guard is closed as a target candidate; calibration outcomes may
not be used to tune a successor claimed on this cohort.

## Evidence

- Source evaluation:
  `results/sota/pokeflex_independent_depth_regret_guard_source_v1/source_cross_object_evaluation.json`
- Prospective evaluation:
  `results/sota/pokeflex_independent_depth_regret_guard_prospective_v1/prospective_evaluation.json`
- Prospective protocol SHA-256:
  `be2bbf6f2e1ac1ce0a536bd02a09633d5607677fe3f7ce8d51cfd8e7d533c447`
- Independent-object calibration evaluation:
  `results/sota/pokeflex_independent_depth_regret_guard_calibration_v1/calibration_evaluation.json`
