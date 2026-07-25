# Deform360 frame-zero initializer source result

## Scope

This is a target-free geometry-coverage result on 12 previously opened and
exhausted Deform360 objects, balanced across filament, sheet, and volumetric
strata. It does not use future RGB, dense future geometry, tracks, or outcome
metrics. It is not a predictive-accuracy result and does not alter the failed
prospective bias-aware belief experiment.

The source candidate is revision `c7752cf` with config digest
`64f72fe964b61e5283c1acd88c3910807695036608c8a01836d9e5bdf565c759`.
The full audit is
`results/sota/deform360_frame_zero_initializer_source_v1/source_audit.json`
(file SHA-256
`eb4de9fdd001e3ad802dcd2ce10d0a536049a10daf35ed13ed3adfa0f82c1f02`,
canonical result SHA-256
`7f22378af2f458d0026ebc627be0df0e28f5a0937ac41b06e31ddf72c8d8c4b7`).

## Candidate

The selector keeps an admitted Splatfacto material cloud exactly. It evaluates
the fallback only when the frozen point-only check fails: at least 128 finite
3D points.

The fallback uses frame zero only:

1. voxelize the fixed one-metre world cube at resolution 120;
2. require support from at least seven calibrated views and at least 55% of
   the peak view count;
3. retain the largest 26-connected component;
4. extract its six-neighbour surface shell;
5. deterministically Morton-subsample to at most 10,000 material points; and
6. color the retained points from their supporting frame-zero views.

The seven-view threshold was selected on this open source panel. It remains
strictly redundant, while the prior eight-view choice left only ten connected
voxels for the open frog case. This is source development, not confirmation.

## Result

| Source diagnostic | Result |
| --- | ---: |
| Cases / physical objects | 12 / 12 |
| Strata | 4 filament / 4 sheet / 4 volumetric |
| Original clouds admitted at 128 points | 9 / 12 |
| Exact admitted-path parity | 9 / 9 |
| Natural sub-128 failures | 3 / 12 |
| Natural failures recovered | 3 / 3 |
| Forced fallback constructions | 12 / 12 |
| Minimum largest-component fraction | 48.83% |
| Median symmetric distance to original cloud | 31.97 mm |
| Maximum symmetric distance to original cloud | 61.56 mm |

All 12 cases pass the locked source gate. Every retained fallback point meets
the selected strict multiview vote threshold, and no accepted original path
changes a byte.

The distance to the original cloud is deliberately diagnostic. Values of
32--62 mm are too large to infer that the hull is an equivalent or more
accurate reconstruction. The result establishes robust, auditable geometric
support and natural failure recovery only. Predictive utility still requires
an opt-in physical-backbone test and then a genuinely fresh-object evaluation.

## Decision

Freeze this candidate before any new-object use. The next valid steps are:

1. run the frozen fallback on the four already opened calibration quality
   failures as a post-open operational diagnostic, without reading outcomes;
2. test automatic graph construction and zero-action Warp stability for the
   fallback clouds;
3. retain exact Splatfacto geometry whenever the original gate passes; and
4. only if the physical-backbone checks pass, register a fresh-object protocol
   before downloading or inspecting another cohort.

The existing prospective calibration rejection, the reserved target seal, and
the frozen Causal4D claims remain unchanged.
