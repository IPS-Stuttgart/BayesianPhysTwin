# AllTracker source-depth smoke result

Date: 2026-07-24

Status: failed development smoke; no larger PhysTwin-19 run is authorized.

## Question

This post-open smoke changed only the dense tracker used by the existing
source-camera RGB-D observation arm. The physical backbone, exact material
association, causal selector, robust endpoint filter, correction grid, and
manual-prefix prohibition were held fixed.

The three difficult cloth cases and the continuation gate were committed
before any AllTracker future score was produced. This is development evidence,
not independent transfer, calibration, or state-of-the-art evidence.

## Result

Lower is better. Values are the locked
`causal_selected_dense_relative_cap` future arm.

| Case | CoTracker3 CD (mm) | AllTracker CD (mm) | CoTracker3 track (mm) | AllTracker track (mm) |
| --- | ---: | ---: | ---: | ---: |
| `single_lift_cloth` | 11.984 | 12.601 | 50.730 | 47.386 |
| `single_lift_cloth_3` | 6.548 | 6.927 | 25.602 | 26.163 |
| `single_lift_cloth_4` | 9.351 | 10.176 | 51.920 | 53.124 |
| **Equal-case mean** | **9.294** | **9.902** | **42.750** | **42.224** |

Relative to CoTracker3, AllTracker changes:

- Chamfer distance by **+6.54%**;
- manual-track error by **-1.23%**;
- both metrics favorably in **0/3** cases;
- the worst case-metric by **+8.82%**.

The maximum-regression guard passes, but the two primary gates fail: both means
do not improve and there are no both-metric wins. The registered decision is:

```text
smoke_gate_passed: false
action: stop the AllTracker source-depth arm
```

No AllTracker quality, cycle, selector, cap, graph, or filter setting will be
tuned against these opened future outcomes, and the remaining PhysTwin-19
cases will not be run for this arm.

## Interpretation

AllTracker helps the identity-sensitive metric on one difficult case, but a
tracker substitution does not resolve the source-camera RGB-D arm's
surface-versus-identity tradeoff. All three Chamfer scores regress, while two
of three track scores also regress.

This rejects only the fixed **AllTracker plus source-depth lifting** interface.
It does not reject AllTracker as a two-dimensional correspondence source. In
particular, the separate Deform360 pairwise-consensus field uses cross-track
agreement and guarded fallback rather than treating one source-camera depth
sample as the full metric observation.

Together with the CoTracker3 multiview result, the smoke narrows the next
method: new camera evidence must alter the belief-update semantics, model
shared bias, and preserve exact fallback. Replacing the tracker while keeping
the same metric lift is insufficient.

## Information boundary

- RGB decoding stopped at each released training endpoint.
- Future cue rows were neutralized.
- No future RGB or future object observation formed a prediction.
- Released manual tracks were used only after prediction for evaluation.
- No held-v8 or PokeFlex target artifact was accessed.
- All three cases were already-open development interactions.

## Provenance

- preregistration commit: `329b11d`
- protocol SHA-256:
  `be8ce678e00277ade2a95610cca014f670fa2a153e70c1f91bbabe46a06c78b6`
- AllTracker candidate result SHA-256:
  `d0d65741505a9574dcd831e0af718a6890de75d99fbbef759a35b67f11acd606`
  (`gpuserver6000:/mnt/corsair/florianpfaff/bpt-phystwin-alltracker-smoke-v1/alltracker_source_depth_smoke_result.json`)
- CoTracker3 comparator SHA-256:
  `42606b9d9e17085a68f417eba79bd3747c5c8aac49134777a533159d09da7f5e`
  (`gpuserver6000:/home/florianpfaff/bpt-cotracker3-multiview-priority-v1/source_depth_result.json`)
- compact mechanical result SHA-256:
  `557b900a706332adfc4134e862f802d3c93d24d4afcdbb63bcb42ddadc03975d`
