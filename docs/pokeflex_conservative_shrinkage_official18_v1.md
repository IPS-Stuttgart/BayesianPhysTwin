# PokeFlex exact official-split evaluation v1

## Purpose

This protocol asks whether the unchanged source-selected conservative Bayesian-PhysTwin
update improves the released PokeFlex Kinect checkpoint on the exact 18 recordings
hard-coded in upstream `test/evaluate.py`.

The prior prospective T2 evaluation improved CD by 1.046% on eight independently locked
objects, but it could not support a direct published-SOTA claim: that cohort differed from
the paper's split, and the released target meshes made boolean-volume Jaccard undefined.
This protocol resolves the split mismatch without replacing or repairing the public target
meshes.

## Evidence boundary

The official split contains 18 exact object-take IDs. Exact-ID provenance search found
three prior development overlaps: `FoamDice_T3`, `PlushOctopus_T6`, and
`ToiletPaperRoll_T1`. They remain in the exact-split result but are labeled exploratory.
The other 15 exact takes form the prospective paired-transfer cohort. No take may be
replaced.

All 18 predictions must be sealed at one clean implementation revision before any of the
15 prospective deformed target meshes are opened. Prediction at frame `f` may use Kinect
and robot history only through `f-1`. Missing action-field inputs force a byte-identical
released-checkpoint fallback.

## Registered comparisons

The primary exact-split score is global equal-frame one-sided L1 Chamfer distance using
10,000 deterministic surface samples in metric coordinates. Object-balanced scores are
reported as diagnostics.

Three gates must all pass:

1. The released checkpoint reproduces the paper's `6.498 mm` Kinect reference within 5%.
2. The candidate exact-split global CD is below `6.498 mm`.
3. On the 15 prospective takes, candidate CD improves the object-balanced baseline, the
   97.5% object-bootstrap upper bound on candidate-minus-baseline CD is below zero, and no
   prospective object regresses. Exact fallbacks may tie.

The public evaluator's volumetric Jaccard is retained as a non-gating diagnostic. Invalid
boolean operations are reported with their valid fraction; no repaired-mesh or voxel
surrogate is substituted. Consequently, this experiment can establish a directly
comparable CD result, not a replacement Jaccard result.

## Claim boundary

If all gates pass, the supported statement is that the unchanged source-selected update
sets a lower reproducible public official-split CD than the published Kinect reference and
transfers prospectively on 15 exact takes. The three disclosed overlaps prevent describing
the full 18-take result as wholly prospective. If baseline reproduction fails, the result
is a paired transfer study only and no direct SOTA claim is authorized.
