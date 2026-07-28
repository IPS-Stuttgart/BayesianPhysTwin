# Direct RGB-D Sentinel V8 Source Result

## Decision

**The frozen V8 arm fails its disjoint hidden-source gate and is closed.** It
does not advance to additional source cases or fresh-object evaluation. No V1
sealed target, fresh target, or held-v8 artifact was read.

V8 is a one-case post-open source diagnostic, not transfer, confirmation,
calibration, or state-of-the-art evidence.

## Frozen Evidence

- Provider revision:
  `e776afc74804909dd2c7126e1e0069337f4b7d2c`
- Evaluator revision:
  `e9e7bb6280c026f2c46621af3dd4203cecb17e60`
- Source prediction digest:
  `d55a2183c1e9abe13b381bc9bcdf5a84b32a3f30a953d4f1dd08a04064dd3a86`
- Hidden audit digest:
  `4c4a584df7c830dd2d4b2f30bfc590eb350294f6d57767780e7b976aaabf94a2`
- Hidden audit file SHA-256:
  `85ddc74c29bba2c25f8ffa0b593b9a3d27d3b6a948c67152a3532d3720941cac`

## Provider Gate

The direct endpoint provider fixed the V5--V7 support failure:

- graph identities passing the endpoint support screen: 265/452;
- scheduled identities supported at both endpoints: 12/12;
- active identities supported: 9/9;
- sentinels supported: 3/3;
- sentinel common-bias gate: accepted;
- active pairwise gate: 9/9 inliers, accepted;
- selected future backbone: persistence;
- final update: nontrivial.

The observation contract also behaved as intended: assignment probability
remained separate from prior reliability, full local `(u,v,depth)` mixture
covariance was propagated into square metres, unknown view correlation used
covariance intersection, and between-view scatter was retained.

## Disjoint Hidden Result

All 12 queried identities were excluded. The untouched frames 58--75 were
scored on the remaining 440 material identities.

| Method | Hidden identity RMSE | Hidden symmetric Chamfer |
| --- | ---: | ---: |
| Physical prior | 19.444 mm | 14.568 mm |
| Persistence | **0.100 mm** | **0.080 mm** |
| Selected backbone | **0.100 mm** | **0.080 mm** |
| Direct RGB-D sentinel V8 | 1.505 mm | 2.552 mm |

Relative to persistence, V8 changes:

- hidden identity RMSE: **+1397.74%**;
- hidden symmetric Chamfer: **+3076.15%**.

The late hidden identity RMSE is 0.102 mm for persistence and 1.503 mm for V8.
The advancement gate therefore fails on both primary metrics.

## Interpretation

V8 separates provider coherence from predictive utility. The multiview depth
associations are available and mutually consistent, but this action-only
source window is almost exactly static in the hidden material identities.
Persistence leaves only about one tenth of a millimetre of error. Even the
strongly covariance-shrunk correction is therefore harmful.

This result closes the exact direct-depth endpoint arm and further tuning on
this opened case. It does not show that metric depth is useless in genuinely
dynamic windows. It shows that camera/depth consistency alone cannot establish
that a nonzero state update has lower regret than an already excellent
baseline.

The next credible route must admit updates using target-free evidence that the
object actually responded to the intervention, not merely that a geometrically
coherent observation exists. Required ingredients are:

1. physical and measured-action support for a dynamic response;
2. a latent shared observation-bias term;
3. source-calibrated baseline-relative regret control;
4. exact fallback when the update cannot beat persistence with sufficient
   confidence;
5. evaluation on genuinely fresh dynamic objects, not this exhausted source
   case.

This is consistent with the independent camera-only impossibility result and
the source-positive guarded online-belief direction. It is not permission to
retune or open any existing prospective cohort.

## Preserved Artifacts

All source prediction and disjoint hidden-audit artifacts are stored under:

`results/sota/diagnostics/deform360_direct_depth_sentinel_v8/059-shoe-ep0000/`
