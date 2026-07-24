# Bias-aware AllTracker multiview ray smoke v1

## Status

This is a locked, post-open development smoke on three previously examined
PhysTwin cloth interactions. It is not independent evidence and does not
support a state-of-the-art claim.

The preregistered arm failed every transfer gate and is stopped. Its ray,
affine-bias, graph, cap, and admission settings must not be tuned against these
future outcomes.

## Method

The arm used exact archived material identities and AllTracker RGB prefixes
from three calibrated views. It:

1. inferred a robust 3D correction directly from image rays relative to the
   physical trajectory;
2. required simultaneous support from all three cameras;
3. used temporal effective sample size and conservative cross-view precision
   averaging instead of independent-pixel confidence accumulation;
4. projected out one shared affine displacement field as an unidentifiable
   camera/frame nuisance and added its magnitude to metric variance;
5. graph-smoothed the remaining correction using metric variance in square
   metres;
6. capped the applied correction at 20 mm; and
7. applied an exact zero-update fallback unless an independent held-out prefix
   point cloud improved by at least 1% and 0.1 mm with no early- or late-prefix
   Chamfer regression.

All settings and cue hashes were frozen before future metrics were read in
`configs/sota/phystwin_alltracker_multiview_ray_bias_aware_smoke_v1.json`.

## Information boundary

- RGB was decoded only on `[0, train_end)`.
- No future RGB, point cloud, or manual track built or selected a prediction.
- Manual tracks were evaluation-only.
- The independent admission signal was a released object point cloud inside
  the allowed prefix.
- The physical backbone retained the known future controller action.
- No held-v8 or sealed PokeFlex artifact was accessed.
- The broad runner's other future candidates were not inspected. A locked
  analyzer emitted only the declared arm, unchanged baseline, and frozen
  CoTracker source comparator.

The first analyzer invocation stopped before metric access because the frozen
comparator contained a superset of the three protocol cases. A mechanical,
recorded parser amendment changed exact-set equality to subset inclusion; no
method or gate changed.

## Result

| Case | Prefix decision | Future CD vs baseline | Future track vs baseline |
|---|---:|---:|---:|
| `single_lift_cloth` | admit | +12.57% | -5.63% |
| `single_lift_cloth_3` | exact fallback | 0.00% | 0.00% |
| `single_lift_cloth_4` | admit | +9.64% | +11.91% |
| Equal-case mean | 2 admit / 1 fallback | **+8.74%** | **+1.88%** |

Only one of three cases won or tied on both metrics. The maximum case-metric
regression was 12.57%. The candidate mean was also worse than the frozen
CoTracker3 source-depth arm on both metrics:

| Arm | Future CD | Future track |
|---|---:|---:|
| Raw physical baseline | 21.230 mm | 46.378 mm |
| Bias-aware AllTracker ray arm | 23.086 mm | 47.248 mm |
| Frozen CoTracker3 source-depth arm | 9.294 mm | 42.750 mm |

All five preregistered gates failed.

## Interpretation

Three-view redundancy, correlation-aware covariance, robust ray likelihood,
affine nuisance removal, and an independent prefix point-cloud gate are not
sufficient to make a persistent camera-derived correction transfer to the
future. Prefix Chamfer improvement was especially misleading:

- `single_lift_cloth` improved prefix Chamfer by 7.14% but worsened future CD
  by 12.57%;
- `single_lift_cloth_4` improved prefix Chamfer by 1.63% but worsened both
  future metrics.

The negative result strengthens the current model boundary:

> Camera evidence can support a geometrically coherent correction without
> establishing that the correction is the future physical state discrepancy.

The unresolved ambiguity is temporal and causal, not just geometric. A useful
next update must model how discrepancy evolves with action/contact and must
calibrate baseline-relative regret across independent source sessions. Another
camera-only persistent-field threshold search on these cases is not justified.

## Decision

Do not run this arm on the remaining opened PhysTwin cohort. Preserve exact
fallback and the observation artifact API, but move method development toward
one of:

- a time-varying Bayesian state/discrepancy update with action and contact
  support, calibrated on independent source sessions;
- an independent sensing modality that breaks coherent camera bias; or
- the separately strong open Deform360 AllTracker association result under its
  fresh-object prospective protocol.

The compact, permitted result is archived at
`results/sota/diagnostics/phystwin_alltracker_multiview_ray_bias_aware_smoke_v1/result.json`.
