# PokeFlex Action-Guard Development Result

## Status

The prospectively frozen action guard did not pass its primary development
transfer gate. Calibration and target objects remain sealed. Development takes
`T2`, `T7`, and `T8` also remain unopened for this method.

The candidate was locked in
`configs/sota/pokeflex_action_guard_development_v1.json` at commit `2863afa`
after inspecting only the five `T3` pilot takes. It used a local state field
supported jointly by the measured tool trajectory and an accepted two-view
Bayesian registration update. The field gain was fixed to `0.125` below 15 N
and `0.5` at or above 15 N. Unsupported updates returned the released
checkpoint vertices exactly.

One bookkeeping defect interrupted the first prospective command before a
score artifact was written or printed. Commit `ccde94e` fixed only the
insertion of unrequested fallback candidates. The candidate equations,
thresholds, gains, and evidence boundary did not change. All reported artifacts
were generated after that registered fix.

## Prospective Result

The frozen method was evaluated on 1,418 causal target frames from 20 previously
unopened takes: `T1`, `T4`, `T5`, and `T6` for each of the five development
objects.

| Object | Released checkpoint CD_UL1 (mm) | Guarded CD_UL1 (mm) | Change |
| --- | ---: | ---: | ---: |
| 3dPrintedHeart | 3.695 | 3.958 | +7.14% |
| FoamDice | 6.034 | 5.647 | -6.41% |
| MemoryFoam | 2.350 | 2.288 | -2.63% |
| PlushOctopus | 5.585 | 5.480 | -1.88% |
| ToiletPaperRoll | 5.581 | 5.310 | -4.85% |
| **Object-balanced** | **4.649** | **4.537** | **-2.41%** |

The method passed the locked object-win gate, with four of five objects
improving, and stayed within the 10% maximum object-regression bound. It failed
the primary requirement of at least 5% object-balanced improvement. Therefore
no calibration outcome and no sealed target outcome may be opened for this
candidate.

The development baseline and candidate means are below the published PokeFlex
6.498 mm Kinect reference, but this is not a direct SOTA result. The object split
differs, Jaccard has not been evaluated, and the paired 5% transfer gate failed.

## Post-Open Diagnostics

The force threshold did not transfer because force magnitude is not a reliable
proxy for update regret. In particular, the strong update helped the `T3`
Heart pilot but harmed the four prospective Heart takes.

Replacing the fixed 60 mm influence radius with an object-relative radius
improved stability. The best fixed post-open candidate, radius `0.7` times the
object radius and scale `0.25`, improved all five objects but reached only 3.12%
object-balanced improvement. Its per-frame candidate oracle reached 8.59%, so
better arm selection has headroom, but the opened data cannot confirm such a
selector.

A causal camera-only selector was also tested diagnostically. Each candidate
predicted at frame `t-1` was scored against both depth views when frame `t`
became observable. The score was baseline-relative, voxel clustered, and
combined conservatively across views. Across 15,816 arm-frame pairs:

- camera-score versus hidden CD-regret Spearman correlation was `0.042`;
- Pearson correlation was `-0.253`;
- the camera score accepted 89.1% of pairs;
- 29.1% of accepted pairs were harmful under hidden CD.

Thus, even covariance-intersection treatment cannot make two coherently biased
cameras identify which action-supported state update is safe. This reproduces
the Deform360 common-mode-bias limitation in a second dataset. A rolling
camera-regret selector was worse than the unchanged checkpoint and is not a
candidate for reserved-take evaluation.

## Decision

Do not open calibration objects, target objects, or the reserved `T7`/`T8`
takes for the current family. Keep the safe weak action field as an exploratory
ablation, not a headline method.

The next credible Bayesian-PhysTwin update must add information that is not
camera-equivalent: trusted robot-contact geometry, tactile or force/torque
evidence, sparse metric anchors, or a physical latent-bias model with a
source-calibrated regret bound. More camera-internal confidence, view counting,
or selector tuning cannot resolve the demonstrated ambiguity.

The compact evidence and all source-artifact hashes are in
`results/sota/pokeflex_action_guard_development_v1/summary.json`.
