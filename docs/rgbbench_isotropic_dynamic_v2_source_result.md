# RGBench Isotropic Dynamic v2 Source Result

## Status

The frozen source gate failed. Calibration and target outcomes remain sealed,
and this method must not be retuned and rerun as v2.

The source operator used commit
`b43b37c812ff5e237cf8948d4edcc165fddef3c5`. All 27 physical baselines were
completed before prefix assimilation. All six temporal arms were sealed for
all 27 cases before any source future point cloud was opened.

The exact source-gate artifact is
`results/sota/rgbbench_isotropic_dynamic_v2/source_gate.json`, with file
SHA-256
`37beb8e29c588c89228ef223477ba470a91384d6264ac3d9cf7842580a070357`.

## Result

Leave-one-garment-out, action-specific temporal shrinkage improved the
object/action-balanced primary RGBench metric by only 0.94% over the
remeshed PyBullet physical baseline:

| Quantity | Result |
| --- | ---: |
| Physical baseline | 45.65 mm |
| Cross-fitted candidate | 45.22 mm |
| Published GarmentDynamics | 28.99 mm |
| Improved garment/action cells | 4 / 9 |
| Cells below published GarmentDynamics | 1 / 9 |

All five registered source gates failed. In particular,
`white_cakeskirt` regressed by 5.07%, while `brown_coat` and
`green_tshirt` improved by 3.71% and 4.18%.

The all-source deployment choices would have been full slope for fling and
grasp, and static graph persistence for fold. They are not authorized for
calibration because the cross-fitted gate failed.

## Model-family headroom

The post-open source-only audit is
`results/sota/rgbbench_isotropic_dynamic_v2/source_headroom_audit.json`, with
file SHA-256
`4018d4eccb3b2241efb11809e537f1a696b7cbb6f8f92a6b84d95bc878f02ac1`.

An oracle that chooses independently among the six frozen temporal arms in
each garment/action cell improves the physical baseline by only 1.16%. Giving
that oracle an additional exact physical fallback raises the ceiling to
3.04%, with five improved cells and still only one cell below published
GarmentDynamics.

This distinguishes the failure from a poor cross-fitting rule. The frozen
linear slope family does not contain enough source headroom to pass the 5%
gate or close the approximately 16 mm gap to the published simulator.

## Decision

Close v2 as a valid negative source result. Do not weaken the source gate,
inspect calibration or target outcomes, or add a larger temporal-shrinkage
grid.

Two post-open source-only controls now close the immediately available public
alternatives; see `docs/rgbbench_public_backbone_source_diagnostics.md`.
Exact endpoint-cloud persistence regresses in all nine garment/action cells,
and the released plain-MuJoCo Flex wrapper fails a one-case competence smoke.
RGBench's published GarmentDynamics trajectories are not released, so they
cannot be assimilated directly. A further RGBench method is justified only by
a substantially stronger public dynamics backbone, not by another correction
layer around either failed alternative.
