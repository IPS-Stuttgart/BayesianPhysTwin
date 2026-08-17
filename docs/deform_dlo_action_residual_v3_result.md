# DEFORM action-conditioned residual analog v3 result

## Decision

The frozen DLO1 source gate failed. The action-conditioned residual analog is
not authorized to read DLO2 or any official DEFORM evaluation trajectory.

The validation selector chose four trajectory-level neighbors, a source-relative
RBF length-scale multiplier of `0.5`, and correction shrinkage `0.125`. On the
eight validation trajectories, coordinate L1 fell from 9.2562 mm to 9.0298 mm,
a 2.446% improvement, with eight of eight wins and a worst-case ratio of
0.9976. The validation gate passed before any source-test trajectory was loaded.

On the eight source-test trajectories, the same sealed arm changed mean L1 from
9.6424 mm to 9.6265 mm, only a 0.165% improvement. It won six of eight cases and
kept the worst case within 1.0434 times baseline, but missed the preregistered
one-percent transfer threshold. The model therefore failed its source gate even
though both its absolute error and baseline remain below the contextual 10.1 mm
DLO1 paper reference.

## Interpretation

Known action similarity explains a small, directionally useful component of the
DEFORM discrepancy, but whole-trajectory residual analogs do not transfer enough
to justify a fresh DLO2 experiment. The gap between the validation gain and
source gain is exactly why the independent source panel was retained.

This result closes the fixed RBF whole-residual family on DLO1. It does not close
action conditioning generally. A successor must predict local residual dynamics
from action and physical state rather than retrieve an entire error trajectory,
and it needs a new source protocol rather than a relaxed threshold or another
arm from this opened bank.

## Custody and reproducibility

The execution used clean source commit
`249cc4a26b966086bac9fa58ed30ae930882acb3`, Python 3.10.12, Torch
2.0.1+cu118, CUDA 11.8, and the exact update-6400 checkpoint. The reproduced
validation and source baselines differed from their archived values by only
`3.37e-8 m` and `1.14e-9 m`, both inside the frozen `1e-7 m` drift bound.

The runner installed read guards for the DLO1 official evaluation directory and
the complete DLO2 tree. The result records `dlo2_read=false`,
`official_eval_read=false`, and
`fresh_dlo2_action_residual_authorized=false`.

Compact evidence is stored under
`results/sota/deform_dlo_action_residual_v3/`. The complete model and source
prediction archive remain immutable at
`/home/florianpfaff/source-only/deform-dlo-action-residual-v3/run-249cc4a2` on
`gpuserver6000`.
