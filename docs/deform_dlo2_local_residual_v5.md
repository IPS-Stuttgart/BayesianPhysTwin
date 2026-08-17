# Fresh DEFORM DLO2 Local-Residual Transfer v5

## Purpose

This protocol is the independent source transfer authorized by the passed DLO1 v4 local-residual gate. It asks whether the exact DLO1-selected correction transfers to a different deformable linear-object family after its physical baseline is trained from scratch.

No DLO2 hyperparameter selection is allowed. The local-residual arm is fixed at ridge `1.0` and shrinkage `0.5`, with the same causal feature definition, covariance construction, metric variance floor, and exact clamped-node policy as DLO1 v4. Prob4D is unused. Official DEFORM evaluation remains unreadable.

## Ordered Stages

1. Verify the exact DLO1 result and source-gate authorization.
2. Outcome-blindly partition all 56 DLO2 training trajectories into 40 fit, 8 validation, and 8 source-test trajectories using the previously registered hash seed.
3. Train the official DLO2 physical baseline from scratch for 6,400 updates and select its checkpoint using only physical-baseline validation L1.
4. Stop in `train-validation` mode before any source-test trajectory is loaded.
5. Fit the fixed local residual on DLO2 fit trajectories, evaluate it on validation, and seal the result.
6. Open the eight source trajectories only if validation improves at least 1%, wins at least 6/8, and has worst-case ratio at most 1.05.
7. Authorize an all-training refit and one-shot official evaluation only if source transfer improves at least 1%, wins at least 6/8, has worst-case ratio at most 1.10, and achieves mean coordinate L1 strictly below the published 9.7 mm DLO2 reference.

An unsuccessful validation stage returns the selected DLO2 physical checkpoint exactly and leaves source outcomes unopened. An unsuccessful source stage leaves official evaluation closed.

## Information Boundary

The predictor receives two observed initial material states, the known future trajectory of the four clamped nodes, and the baseline physical rollout. Future free-node truth is used only as a fit target or a post-seal score. There is no query innovation input. The four clamped nodes are unchanged from the physical rollout.

The generic training runner has an opt-in `train-validation` mode that writes checkpoint and validation provenance, then returns before the source loader. Existing `run`, `smoke`, and default behavior are unchanged.

## Claim Boundary

Passing this source stage would be fresh transfer evidence, not an official benchmark result. The final SOTA comparison requires a separately frozen all-56 refit and one-shot evaluation on the untouched official partition. Online prefix assimilation remains a separate-information result and cannot be mixed into the identical-information table.
