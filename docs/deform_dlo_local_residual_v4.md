# DEFORM DLO Local-Residual Source Protocol v4

## Purpose

This post-open DLO1 study tests whether a trajectory-grouped local residual model can improve the selected update-6400 DEFORM rollout. It addresses the main limitation exposed by the closed v3 action-analog result: whole-trajectory retrieval transferred weakly because similar actions need not share the same local simulator error.

This is backend research, not a Causal4D main-paper result. DLO2 and official DEFORM evaluation artifacts remain unreadable by the runner.

## Predictor

For each internal material node and future step, v4 predicts the DEFORM residual with a ridge-regularized linear local-dynamics model. Query features contain only:

- the two observed initial material states;
- the known future trajectory of the four clamped nodes;
- the selected physical baseline rollout;
- local position, velocity, acceleration, curvature, node-to-action vectors, normalized time, and arc position.

All features are expressed in the initial action/gravity frame. Future free-node truth is available only on the fit split as the regression target and on validation/source splits for scoring after the corresponding seal. The query API has no argument for future free-node observations or a query innovation.

Exact duplicate causal query trajectories are collapsed before fitting. Coefficient uncertainty uses trajectory-cluster sandwich covariance rather than treating hundreds of time-node rows as independent. Predictive variance adds fit residual variance, a metric variance floor in square metres, and the unresolved part of a shrunken mean correction. The four clamped nodes remain exactly equal to the physical baseline.

## Frozen Selection

The finite bank crosses five ridge values with five shrinkage values. Selection uses only the eight DLO1 validation trajectories. An arm must improve mean coordinate L1 by at least 1%, win at least six of eight trajectories, and keep every trajectory within 1.05 times its baseline error. If no arm qualifies, the output is the exact selected-update-6400 baseline and the source-test split remains unopened for this run.

Only the sealed selected arm may be evaluated once on the eight already-open DLO1 source trajectories. DLO2 work is authorized only if the source arm again improves by at least 1%, wins six of eight, has worst-case ratio at most 1.10, and remains below the published DLO1 mean. Passing would authorize a new, separately locked DLO2 source protocol, not official evaluation.

## Claim Boundary

The DLO1 fit, validation, and source partitions have already been examined by prior backend studies. Results are therefore exploratory method-development evidence. No result from this protocol can be called prospective, confirmatory, or state of the art. Exact fallback, immutable parent hashes, baseline reproduction checks, and read guards preserve the boundary for a later fresh DLO2 test.
