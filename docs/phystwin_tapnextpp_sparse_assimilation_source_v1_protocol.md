# TAPNext++ Sparse Assimilation Source Protocol

## Question

The frozen RGB-D completion provider passed its eight-case source-transfer gate
with 90.09% row support and 4.66 mm case-balanced prefix identity RMSE. This
study asks the next, separate question:

> Does that automatic sparse observation channel improve Bayesian-PhysTwin's
> untouched future beyond the existing dense persistence correction?

The panel contains the same eight previously opened source cases. It is an
exploratory transfer test, not independent confirmation.

## Frozen Arms

1. `physical`: the unmodified released PhysTwin rollout.
2. `dense_persistence`: a robust endpoint correction inferred from released
   dense pseudo-tracks before `train_end`, held constant in the future.
3. `tapnext_direct`: dense persistence plus the sparse update at its
   geometry-MAP graph nodes.
4. `tapnext_graph`: the same sparse update propagated with the normalized
   PhysTwin spring-graph Laplacian. This is the primary arm.

No future outcome selects a cap, graph strength, temporal gain, arm, or case.
The failed provider case remains in all aggregate accounting and must be
bit-exactly equal to `dense_persistence`.

## Observation and Association Boundary

TAPNext++ supplies metric points, support, residual-independent prior
reliability, and covariance in square metres. Geometry at the query frame
determines four candidate graph nodes and their association probabilities.
The physical residual does not change these probabilities or the prior
reliability.

Assignment-mixture spread is added to observation covariance. Repeated
tracker rows are capped at four effective observations per identity. The state
innovation enters once through a Gaussian inlier/outlier mixture. If multiple
correlated identities map to one node, only the most precise representative
contributes; their precision is not multiplied.

The sparse posterior is expressed relative to the already applied dense
correction. Both a direct field and a graph-smoothed field are limited to a
10 mm norm. Prediction variance includes the metric endpoint covariance,
graph-posterior covariance, a 5 mm unresolved floor, and random-walk growth
from the last observation to each future frame.

## Custody

Staging creates two separately hashed artifacts per case:

- prediction input: released prefix observations, graph geometry, sealed
  provider output, and the physical rollout;
- withheld source outcome: future point cloud, visibility, and manual tracks.

The prediction runner cannot read the withheld artifact. The evaluator first
validates the prediction archive, report, and seal, then opens the staged
source future. No held-v8 artifact is authorized or accessed.

## Metrics

The report includes future Chamfer distance and manual-track error, observed
query-identity error, disjoint hidden-identity error, early/middle/late
horizons, 90% conditional coverage, and NEES. Hidden identities exclude every
TAPNext++ query identity.

## Advancement Gate

Relative to `dense_persistence`, the primary graph arm must simultaneously:

- improve case-balanced Chamfer distance by at least 5%;
- improve all-identity track error by at least 5%;
- improve queried-identity track error by at least 10%;
- regress hidden-identity track error by no more than 2%;
- jointly avoid CD and all-track regression in at least 6 of 8 cases;
- retain at least 95% hidden future-frame support;
- attain at least 80% conditional 90% coverage and move no farther from 90%
  than the dense comparator; and
- preserve exact fallback for every failed provider case.

Only a complete pass authorizes a new, independently preregistered evaluation.
Failure stops this arm without tuning against the opened futures.

## Claim Boundary

Manual material identity is used at the query frame and in post-seal source
scoring. The experiment can establish provider-to-assimilation transfer on
opened source data. It cannot establish independent confirmation, a fair
open-loop state-of-the-art result, or target calibration.

