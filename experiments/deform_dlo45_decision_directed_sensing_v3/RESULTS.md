# Non-overlapping replication result

Exact scientific revision: `56c31bc56b2fc3526f519f726ff7b922b909c65c`

Workflow run: `33613192892`

Artifact: `9839951506`

Artifact digest: `sha256:31ddbb20dc084514afb6b021ec4f6b0bdff72e22124c4734fb18b1952b347889`

Outer result ID: `ba5f43a2ed1ca9c95a2a032e95679df06d3b6c6ec20cc3ad962ea6326f543fe0`

Core result ID: `ac1626f7392c1de2d95ff4d5fa5e937d113f46c573c48b4dcef5fd3b38ef6ade`

Classification: **mixed non-overlapping source replication**.

The fixed v2 decision-regret policy was evaluated on 16 complete source-test
trajectories whose filenames have zero overlap with the 16 v2 source-test
trajectories. Official DLO4/DLO5 evaluation files were absent from the runtime
filesystem.

At the fixed four-measurement budget, decision-regret acquisition achieved:

- pooled task RMSE: **87.216 mm**, versus **167.806 mm** for physical fallback;
- equal-trajectory improvement over physical fallback: **48.11%**;
- trajectory-bootstrap 95% interval: **[44.51%, 51.82%]**;
- nonfallback decisions: **251/304 (82.57%)**;
- mean acquired node blocks over all decisions: **1.155**;
- decisions acquiring at least one block: **132/304**;
- harmful fraction among nonfallback decisions: **0/251**;
- mean effective hypotheses when acting: **11.99**;
- state-ambiguous fraction when acting: **97.61%**.

The effect direction against state-variance and posterior-entropy acquisition
replicated. The paired equal-trajectory improvement advantage was 1.07
percentage points over state variance and 1.62 points over posterior entropy,
but both 95% intervals included zero. The standalone replication therefore did
not satisfy the preregistered strictly-positive interval criteria and remains
classified as mixed. No acceptance threshold was changed after observing this
result.

A trajectory-level additive transport slack of **0.094876** normalized-regret
units was calibrated from 18 complete calibration trajectories. It covered all
**16/16** replication trajectories for the registered trajectory-mean target.
This is a trajectory-mean empirical envelope under exchangeability, not a
simultaneous per-decision guarantee.

## Provenance correction

An earlier prose-only summary on this branch reported numbers from a superseded
intermediate result rather than the immutable artifact currently attached to
run `33613192892`. This file and the pull-request description now use the exact
artifact values above. No scientific code, protocol, split, policy, threshold,
trajectory outcome, or artifact has been changed.

## Boundary

This is source-test-only evidence inside the public DEFORM training partition.
It is not official evaluation-split performance, unseen-object validation,
learned-sensor validation, continuous-control certification, deployment safety,
or state of the art. The finite-support certificate and the empirical
trajectory-mean transport envelope are separate claims.