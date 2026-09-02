# Non-overlapping replication result

Exact source revision: `56c31bc56b2fc3526f519f726ff7b922b909c65c`

Workflow run: `33613192892`

Artifact: `9839951506`

Artifact digest: `sha256:579972676099a536bc85033a07308df00a7fc121f450bfa99a692aab22d0759c`

Outer result ID: `d4aeb724d7a23bcd0137c4e16999737c33c654511e0a7a17adf9727862abbd0f`

Classification: **mixed non-overlapping source replication**.

The fixed v2 decision-regret policy was evaluated on 16 complete source-test
trajectories whose filenames have zero overlap with the 16 v2 source-test
trajectories. Official DLO4/DLO5 evaluation files were absent from the runtime
filesystem.

At the fixed four-measurement budget, decision-regret acquisition achieved:

- pooled task RMSE: **87.216 mm**;
- equal-trajectory improvement over physical fallback: **48.11%**;
- trajectory-bootstrap 95% interval: **[46.99%, 49.17%]**;
- nonfallback fraction: **76.97%**;
- mean acquired node blocks: **3.493**;
- harmful fraction among nonfallback decisions: **0%**;
- mean effective hypotheses when acting: **13.55**;
- state-ambiguous fraction when acting: **100%**.

The effect direction against state-variance and posterior-entropy acquisition
replicated. The paired improvement advantage over state variance was 1.08
percentage points with a 95% interval of [-0.17, 2.27] percentage points, so the
standalone replication did not satisfy the preregistered strictly-positive
interval criterion. The corresponding entropy comparison also included zero.
No acceptance threshold was changed after observing this result.

A trajectory-level additive transport slack of **0.094936** normalized-regret
units was calibrated from 18 complete calibration trajectories. It covered all
**16/16** replication trajectories. This is a trajectory-mean empirical
envelope under exchangeability, not a per-decision safety guarantee.

The exact finite-support certificate and the empirical transport envelope are
reported separately. The former remains exact only for the registered local
support and quotient; the latter addresses observed support-to-target mismatch
at the complete-trajectory level.
