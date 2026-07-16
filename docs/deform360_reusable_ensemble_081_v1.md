# Deform360 reusable-twin ensemble 081 v1

## Question and boundary

The frozen reusable-dynamics experiment selected one PhysTwin parameter tuple
using raw Warp error, while its deployed predictor combined driven and
zero-action Warp trajectories with a source-frozen trust rule. This follow-up
asks two source-development questions:

1. Does a Gibbs posterior over plausible physical tuples transfer better than
   the best single tuple?
2. Does selecting the tuple for the deployed trusted predictor transfer better
   than selecting it for raw Warp?

The method was designed after the independent calibration outcome for episodes
0, 2, and 8 was known. Numerical selection uses only source episodes 1, 4, and
6 over frames `[1, 60)`. Reuse of the opened calibration episodes is explicitly
post hoc and exploratory. Episode 5 remains sealed and cannot be opened by this
protocol.

## Source-only Gibbs posterior

Eighteen finite physical tuples from the parent grid were retained. For tuple
`j`, the source loss is the execution-balanced relative score of its trusted
prediction. The Gibbs weights are

```text
w_j(T) proportional to exp(-(loss_j - min(loss)) / T).
```

Temperature was selected by leave-one-source-action-out prediction from the
locked grid `0.01, 0.03, 0.1, 0.3, 1.0`, subject to an effective candidate
count of at least two. The selected temperature was `0.01`, with effective
candidate count `2.376`.

| Source control | Relative score vs persistence | Lower is better |
| --- | ---: | --- |
| Gibbs posterior mean | 0.82796 | Yes |
| Point MAP | 0.81642 | Yes |

The mixture beats persistence but not its point-MAP control. The registered
source gate therefore fails and no mixture was run as an independent method on
calibration or target data. This rejects simple averaging of nearby physical
twins as the next accuracy improvement.

## Trust-aligned point MAP

The source point MAP was `spring=50000`, `drag=1`, `dashpot=50`. The same tuple
was selected in all three outer leave-one-action-out folds, jointly beat
persistence in two of three held-out source actions, and had only `1.25%`
maximum source metric degradation. This justified an exploratory mechanism
check on the already-open calibration episodes, not a new claim.

| Calibration predictor | Track RMSE | Symmetric CD | Track gain vs persistence | CD gain vs persistence |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 13.695 mm | 13.630 mm | - | - |
| Frozen parent | **10.665 mm** | 11.295 mm | **22.12%** | 17.13% |
| Trust-aligned MAP, commanded-count trust | 11.037 mm | **11.189 mm** | 19.41% | **17.91%** |
| Trust-aligned MAP, supported-count trust | 11.552 mm | 11.330 mm | 15.65% | 16.88% |

The point MAP does not dominate the frozen parent: supported-count track error
is `8.31%` worse and CD is `0.31%` worse. Commanded-count trust recovers some
track accuracy and slightly improves CD, but still has `3.48%` worse track
error than the parent. Consequently, selecting physical parameters against the
deployed trust score is not promoted.

Episode 8 has two commanded controller groups but only one controller spring.
Normalizing by the one supported group increases the trusted action response
and worsens that episode. This shows that an unsupported contact is an
association defect to detect or prevent, not a confidence problem that can be
repaired by increasing the remaining control weight.

## Decision

Three inexpensive routes are now closed on this rope:

- simple Gibbs averaging does not beat point selection on source actions;
- trust-aligned point selection does not beat the frozen parent on opened
  calibration actions;
- supported-contact normalization does not repair missing actuation support.

No additional selection should be performed on `081-stripe-rope`, and episode
5 remains sealed. The next credible state-of-the-art experiment must use fresh
objects and place the improvement upstream:

1. build one reusable canonical graph per object from source observations;
2. require every commanded contact patch to have geometric graph support before
   rollout, with unsupported points rejected rather than renormalized;
3. freeze one source-selected physical/trust recipe per object or topology;
4. evaluate untouched actions on a topology-stratified multi-object panel using
   the official Deform360 metrics and evaluator;
5. compare reusable PhysTwin with persistence, per-episode PhysTwin as an upper
   control, and ParticleFormer's published multi-episode result;
6. report object-clustered uncertainty, horizon behavior, direct support, and
   the frequency of safe association rejection.

The current one-rope errors are numerically below the published multi-episode
baseline, but the protocols and breadth differ. They are not a state-of-the-art
result. The publishable contribution would be filling Deform360's reusable
multi-episode PhysTwin setting on a fresh multi-object panel.
