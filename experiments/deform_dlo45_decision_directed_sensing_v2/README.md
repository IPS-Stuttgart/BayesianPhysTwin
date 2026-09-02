# DEFORM decision-directed virtual sensing v2

This source-calibrated, source-test-only pilot asks a stronger question than the
first replay:

> Can a physical twin choose **which recorded DLO node to observe next** by the
> expected reduction in exact decision regret, and thereby reach a useful action
> with less sensing than state-oriented acquisition?

No new data are collected. The 14 official evaluation trajectories of each DLO
are absent from the runtime filesystem. For each of DLO4 and DLO5, 39 training
trajectories fit the model, 9 disjoint training trajectories calibrate one shared
operating point, and 8 disjoint training trajectories form the source-test
cohort.

## Stronger action semantics

Version 1 used three scalar multiples of one correction. That portfolio was too
easy: the actions were largely collinear and often did not create a genuine
choice. Version 2 instead clusters source future responses in the registered
task space and freezes a global action portfolio consisting of:

- the exact endpoint-only fallback; and
- one full future-shape correction prototype for every source-response class.

The actions can disagree in direction and temporal shape. The result records
pointwise winner counts and pairwise action distances so that action competition
is auditable rather than assumed.

The registered task is the future 25-frame trajectory of central internal nodes
4--7. The local finite support and response quotient are built from endpoint
geometry and the known future endpoint-action path.

## Physical virtual measurements

One acquisition reveals an already recorded internal node's:

- current 3-D line-relative position; and
- one-frame 3-D line-relative velocity.

The eight candidate nodes are 2--9. Their values are masked at the start of each
case and revealed sequentially. Every acquisition path and action is frozen
before future internal-node outcomes are sliced.

## Acquisition policies

The comparison includes:

- exact worst-case decision regret;
- posterior Bayes risk;
- full-support entropy;
- response-class entropy;
- projected full-state variance;
- projected task-query variance;
- fixed center-out acquisition;
- deterministic random acquisition; and
- an unattainable diagnostic prefix oracle.

Adaptive policies maximize expected metric reduction per measurement cost using
the current local-support predictive mixture. This naturally accounts for
previously acquired, redundant measurements.

## Fixed risk requirement and source calibration

The normalized-regret tolerance is fixed at **0.05** as a task requirement. It is
not selected to maximize RMSE. This corrects the first v2 run, whose loose
calibrated tolerance admitted every action before any measurement and therefore
collapsed all acquisition policies to the same zero-sensing policy.

At this fixed risk budget, a grid selects one shared sensor likelihood scale and
one shared action-prototype shrinkage across DLO4 and DLO5. A candidate is
eligible only when it:

1. produces at least 20 nonfallback calibration decisions;
2. improves equal-trajectory task RMSE; and
3. keeps the harmful fraction among nonfallback decisions at or below 5%.

Eligible candidates are ranked by equal-trajectory task improvement, then action
coverage, then lower sensing cost. If no candidate passes, the run still emits a
deterministic diagnostic operating point but labels the calibration gate as
failed.

## Evidence reported

In addition to ordinary RMSE and action-coverage summaries, the run reports:

- DLO-stratified paired trajectory bootstrap intervals against every baseline;
- sensing-cost versus task-RMSE Pareto frontiers;
- empirical target violations of the finite-support regret certificate;
- action-use distributions and selected-node distributions; and
- the effective number of physical hypotheses remaining when a nonfallback
  action is taken.

## Interpretation boundary

The exact regret certificate is conditional on the registered finite local
support, quotient, and action loss matrix. Source-test target violations remain
empirical. This replay does not establish official evaluation-split transport,
unseen-object generalization, learned-vision sensing, continuous-control safety,
deployment safety, or state of the art.
