# Native Slingshot: Matched Belief-Control Source Study

This new experiment follows the passing nine-world decision-value screen. It
does not revise any earlier failure or the existing DEFORM results. Execution
is local, CPU-only, public-simulator-only, with no new recordings, robot use,
protected targets, or GitHub publication. The exact output root is
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1`.

## Question

Does preserving shared physical uncertainty across candidate and incumbent
actions improve a calibrated decision rule beyond matched point, predictive
mean, and independently coupled controls? The Bayesian update, weighted
quantiles, and split-conformal order statistic are standard methods, not a new
theorem. A positive result would be controlled simulator evidence, not proof
of real-world safety, automatic RGB perception, or official Slingshot SOTA.

The source screen's best fixed action (action 5, positive final yaw variation)
is the incumbent. All seven unique source actions are unchanged. Slot 7 is
now an exact duplicate of action 5 for numerical QA. Their first macro action
is identical. The native robot, solver, control interpolation, contact model,
release, and reward remain unchanged. The nominal controller was optimized
with only 64 candidates, not the published full CMA-ES training budget.

## Worlds And Observations

The particle bank is the 27-point Cartesian grid of x placement
`[-0.02, 0, 0.02] m`, bending parameter `E=[50000,100000,200000]`, and stretching
parameter `K=[400000,800000,1600000]`. These are native simulator parameters,
not experimentally identified material properties. Nine already-open source
worlds are reused with exact hashes; the remaining 18 receive fresh native
rollouts. Tensor weights `[0.25,0.5,0.25]` approximate uniform placement and
log-uniform E/K by trapezoidal quadrature. No bank parameters are fitted to
calibration or evaluation outcomes.

There are 19 calibration worlds and 32 evaluation worlds, drawn continuously
inside the same ranges using NumPy seeds 260901 and 260902. They are not
restricted to the particle grid. Each episode has one independent synthetic
sensor draw (seeds 260903 and 260904). The statistical unit is the whole world
and sensor draw, not a coordinate, frame, action, batch slot, or padded copy.

The allowed observation is the 3D position of rod nodes 3, 6, and 8 and the
sphere center at native frames 139, 219, and 299: 12 observations. Each has
2 mm independent xyz Gaussian noise and one shared 5 mm xyz Gaussian bias
per episode. The likelihood analytically marginalizes that common bias.
Identities and the metric frame are assumed known. This is a controlled
partial-observation model, not a claim that current camera providers meet it.

## Causal Execution

A prefix observer stops the native simulation immediately after step 300,
before the second action is entered. It exposes no future trace or full-task
reward. A preliminary eight-world mixed-parameter prefix run is compared with
the corresponding already-open native source prefixes. All position errors
must be at most 0.5 mm before new model/calibration execution is allowed.

Each full rollout starts from a fresh native process. No hidden-state restart
is used. A prefix batch may contain eight different worlds; its final batch
is padded with registered duplicates, which never increase the denominator.
Every full world rollout contains the seven candidate actions plus the exact
incumbent duplicate. Common-prefix and duplicate positions must agree within
0.5 mm, duplicate rewards within 0.001, fixed endpoints within 1e-9 m, and the
full rollout must replay its separately sealed prefix within 0.5 mm.
These engineering envelopes are not statistical coverage guarantees.

The order is fixed: prefix qualification, model bank, all calibration-prefix
predictions, calibration futures, calibrator, all 32 evaluation-prefix
decisions, all evaluation futures, then scoring. Before any evaluation future
is initialized, a write-once barrier must revalidate all 32 prefix predictions
and their command bytes. Workers consume write-once claims and cannot use a
different output root, skip cases, rerun a claimed task, or silently replace
a failed world. A technical failure terminally fails this attempt; artifacts
remain preserved. All earlier tight replay failures remain failed.

## Matched Controls

1. The best source-selected fixed incumbent.
2. The nominal native point model.
3. The prior predictive mean (no prefix update).
4. The MAP particle point model given the prefix.
5. The posterior predictive mean.
6. A posterior-mean ablation that incorrectly ignores the shared sensor bias.
7. A calibrated mean-regret guard.
8. A calibrated independently coupled regret guard.
9. The primary calibrated jointly coupled regret guard.

All posterior methods use the same particles, actions, observations, and native
reward table. For this linear reward, expected reward from the posterior mean
trajectory equals posterior expected reward; it is not counted as a separate
Bayesian improvement. The joint and independent guards preserve exactly the
same action marginals and differ only in their coupling to the incumbent.
The MAP baseline is a discrete-particle point controller, not a separately
optimized continuous parameter estimator.

For loss `L=-reward`, each guard estimates an upper bound on
`L(action)-L(incumbent)`. Joint and independent guards use the 90th posterior
quantile; the mean guard uses the posterior mean difference. Every alternative
gets a 0.002 native-reward numerical margin. The incumbent regret is exactly
zero. One maximum-over-alternatives nonconformity score is formed per complete
calibration world. The nonnegative 18th of 19 score is added to each guard's
bound. An action is admitted only with a strictly negative calibrated bound;
otherwise the original incumbent command object/bytes are returned.

## Frozen Decision Gate

The primary must have all 19 calibration and 32 evaluation worlds, with no
replacement and all native QA passing; at least eight nonfallback decisions;
mean reward gain at least 0.005 and at least 10% of the incumbent's excess over
zero (denominator floor 0.01); and a positive paired 95% bootstrap lower bound
against the incumbent, mean-only guard, and independent guard. Use 10000
world-level bootstrap replicates, seed 260905, with equal world weighting.

Require at least 87.5% simultaneous action-bound coverage and at most 10% for
the one-sided 95% binomial upper bound on harmful decisions. Harm means reward
below the incumbent by more than 0.002. With 32 worlds, that harm gate requires
zero observed harmful decisions. Require at least two distinct oracle actions.
Report every arm's reward, gain, harm count, nonfallback count, worst-four mean
regret, and paired intervals; a guard's conservatism is not hidden by reporting
only average reward. Failure to outperform the calibrated controls is a
negative contribution result even if the method beats doing nothing.

Exchangeability is a property of this frozen synthetic generator and split,
not of real objects or hidden simulator states. The grid approximates a
continuous distribution, and the native dynamics have a measured numerical
floor. Calibration and replay checks address this experiment only. No outcome
automatically authorizes real or reserved-target evaluation.
