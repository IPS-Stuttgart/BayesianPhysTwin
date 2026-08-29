# DLO-Lab Task-aware Value-of-information Source Protocol v1

## Question

Can a probe selected for expected downstream task value outperform a probe that
maximizes generic latent-state information when deformable manipulation depends
primarily on only part of the uncertain physics?

This is a controlled public-native source study, not a real-data confirmation or
a retry of the closed MI-only matched-reset protocol. The physical task, latent
space, action family, selector, controls, seeds, gates, and output root are new.

## Native System and Latent Bank

The study uses the pinned public DLO-Lab Genesis ROD implementation on CPU in
float64. A 16-node cantilever is fixed at its first two material nodes. The
finite source prior crosses five bending-modulus scales with three
twisting-modulus scales, giving 15 equally weighted physical worlds. Twisting is
included as a potentially observable but less task-relevant physical nuisance;
it is not an artificial label or future-dependent corruption.

Ninety source-test worlds are drawn once from continuous log-uniform bending and
twisting ranges. The three vertical tip-height goals are assigned cyclically.
No source-test task future is available during probe selection or decision
construction.

## Matched-reset Contract

One complete native solver snapshot is captured before intervention. Before
every probe and task branch, all native state fields are restored and rehashed.
Probe and task trajectories are separate branches; probe mechanics cannot enter
the task initial state. Bending and twisting arrays, native state, commands, and
initial geometry are content-bound.

## Probe and Task Banks

The five probes are null hold, slow vertical bend, slow lateral bend, slow
conical motion, and fast conical motion. Every nonnull probe returns the two
clamped nodes to their exact initial positions. Features use four registered
material nodes at four registered times.

The task bank contains nine smooth vertical tilts of the second root node,
followed by a hold. Loss is the last-16-frame mean squared height error of the
four free-tip nodes plus a fixed effort term. The action bank and three goals are
not optimized after native execution.

## Proposed Selector and Controls

For probe `q`, the proposed selector estimates

```text
E_world,observation,goal[
    loss(world, argmin_action E[loss | observation from q])
]
```

using only the 15-world particle probe and task bank. It chooses the probe with
minimum expected downstream Bayes task loss. This is task-aware expected value
of information, not maximum parameter entropy.

The matched controls are:

- no-probe Bayesian action;
- fixed slow-lateral probe Bayesian action;
- generic maximum-full-latent-MI probe Bayesian action;
- task-aware-probe MAP action;
- task-aware-probe Bayesian action (primary);
- task-aware guarded action with exact best-fixed fallback;
- goal-conditioned best fixed action;
- post-outcome finite-action oracle, reported only after scoring.

All probe arms share the same observation-bias and independent-noise draws.
The generic-MI and task-aware selectors must choose different nonnull probes
before source-test observations are generated.

## Staged Custody

The write-once order is:

```text
particle probe/task bank
-> particle selector and task-headroom analysis
-> source-test probe observations
-> decision seal
-> source-test task futures
-> score
```

The source-test future stage requires the content-bound decision seal. A terminal
result or failure forbids another attempt at the registered root.

## Gates

The selector gate requires the task-aware probe to differ from generic MI and to
reduce particle expected task risk by at least 3% versus null and 1% versus both
generic MI and the fixed probe. The headroom gate requires at least two
bending-conditioned oracle actions per goal, at least 8% oracle gain over the
best fixed action, and no dominant twisting-only action split.

The primary source-value gate requires task-aware Bayes to improve at least 3%
over best fixed, 1% over no-probe Bayes, and 0.5% over both generic-MI and fixed
probe Bayes. All paired 95% bootstrap upper bounds must be below zero, at least
12 decisions must differ from generic MI, and the task-aware arm may harm no more
than 25% of episodes relative to that control. All 90 source-test worlds remain
in the denominator; no replacement or secondary-arm rescue is allowed.

## Claim Boundary

A pass would establish a controlled public-simulator source result that
task-aware probing can beat generic information gathering for downstream
deformable action selection under a finite physical prior. It would not establish
real-world sensing, population calibration, SOTA point prediction, a published
DLO-Lab controller comparison, or target transfer. A failed gate remains useful
negative evidence and does not alter the successful DEFORM forecast.

No GPU, recording, protected target, held-v8 artifact, DLO4/DLO5 data, official
DLO3 evaluation, PokeFlex continuation, or fresh/reserved Deform360 data is
authorized. One registered source attempt only; no retry or post-result tuning.
