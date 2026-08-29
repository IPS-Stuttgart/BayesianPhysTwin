# DLO-Lab wrapping model-resolution ensemble source study v3

## Question

The frozen continuous-material v2 study established that Bayesian action choice
beats fixed control on 32 off-grid material worlds, but an 81-point interpolated
posterior did not improve on the original nine-particle posterior. A post-result
development diagnostic reproduced every registered finite-Bayes, continuous-Bayes,
MAP, and fixed decision before evaluating a narrower lead: averaging the two
posterior action-value vectors retained 98.966% of finite Bayes's gain, lost only
0.000225 mean reward to finite Bayes, gained 0.000668 over continuous Bayes, and
removed finite Bayes's two harmed worlds on that opened cohort. The diagnostic is
explicitly post-open and has artifact ID
`b2c2365c3d8f8f5702d250d7d1de62cfd2ac283ba297203e8e920a51b5c6b594`.

This distinct prospective study asks whether equal model-resolution averaging
retains finite-particle value with fewer baseline-relative harms on fresh worlds.
The v2 result remains failed and is not reclassified.

## Frozen controller

Both physical belief resolutions use the unchanged public-simulator 3x3 source
bank and the same correlation-aware prefix likelihood:

- finite Bayes integrates expected reward over the nine source particles;
- continuous Bayes integrates over the fixed 9x9 bilinear log-material grid;
- the primary ensemble averages their eight posterior action values with exact
  weights `0.5/0.5`, then takes the maximizing action;
- a resolution-maximin arm maximizes the componentwise minimum expected value;
- continuous MAP and the continuous-prior best fixed action remain controls.

The equal weights, quadrature, sensor model, action bank, and gates are fixed.
No mixture weight, disagreement penalty, or threshold is fit from v2 outcomes.

## Fresh worlds and information boundary

Forty-eight materials are drawn once from the registered log-uniform rectangle
under new world and sensor seeds. They are disjoint from the nine source particles,
the terminal v1 roster, and all 32 v2 development worlds.

Six 600-step prefix-only batches expose no task future or reward. All 4,096 noisy
observations per world and every arm decision seal before the barrier can authorize
any 2,200-step future. Every prefix trajectory must match its corresponding full
run within 1 mm. Any missing task, native-QA failure, or failed barrier terminates
the one attempt without retry or replacement.

The public source bank, terminal v1 failure, complete v2 development result,
runtime, source files, and exact paths are hash-bound. Protected data, held-v8,
DLO4/DLO5, official DLO3 evaluation, GPUs, and new recordings are excluded.

## Gates

Before futures, the ensemble must differ from fixed, finite Bayes, and continuous
Bayes on at least 256 sensor decisions, use at least two actions, and retain all
six qualified prefix batches.

The source result passes only if all 48 worlds qualify and the ensemble:

- gains at least `0.015` native reward over the best fixed action with paired 95%
  world-bootstrap lower bound above zero;
- loses no more than `0.001` mean reward to finite Bayes and has paired 95% lower
  bound above `-0.002`;
- retains at least 95% of finite Bayes's gain over fixed;
- has nonnegative mean gain over continuous Bayes;
- harms no more worlds than continuous Bayes and strictly fewer than finite Bayes
  beyond the frozen `0.002` numerical margin;
- captures at least 50% of oracle headroom.

At least two oracle actions must occur. Inference uses 20,000 paired world-bootstrap
replicates under the registered seed. This is a Pareto gate: a small mean sacrifice
is allowed only when the ensemble demonstrably reduces harms.

This is public-simulator source evidence only. It carries no official benchmark,
SOTA, real-world, perception, physical-parameter, or safety claim, and no passing
result automatically authorizes a successor.
