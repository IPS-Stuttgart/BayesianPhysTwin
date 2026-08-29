# DLO-Lab wrapping finite-sample certified guard v9

## Question

The prospective v8 public-simulator replication failed its original zero-harm
gate, but the frozen 0.975 guard still improved mean reward by `0.005042`,
reduced unguarded harms from 10 worlds to 2, and cut mean downside by 94.29%.
Because that controller was fixed before v8 outcomes, its 2/144 harm count also
supports an exact one-sided 95% binomial upper bound of `0.043073`.

V9 asks a narrower and more defensible question: does that already-fixed guard
retain positive value on a fresh panel while its independently measured
world-level harm risk remains below a registered 5% budget?

## Certify or fall back

The action policy is unchanged from v8. A 9x9 interpolated material belief is
updated from the same correlation-aware noisy prefix. The primary controller
chooses the highest-posterior-mean action whose posterior probability of
beating the fixed action by at least `0.002` reward is at least `0.975`.
If no action qualifies, it executes the fixed action exactly.

The study-level wrapper admits this controller only because the bound v8
calibration certificate passes. If the certificate or any custody check fails,
the candidate is not executed and the registered fixed action is retained.
The 5% budget was frozen for v9 after v8 opened; v8's original failed gate is
not reclassified, and no v8 threshold sweep selects the v9 controller.

## Fresh panel

The denominator is 288 public-simulator worlds drawn once from the same
registered action-transition stress distribution with seed `261910`. Every
material is disjoint from the source particles, v1-v4 worlds, and all 144 v8
calibration worlds. Thirty-two prefix batches precede one sealed decision
barrier; only a passing prefix and certificate gate authorizes future rollout.

Execution uses the exact native-Linux runtime qualified by v7. The runner reads
only compact, hash-bound v8 evidence and the compact calibration certificate;
it does not read v8 raw arrays, partial v4 futures, protected data, held-v8, or
DLO4/DLO5.

## Registered gate

The primary world-level harm event is a mean reward more than `0.002` below the
fixed action. The complete 288-world result must have an exact one-sided 95%
Clopper-Pearson upper bound no greater than `0.05`. It must also gain at least
`0.003` over fixed with a positive paired bootstrap lower bound, reduce at
least five unguarded harms and at least 75% of unguarded mean downside, retain
at least 15% of continuous-Bayes gain, and capture at least 5% of oracle
headroom. All native and prefix-parity checks are required.

The study has one attempt, no retry, no replacement, and no partial-case
estimand. It uses only public simulation and makes no real-world safety,
official benchmark, SOTA, perception, or material-identification claim.
