# DLO-Lab wrapping chance-guard native-Linux replication v8

## Question

The v4 method was frozen before a 72-world prospective run, but that run ended
after 69 futures when the WSL Genesis runtime crashed. No v4 task-value score
was computed. The separately frozen v7 qualification now passes 24/24 fresh
constructors and 4/4 complete rollouts on native Linux.

V8 asks the original scientific question on a larger, non-overlapping panel:
can a baseline-relative posterior chance guard retain useful wrapping reward
while preventing the downside of unguarded expected-utility action choice?

## Frozen method

The method is unchanged from v4. A 9x9 interpolated material belief is updated
from the same correlation-aware noisy prefix. For each action, it estimates the
posterior probability of beating the registered fixed action by at least
`0.002` reward. The primary controller chooses the highest-posterior-mean action
whose probability is at least `0.975`; otherwise it executes the fixed action
exactly. Controls remain unguarded continuous Bayes, 0.90 and 0.99 guards,
finite-particle Bayes, continuous MAP, and the fixed action.

The threshold, observation model, quadrature, action bank, reward, comparators,
and scientific gates are copied from v4 without outcome-dependent changes.

## Fresh panel

The denominator is doubled to 144 public-simulator worlds drawn once from the
same source/development-defined action-transition region. Seed `261810` is
frozen before any prefix or future. Every material is disjoint from the source
particles and all v1-v4 worlds. Sixteen prefix batches precede one sealed
decision barrier; only a passing prefix gate authorizes the 144 futures.

Execution uses the exact native-Linux runtime qualified by v7. V4 partial
future arrays, rewards, and logs are excluded. Only its committed terminal
summary is read. The v7 arrays are likewise excluded from method selection.

## Scientific gate

The v4 gate is retained on the complete 144-world denominator. The primary arm
must gain at least `0.005` over fixed with a positive paired 95% bootstrap lower
bound, harm zero worlds beyond the `0.002` numerical margin, reduce at least two
continuous-Bayes harms and mean downside by at least 50%, retain at least 50%
of continuous-Bayes gain, lose no more than `0.012` mean reward to it, and
capture at least 20% of oracle headroom. At least two oracle actions and all
native/prefix-parity checks are required.

The study has one attempt, no retry, no replacement, and no partial-case
estimand. A failure remains a failure. It uses only public simulation and makes
no real-world safety, official benchmark, SOTA, perception, or parameter-
identification claim.
