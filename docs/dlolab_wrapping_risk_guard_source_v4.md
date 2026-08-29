# DLO-Lab wrapping posterior chance-guard source study v4

## Question

Two prospective public-simulator studies now show that Bayesian action choice
beats a fixed action on fresh off-grid wrapping materials. They also expose a
smaller but important downside: an expected-utility controller can select an
action that is substantially worse than the fixed baseline when the compact
physical model is wrong near an action transition.

This study asks whether a baseline-relative posterior chance constraint can
retain useful task reward while reducing that downside. It is a new decision
rule and a new 72-world stress panel, not a retry or reclassification of the
failed v2 or v3 mechanism gates.

## Development boundary

The already-open 32-world v2 and 48-world v3 studies are development data. A
hash-bound diagnostic exactly reproduces their registered fixed, finite-Bayes,
continuous-Bayes, and MAP decisions before evaluating eight chance thresholds.
The smallest candidate threshold with zero worlds harmed beyond the registered
`0.002` reward margin is `0.975`; it gains `0.0113903` over fixed across the 80
opened worlds. This is candidate selection, not prospective evidence.

The fresh panel is drawn once from the source/development-defined action-switch
region in normalized log material coordinates:

- stretching: `[0.60, 0.995]`;
- bending: `[0.02, 0.70]`.

The panel definition and threshold are frozen before any new native prefix or
future. Every material is disjoint from the nine source particles and all v1,
v2, and v3 worlds.

## Controller

The unchanged 9x9 interpolated physical belief supplies posterior weights after
the same correlation-aware noisy prefix. For action `a`, define

```text
p_improve(a) = P(r(a, theta) - r(a_fixed, theta) >= 0.002 | prefix)
```

The primary controller chooses the highest-posterior-mean action among those
with `p_improve >= 0.975`. The fixed action is always eligible, so rejection is
an exact decision-level fallback. Its registered source-bank index is `4`; its
stored improvement probability remains the literal probability of beating
itself by `0.002` (zero), rather than being overloaded as an eligibility flag.
The state innovation is used only by the registered likelihood; realized future
reward never enters admission.

Controls are unguarded continuous Bayes, 0.90 and 0.99 sensitivity guards,
finite-particle Bayes, continuous MAP, and the unchanged fixed action. All arms
share one prefix, source bank, sensor draws, action bank, and native futures.

## Information boundary

Eight 600-step prefix-only batches expose no future reward. All 4,096 sensor
decisions per world, posterior probabilities, and posterior expected values are
sealed before the decision barrier can authorize any 2,200-step future. Every
prefix must match the corresponding full rollout within 1 mm. Any missing task,
native-QA failure, or failed barrier terminates the one attempt without retry or
replacement.

The parent source, terminal v1 failure, v2/v3 development evidence, development
diagnostic, native runtime, source files, and exact roots are hash-bound.
Protected data, held-v8, DLO4/DLO5, official DLO3 evaluation, GPUs, and new
recordings are excluded.

## Gates

Before futures, all eight prefixes must qualify. The primary guard must make at
least 256 nonfixed decisions, differ from continuous Bayes at least 256 times,
use at least two actions, and every admitted nonfixed decision must satisfy the
registered `0.975` posterior threshold.

The complete source result passes only if all 72 worlds qualify and the primary
guard:

- gains at least `0.005` over fixed with paired 95% world-bootstrap lower bound
  above zero;
- harms zero worlds beyond the `0.002` numerical margin;
- reduces at least two harmed worlds from continuous Bayes, which itself must
  harm at least two worlds;
- reduces mean downside below fixed by at least 50% relative to continuous
  Bayes;
- retains at least 50% of continuous Bayes's positive gain and loses no more
  than `0.012` mean reward to it;
- captures at least 20% of oracle headroom.

At least two oracle actions must occur. Aggregation is equal by world after
averaging sensor draws, with 20,000 paired bootstrap replicates under the frozen
seed. The gate cannot pass through trivial fallback alone.

This is a public-simulator downside-control test. It makes no physical-safety,
official-benchmark, SOTA, real-world, perception, or parameter-identification
claim, and no result automatically authorizes a successor.
