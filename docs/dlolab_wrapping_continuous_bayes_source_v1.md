# DLO-Lab wrapping continuous-material Bayes source study v1

## Question

The opened finite-bank wrapping study found a positive Bayesian expected-utility
signal, but it did not pass its original promotion gate. This prospective public
simulator study asks a narrower and more useful question: does the unchanged
bias-aware posterior controller improve decisions when the true stretching and
bending parameters lie outside its nine-particle support?

The parent failure remains a failure and is not reclassified. This study tests a
new, frozen generalization hypothesis; it is not a retry of a parent trajectory.

## Frozen worlds and method

Thirty-two material worlds are drawn once from registered log-uniform ranges:

```text
stretching K in [20,000, 500,000]
bending E    in [1,000, 100,000]
```

The seed and exact values are committed in the protocol. All 32 pairs are distinct
from the nine source particles. The source bank, action bank, native reward, public
DLO-Lab controller, and Genesis rod simulator are unchanged.

Each world receives 8,192 synthetic prefix sensor draws with a 5 mm shared
translation bias and 2 mm independent noise. Four arms are frozen:

- the source-best fixed action;
- plug-in MAP material selection;
- posterior expected native reward under the bias-aware likelihood;
- posterior expected reward under a deliberately misspecified independent-noise
  likelihood.

The unit of inference is one continuous material world after averaging its sensor
draws. Confidence intervals use 20,000 paired world-bootstrap replicates with the
registered seed.

## Information boundary

The simulator first generates four 600-step, prefix-only batches. Prefix artifacts
contain no future and no reward. All noisy observations and all four decisions are
then sealed. A barrier is rederived from the prefix traces, source bank, and decision
arrays before any 2,200-step task future may be generated or read.

Only after the pre-future gate passes are the 32 full action banks simulated. Every
prefix-only trajectory is compared with the corresponding full-run prefix to detect
reset mismatch. A failed task, failed native QA check, or failed barrier terminates
the one attempt; there is no retry or replacement.

Protected data, held-v8, DLO4/DLO5, official DLO3 evaluation, GPUs, robots, and new
recordings are outside scope. This is public simulator source evidence, not an
official benchmark, SOTA, or physical-safety claim.

## Pre-future gate

Before any future can run:

- all four prefix batches must pass native QA;
- posterior Bayes must differ from the source-best fixed action in at least 256
  sensor decisions;
- posterior Bayes must differ from MAP in at least 256 sensor decisions;
- posterior Bayes must select at least two distinct actions.

Failure closes the study without task-future generation.

## Source advancement gate

The result advances only if all 32 worlds and native QA checks complete and
posterior Bayes simultaneously:

- gains at least 0.01 native reward over the source-best fixed action;
- gains at least 0.001 over MAP;
- gains at least 0.003 over the ignored-shared-bias arm;
- has a positive paired 95% world-bootstrap lower bound for all three gains;
- captures at least 20% of the oracle headroom over the fixed action;
- harms no more worlds than MAP beyond the frozen 0.002 numerical margin.

The 32 worlds must also contain at least two distinct oracle actions. Even a passing
source result authorizes no automatic successor, target study, or method promotion.
