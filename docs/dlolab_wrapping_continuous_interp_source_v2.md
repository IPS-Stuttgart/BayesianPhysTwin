# DLO-Lab wrapping continuous interpolation source study v2

## Question

The opened finite-bank wrapping study showed positive Bayesian decision value, but
the finite-particle gate failed. A later discrete-particle continuous-world attempt
completed its prefixes and futures but terminated before scoring on JSON metadata
serialization. That attempt remains terminal and unscored.

This distinct study asks whether a continuous material-response approximation can
improve off-grid decisions without fitting anything to the failed attempt's
futures. The scientific change is the model class, not a retry of its analysis.

## Frozen continuous model

The unchanged 3x3 public-simulator source bank is laid out in normalized log
stretching and log bending coordinates. Piecewise bilinear interpolation produces
prefix coordinates and native action rewards on a fixed 9x9 tensor grid. The prior
is the normalized tensor-product trapezoidal rule. There are no learned weights,
bandwidths, or outcome-fitted hyperparameters.

Five arms are frozen:

- the best fixed action under the continuous prior;
- posterior expected utility over the original nine finite particles;
- plug-in MAP on the interpolated 9x9 grid;
- posterior expected utility on the interpolated 9x9 grid;
- the same continuous expected-utility rule under a deliberately misspecified
  independent-noise likelihood.

The primary arm is continuous posterior expected utility. All arms see the same
permitted prefix and the same 4,096 sensor draws per world. Sensor noise contains a
5 mm shared translation bias and 2 mm independent noise.

## Fresh worlds and information boundary

Thirty-two continuous materials are drawn once from the same registered log-uniform
rectangle under a new seed. They are disjoint from both the nine source particles
and all 32 worlds in the terminal v1 attempt.

Four 600-step prefix-only batches contain no task future or reward. All observations,
quadrature metadata, and arm decisions seal before the barrier can authorize any
2,200-step task future. Every prefix-only trajectory must match its corresponding
full-run reset within 1 mm. Missing tasks, failed native QA, or a failed barrier
terminate the one attempt without retry or replacement.

The terminal v1 failure, source bank, runtime, source files, and exact paths are
hash-bound. The runner verifies that v1 has no result or generation seal and never
reads its numerical payload for method or threshold selection.

## Gates

Before futures, continuous Bayes must differ from the best fixed action, finite
Bayes, and continuous MAP in at least 256 sensor decisions, select at least two
actions, and retain four qualified prefixes.

The source result advances only if all 32 worlds qualify and continuous Bayes:

- gains at least 0.01 native reward over the strongest fixed action;
- gains at least 0.002 over finite-particle Bayes;
- gains at least 0.001 over continuous MAP;
- gains at least 0.003 over the ignored-shared-bias arm;
- has a positive paired 95% world-bootstrap lower bound for all four gains;
- captures at least 30% of the oracle headroom;
- harms no more worlds than either finite Bayes or MAP beyond the frozen 0.002
  numerical margin.

The 32 worlds must also contain at least two oracle actions. Inference uses 20,000
paired world-bootstrap replicates with the registered seed.

This is public-simulator source evidence only. It carries no official benchmark,
SOTA, real-world, perception, material-property, or physical-safety claim, and no
passing result authorizes an automatic successor.
