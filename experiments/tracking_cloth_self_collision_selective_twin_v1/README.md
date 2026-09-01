# Fresh self-collision selective-digital-twin confirmation

This experiment is the prospective confirmation requested by the selective
physical-twin paper direction. It asks whether a Bayesian contact-physics arm
adds value beyond an equally informed simulator-free residual selector on a
new repetition of public real cloth trajectories.

## Repetition protocol

The complete 36-recording self-collision factorial contains four materials,
three cloth/rod interaction geometries, and three repetitions. The numerical
information order is fixed:

1. `rep1` fits the finite contact-physics bank;
2. `rep2` freezes a leave-one-material-out handoff among persistence, constant
   velocity, a decaying local-velocity residual, and Bayesian contact physics;
3. `rep3` supplies only a 0.5 s causal prefix and future timestamps until all
   predictions are jointly sealed; and
4. the scoring stage opens `rep3` future cloth outcomes once.

The publisher describes the self-collision rod as static and stores its two
markers before the cloth markers. The parser accepts the documented
`four/two` and `normal/parallel` filename factors, and the reduced simulator
holds a robust prefix estimate of the rod pose fixed after initialization. It
receives no future rod or cloth coordinates. The physical model is a
transparent equal-marker-mass spring mesh with cylindrical rod contact,
friction, and optional non-neighbor cloth repulsion. It is a competence probe,
not a reproduction of the dataset authors' simulator.

## Primary comparison

The principal contrast is

```text
physics-enabled selector - matched residual selector
```

Both selectors see the same rep2 training contexts and use the same acceptance
rule. The matched selector can use persistence, constant velocity, or the
local residual predictor. The physics-enabled selector differs only by access
to the Bayesian contact-physics arm. Rejections copy persistence exactly.

## Finite-sample guarantee

`docs/selective_competence_theorem_v1.md` separates the deterministic exact
fallback guarantee, simultaneous bounded-loss source admission, and the exact
one-sided Clopper--Pearson harmful-use endpoint. The statistical unit and
population remain part of the certificate; no universal safety claim follows.

## Execution

A dedicated permanent workflow is used to preserve the source/prediction/score
information order. A later commit must add exactly one request file:

```text
.github/requests/tracking-cloth-self-collision-selective-twin-v1.json
```

The workflow uses `[self-hosted, Linux, X64, gpuserver4090]`, uploads the source
artifact, publishes the complete prediction seal before scoring, and never
uploads the private prediction arrays or raw trajectories. A failed source gate
prevents `rep3` numerical outcome access.

The reviewed branch contains no temporary formatter or write-enabled workflow;
formatting was applied in a separate self-removing branch operation before the
final repository-wide validation run. The confirmation request is intentionally
absent from this method-review branch.
