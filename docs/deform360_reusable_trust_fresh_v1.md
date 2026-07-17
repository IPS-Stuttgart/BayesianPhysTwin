# Deform360 reusable-twin trust gate v1

## Question

Can an automatic, frame-zero PhysTwin fill the empty physics row in
Deform360's same-object unseen-episode benchmark without the catastrophic
over-transmission observed under a fixed response scale?

This remains separate from Causal4D's partial-observation setting. The official
multi-episode comparison is zero-shot on each test episode. The `PhysTwin*`
appendix visualization that observes 80% of a test episode is an upper-bound
control, not the Table 4 comparison.

## Exhausted-source diagnosis

The frozen 27-episode action-support predictor failed because one global
response scale transmitted controller motion in episodes where the bilateral
virtual attachment was not a valid contact model.

| Source-only arm | Future track | Future CD | Late track | Late CD | Worst future degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed response | -15.09% | -11.67% | -16.90% | -12.68% | 1210.2% track |
| Binary closure gate | +3.16% | +3.20% | +7.31% | +7.98% | 75.8% CD |
| Simulator self-diagnostic | +3.21% | +3.54% | +3.39% | +5.64% | 257.6% track |
| Closure AND self-diagnostic | **+5.65%** | **+6.00%** | **+8.21%** | **+9.62%** | **2.7% track / 2.6% CD** |
| Nondeployable alpha oracle | +9.22% | +9.26% | +12.10% | +13.68% | n/a |

The same-object mean and same-object ridge variants degraded the average. The
useful variable is the realized interaction regime, not object identity alone.
The ungated self-diagnostic has positive mean performance but unsafe outliers.
Intersecting it with the independently cross-fitted closure gate recovers about
60% of the oracle gain and removes the catastrophic tail.

This is the concrete Prob4D lesson carried into PhysTwin: correlated or fitted
evidence is not automatically trustworthy. A correction receives influence
only when a separate observation of its validity agrees, and rejection returns
the exact baseline rather than a numerically approximate fallback.

## Frozen predictor

The candidate consumes only inputs available in the zero-shot episode setting:

1. the known robot trajectory and gripper openness;
2. frame-zero object geometry;
3. the predicted PhysTwin response relative to persistence.

It does not consume tactile, symbolic action labels, post-initial object
observations, or object outcomes. A source-fitted closure threshold first tests
whether transmission is admissible. If accepted, a 47-feature ridge model
predicts a response scale from action kinematics, geometry scale, and the
simulator's own spatial/temporal response signature. If rejected, `alpha=0` and
the output is byte-identical persistence.

The frozen candidate hash is
`177c9642a0e043f69afe206c39c748fda35ea6d48cd70aa36f0752035b39da11`.
The loader verifies this checksum before prediction. The standalone artifact
also records every source-input hash and its Python 3.12.3 / NumPy 2.0.2 fit
runtime.

Lock revision 2 supersedes the initially recorded candidate hash before any
fresh object was downloaded or inspected. This amendment only made the fitted
candidate standalone and runtime-explicit; it did not change its feature set,
gate, source episodes, or predictions at reported precision.

## Fresh admission panel

The lock was written before downloading or inspecting the following object
data:

| Topology | Object | Fit episodes | Held-out episodes |
| --- | --- | --- | --- |
| 1D | `003-cable` | 1, 3, 4, 6, 7, 9 | 0, 2, 5, 8 |
| 2D | `086-cotton-scarf-cloth` | 1, 3, 4, 6, 7, 9 | 0, 2, 5, 8 |
| 3D | `171-penguin` | 1, 3, 4, 6, 7, 9 | 0, 2, 5, 8 |

All twelve held-out predictions must be generated and hashed before any of
their outcomes are opened. The conjunctive admission gate requires at least 3%
future improvement in both track and Chamfer, 5% late improvement in both,
less than 10% per-episode degradation, and no object with median degradation.

Passing this panel does not establish state of the art. It only admits the
method to the expensive benchmark reproduction.

## SOTA boundary

Deform360 Table 4 reports ParticleFormer at 51 mm future Chamfer and 79 mm
future track error. The current 76-frame active-window errors are numerically
smaller, but the horizon, episode subset, and aggregation are not the published
protocol, so comparing those absolute values would be invalid.

A defensible SOTA claim requires:

1. the exact full-horizon same-object unseen-episode split;
2. the official particle identities and metric definitions;
3. the same initial-state and known-action inputs for every method;
4. ParticleFormer, persistence, pooled-physics, and single-episode controls;
5. a multi-object execution-balanced result below both 51 mm CD and 79 mm track;
6. uncertainty and failure-rate reporting in addition to mean error.

## Implementation

- `src/causal4d_public/deform360_reusable_trust.py` implements typed loading,
  feature construction, checksum validation, and the exact fallback.
- `scripts/remote/diagnose_deform360_same_object_trust.py` reproduces the
  exhausted-source cross-fit and emits the frozen candidate.
- `scripts/remote/apply_deform360_reusable_trust.py` applies the candidate to a
  prediction-only archive and records all hashes and information boundaries.
- `configs/causal4d_public/deform360_reusable_trust_fresh_v1.json` is the
  prospective fresh-object lock.
- `milestones/deform360-reusable-trust-source-v1/` archives the exact source
  diagnosis and standalone fitted candidate.

The next action is data processing, not method selection: download the three
locked objects, fit object-level physics only on the six declared training
episodes, seal all twelve held-out predictions, and then score the admission
gate once.

## Physical-fit addendum

The raw transfer began after the source milestone was tagged. Before inspecting
fresh object metadata, media, or outcomes, the physical fit was made exact in
`deform360_reusable_trust_physics_addendum_v1.json`. It uses the 18 finite
parameter tuples inherited from the earlier reusable-dynamics grid: spring
stiffness 10k, 30k, or 50k; drag 1, 3, or 10; and dashpot 50 or 100.

The trust decision is computed once from the fixed 10k/10/100 reference
response. Candidate physics cannot alter that decision or its alpha. The pooled
tuple minimizes an execution-balanced normalized track-and-Chamfer score over
the six declared fit episodes. Leave-one-fit-episode-out and single-episode
selection controls are reported. The selected tuple is then frozen before the
twelve held-out predictions are generated.
