# Deform360 Dynamic Pairwise Belief Open27 V1 Result

## Status

This is an already-open source-development result on 27 episodes from five
Deform360 objects. It may close this exact method or motivate a new protocol,
but it cannot establish state of the art, calibration, non-regression, or a
fresh-object claim.

The frozen dynamic-pool arm failed its advancement gate. It must not be tuned
on these 27 cases or advanced to a fresh preregistered evaluation.

## Custody and execution

The target-free preflight passed all 27 cases with exactly 64 frame-zero
material identities and eight calibrated cameras. Causal AllTracker prefixes
were then built at update frames 19, 38, and 57. Before source outcomes were
opened, an independent custody audit verified all 27 cases:

- manifest content hashes and measurement archive hashes;
- hashes of every bound prediction, calibration, and seal input;
- the frozen AllTracker source and checkpoint hashes;
- no target or outcome reads during measurement construction;
- exactly 64 unique observed identities, eight cameras, and three updates.

All 64 observed identities were excluded from every future score. Every
rejected update preserved the selected backbone exactly, and every accepted
update passed the frozen physical and multiview guards.

## Source result

The dynamic arm accepted only 2 of 81 candidate updates; the other 79 were
exact fallbacks. Its small improvement over its own weaker dynamic-pool
backbone did not transfer against the registered fixed-16 pairwise-consensus
baseline.

| Arm | Hidden identity RMSE | Hidden Chamfer | Late identity RMSE |
| --- | ---: | ---: | ---: |
| Fixed-16 pairwise-consensus RBF | 7.550 mm | 6.970 mm | 9.290 mm |
| Dynamic 64-pool recursive RBF | 8.853 mm | 7.962 mm | 10.959 mm |
| Relative change | +17.26% | +14.23% | +17.96% |

The object-cluster 95% intervals for the dynamic-minus-fixed difference were
strictly positive for all three metrics:

- hidden identity RMSE: `+1.303 mm`, 95% CI `[+0.488, +2.119] mm`;
- hidden Chamfer: `+0.992 mm`, 95% CI `[+0.204, +1.757] mm`;
- late identity RMSE: `+1.669 mm`, 95% CI `[+0.171, +3.142] mm`.

No object jointly improved. The worst object-level regressions were 46.59% in
identity RMSE and 33.03% in Chamfer. Consequently, every scientific
advancement gate failed: primary improvement, cluster intervals, late error,
joint object wins, and the maximum-regression bound.

## Interpretation

Increasing the observation pool from 16 to 64 identities did not create a
better guarded state update. The frozen nuisance-aware selector found very few
updates that were simultaneously multiview-supported, action-supported, and
physically compatible. Those rare admissions did not recover the strength of
the fixed-16 pairwise field, while the larger pool also selected a weaker
physical/persistence backbone.

This closes the exact combination of:

- 64 frame-zero AllTracker identities;
- updates at frames 19, 38, and 57;
- three-view, action-support, and low-rank physical-response admission;
- nuisance-aware preselection followed by pairwise consensus;
- recursive dual physical/persistence RBF belief;
- fixed physical-cosine and correction-to-motion guards.

The result does not reject online Bayesian state updates in general. A future
candidate must add genuinely new information or a different inferential
contract, rather than retuning this selector on Open27. In particular, it
should address the observed support bottleneck and the common-mode camera-bias
ambiguity without weakening exact fallback.

## Provenance

- implementation commit: `5d6b196144b004a6d72d714e2a993c0c488a06b5`
- code archive SHA-256:
  `65493c1c9e9d2da076c1eac3c8b5c64052ae3a247dad82299b5401fd0a980c1a`
- transfer manifest SHA-256:
  `02babb4e041fce354a168a76c055a043032dbbf5eb5320e2c8a0e409bb2d83bb`
- target-free preflight canonical SHA-256:
  `de07052d3a5e17ac9e0d6f5db9ba2795bbca532589df8590e83272952f698f00`
- measurement custody canonical SHA-256:
  `94b79fef3fe3b876fb434ddb33c6af8189a65d7539445fdf8d8b0b447f4b6576`
- measurement custody file SHA-256:
  `07cde310e4404f918c1580fee9f4ed6dd2a4780e970f4803de766e2a0ababc39`
- source summary SHA-256:
  `11c56e0d97098ce60e5f49830f8cede7769d50e1053155676bbd454195b9e428`
- compact evidence:
  `results/sota/deform360_dynamic_pairwise_belief_open27_v1/`
