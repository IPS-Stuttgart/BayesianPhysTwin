# Deform360 causal-expert router source milestone

This milestone closes the first causal-transport and competence-routing study
without opening the sealed rubber-band episode or any PokeFlex target.

## Result

The fixed 49-candidate transport policy failed all three frozen source gates.
Pooled selection returned exact persistence, leave-one-action-out never beat
persistence, and its mean normalized score was `1.05084`.

The candidate-aware router was trained on 27 already exhausted episodes from
five objects. Its complete leave-one-object-out result was safe but too small:

| Source result | Value |
| --- | ---: |
| Mean normalized score | 0.99365 |
| Mean improvement | 0.635% |
| Maximum normalized score | 1.00000 |
| Accepted fraction | 7.41% |
| Win fraction | 3.70% |

The router therefore failed the preregistered 2% mean-improvement gate. The
old panel contained only 0.750% non-deployable oracle headroom for this expert
bank, so further tuning of the router cannot create the missing gain.

An explicitly post-failure rubber-band transfer diagnostic accepted one of six
open source episodes. It improved aggregate future track error by 0.482% and
Chamfer by 0.495%, with no episode degradation, but failed every accuracy gate.
The sealed episode remains unread.

Adding gains `0.05`, `0.1`, `0.2`, and `0.3` did not fix transfer. Persistence
remained the best fixed candidate; leave-one-action-out still selected a
candidate that degraded one episode by 30.50%. The per-action oracle improved
9.87%, confirming useful action-specific headroom but not a transferable rule.

## Interpretation

The uncertainty gate behaved correctly: it removed the catastrophic tail and
returned exact persistence when competence was unsupported. The bottleneck is
the expert model, not confidence calibration. Euclidean contact transport has
no material topology and cannot distinguish nearby but disconnected strands
in the tangled rubber-band geometry.

Do not expand the confidence model or inspect the sealed episode. A future
Deform360 attempt needs either topology-aware graph-geodesic/ARAP transport or
a stronger learned dynamics backbone. For the PhysTwin 22-case benchmark, the
faster SOTA path is to apply the validated Bayesian anchor and calibration
layer to a fairly reproduced stronger backbone such as NeuSpring or MatPhys.

## Information boundary

- The 27-episode panel was already exhausted before this study.
- Rubber-band episodes 1, 3, 4, 6, 7, and 9 were source-development episodes.
- Router decisions use action, opening, frame-zero geometry, contact, and
  predicted-response features; they do not use object outcomes.
- Outcomes enter only source fitting and post-decision scoring.
- Rubber-band episode 0, fresh-object outcomes, confirmatory data, and the
  PokeFlex target were not read.
- No state-of-the-art claim follows from these active-window diagnostics.

## Verification

Focused router, transport, method-lock, and trust tests passed: `15 passed`.
Ruff and `git diff --check` pass on all intended files. The repository-wide
suite currently has unrelated collection failures in the pre-existing
temporal-residual tests and a missing optional `h5py` dependency; those are not
claimed as caused or fixed by this milestone.

