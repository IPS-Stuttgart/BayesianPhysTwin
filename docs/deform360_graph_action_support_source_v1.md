# Graph-local action support: source discovery v1

## Question

The automatic frame-zero PhysTwin rollouts produced a useful causal response,
but the raw driven-minus-zero field often moved too much of the graph. This
source-only experiment asks whether known contact topology can localize that
response without using a future object residual.

The candidate prediction is

```text
frame-zero state
+ 0.9 * exp(-graph distance to contact / 0.12 m)
      * (driven Warp rollout - zero-action Warp rollout).
```

An action response of zero is exact frame-zero persistence. The support prior
depends only on the frame-zero graph and registered contact anchor. It cannot
inspect the PhysTwin innovation or any future object observation.

## Frozen source construction

- Build an automatic episode twin from frame-zero multiview geometry.
- Retain 384 observed graph nodes and any latent connectivity nodes.
- Attach up to 16 graph-local material nodes within 30 mm of each contact
  anchor; fewer are allowed when the graph contains fewer admissible nodes.
- Run matched official-Warp driven and zero-action rollouts.
- Use a 50 mm/s multiview motion-consensus threshold with at least two camera
  contributors for the source diagnostic target.
- Select one support scale and action gain on frames 1--59 of scarf episode 1
  and squirrel episode 0. Frames 60--75 are untouched source tails.
- Freeze the selected values before evaluating rope episode 0.

The current observation staging marks all retained target points valid and
does not yet retain contributor counts or metric covariance. The experiment is
therefore point-estimation evidence, not a calibrated uncertainty result.

## Result

| Role | Episode | Track, mm | Persistence, mm | Track gain | CD, mm | Persistence, mm | CD gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selection | `085-scarf-cloth/1` | 16.45 | 17.65 | 6.8% | 9.29 | 9.78 | 5.0% |
| Selection | `092-squirrel/0` | 1.74 | 2.99 | 42.0% | 0.28 | 0.46 | 39.2% |
| Held-out source transfer | `002-rope-silk/0` | 17.44 | 35.95 | 51.5% | 8.18 | 19.25 | 57.5% |

The selected source-train combined score improves over persistence by 20.37%.
Its untouched source-tail score improves by 23.26%. The untouched-tail oracle
selects the same 120 mm support scale and changes only the action gain from 0.9
to 0.8, with a negligible score difference. The frozen candidate then improves
the held-out rope tail by 54.49% on the combined score without retuning.

## Interpretation

The result supports a specific hypothesis: PhysTwin's action response contains
useful directional information, but its spatial support should be regularized
by material-graph distance from contact. This is stronger than adding a learned
residual because the response is an interventional difference and the support
uses no outcome residual.

It is not yet a state-of-the-art result. Only three source episodes have been
examined, the graph is reconstructed automatically per episode rather than
reused unchanged across episodes, and uncertainty calibration is absent. The
published Deform360 comparison may be made only after the independent source,
calibration, and sealed target gates pass under the official metric protocol.

## Independent gate

The candidate is now locked in
`configs/causal4d_public/deform360_graph_action_support_independent_source_v1.json`.
The three discovery episodes are excluded. The next evaluation contains 27
untouched source episodes across five objects. It must improve execution-balanced
track and Chamfer errors by at least 5%, improve both late metrics by at least
3%, jointly win at least 18 episodes, and show no object-level median
degradation. Every gate is conjunctive. Calibration and target outcomes remain
sealed on failure.

## Bayesian continuation

If the point predictor passes independently, Bayesian-PhysTwin should place a
posterior over physical particles, contact-patch support, and action-response
scale while preserving the exact persistence component. Observation covariance
must then use retained contributor counts, clustered multiview uncertainty, and
metric variance before NEES or coverage is reported. The independent point gate
comes first so uncertainty is not calibrated around a failing mean model.
