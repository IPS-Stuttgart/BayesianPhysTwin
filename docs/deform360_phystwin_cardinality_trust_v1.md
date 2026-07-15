# Deform360 controller-cardinality trust hypothesis v1

## Discovery result

The frozen global v2 policy failed on the first unseen bimanual prehensile
action, episode 6 of `081-stripe-rope`. Weights `(a,b)=(0.4,0.1)` learned from
two unimanual grasps degraded its untouched tail by 41.5% in track RMSE and
64.4% in Chamfer. This rejects one shared aggregate action-response weight.

Episode 6 itself is nevertheless predictable from its permitted training
interval. Its source-only optimum is `(0.2,0.1)`, which improves the untouched
tail by 26.5% in track RMSE and 19.6% in Chamfer. Since the episode has two
controller attachments while episodes 1 and 4 have one, the rule

```text
a_effective = a_base / controller_count
```

maps the independently learned unimanual `a_base=0.4` to `0.2` without using a
future object observation.

After applying this rule, leave-one-action-out results on the three prehensile
discovery actions are:

| Held action | Controllers | Effective `a` | Track change | Chamfer change |
|---|---:|---:|---:|---:|
| episode 1, move center | 1 | 0.3 | -19.0% | -15.9% |
| episode 4, curl edge | 1 | 0.4 | -2.6% | +3.4% |
| episode 6, move both edges | 2 | 0.2 | -26.5% | -19.6% |

Negative changes denote improvement. The result is mechanistically plausible
and substantially more stable than global trust, but it is post-hoc because
the normalization was formulated after observing episode 6.

## Independent lock

The independent source test uses `002-rope-silk` under the episode split that
was already locked in the parent Deform360 replication protocol:

- source: episodes 0, 2, 5, 6, 7, and 9;
- calibration: episodes 3, 4, and 8;
- sealed target: episode 1.

The source panel contains two unimanual and four bimanual prehensile actions.
Its outcomes had not been read when the executable addendum
`deform360_cardinality_trust_002_rope_silk_v1.json` was written. The addendum
freezes automatic registration, physical-parameter controls, cardinality
normalization, leave-one-action-out evaluation, and pass/fail thresholds.

Calibration and target data remain forbidden unless the independent source
gate passes. Even a pass is not a state-of-the-art claim: the official
multi-object protocol remains required.

## Interpretation boundary

Controller-cardinality normalization is model-form shrinkage, not a law that
deformable responses add linearly. It compensates for the fact that the current
aggregate driven-minus-zero trajectory can grow with the number of bilateral
virtual attachments. A future unilateral contact model may make this shrinkage
unnecessary. The independent test asks only whether the rule transfers before
that larger mechanism is built.
