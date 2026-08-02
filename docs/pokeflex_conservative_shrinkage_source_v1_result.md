# PokeFlex Conservative Shrinkage Source v1 Result

## Decision

The frozen source gate passed. This authorizes a separately committed target
protocol and prediction/scoring implementation. It does not authorize target
archive access before that second lock, and it is not itself a SOTA result.

The canonical source result has SHA-256
`0075c331fc23ffadb2e9ebdd4b58093c76d25ce39c2bcf33e84d80d50a338bda`.

## Result

The selector chose
`checkpoint_action_local_state_relative_0.4_residual_scale_0.125`. The same arm
was selected in all nine leave-one-object-out folds. Every held object and
every full-panel object improved.

| Object | Released checkpoint CD_UL1 | Selected CD_UL1 | Improvement |
| --- | ---: | ---: | ---: |
| 3D-printed heart | 3.695 mm | 3.638 mm | 1.52% |
| 3D-printed pyramid | 2.369 mm | 2.364 mm | 0.21% |
| Beanbag | 4.876 mm | 4.836 mm | 0.82% |
| Foam cylinder | 4.432 mm | 4.308 mm | 2.81% |
| Foam dice | 5.631 mm | 5.564 mm | 1.19% |
| Memory foam | 2.350 mm | 2.290 mm | 2.53% |
| Plush moon | 7.589 mm | 7.555 mm | 0.45% |
| Plush octopus | 6.070 mm | 6.010 mm | 0.98% |
| Toilet-paper roll | 5.581 mm | 5.522 mm | 1.06% |
| **Equal-object mean** | | | **1.286%** |

The minimum object gain was 0.214%. Cross-fitted equal-object improvement was
also 1.286%, with 9/9 held-object wins and identical arm selection in every
fold. On all 131 frames without both an accepted graph update and action
support, the selected arm's recorded metric was exactly the released
checkpoint metric; there were zero fallback mismatches.

## Interpretation

The result explains why the earlier learned D405 guard was fragile. The useful
transferable component is small and broad across objects; choosing stronger
arms for larger source gains exposes a severe object-specific tail. The parent
protocol's strongest-shrinkage tie-break selects the unique arm that clears a
whole-object non-regression gate on all nine opened objects.

The remaining question is genuinely prospective: does this fixed weak update
beat the released checkpoint and the published 6.498 mm PokeFlex reference on
the eight still-sealed target objects while preserving the published Jaccard
score? The target run must generate and seal every prediction before any target
mesh is scored.
