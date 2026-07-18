# Short-horizon unrolled PGRD adaptation

Run date: 2026-07-18

Status: the frozen three-action development gate failed. The exploratory
19-case future cohort remains sealed.

## Registered hypothesis

The temporal-head-only PGRD transfer may have failed because teacher-forced
one-step targets did not train the model for recursive rollout. This successor
keeps the pinned PGRD point-transformer encoder frozen, trains its spatial
decoder and temporal transformer on five-step PhysTwin source-prefix rollouts,
and preserves the exact dense Bayesian endpoint anchor.

The protocol is frozen in
`configs/sota/phystwin_pgrd_unrolled_development_v1.json`. Training uses only
released source-case prefixes. The three sloth development tails are excluded
from fitting and can reject, but not confirm, the model family. No future frame
is available to training or model selection.

## Implementation boundary

- PGRD commit `e294d96723054f77a1cfdd3c2c052de7b7cd9ce3` and checkpoint
  SHA-256
  `79cc402835b73d6f7dc38a59ea37531f52ea3d2909d434ed9a2a8673509e073c`
  initialize the model.
- The point-transformer encoder is frozen. The spatial decoder (12,611
  parameters) and temporal transformer (67,590 parameters) are trainable.
- Each training example recursively advances five 10 Hz steps; predictions,
  rather than observed states, are fed back after the first step.
- Four complete windows per source case and epoch give cases equal expected
  weight despite unequal sequence lengths.
- A 2 mm Smooth-L1 transition, L2 starting-point penalty, and 10 mm residual
  cap are fixed before validation.
- Evaluation uses one shared model, exact endpoint persistence as the
  reference, and no target-specific fallback.

## Frozen result

Training used 18 source cases and 70 complete windows per epoch for 20 epochs.
The mean unrolled loss fell from `0.005618` m to `0.003756` m, a `33.1%`
reduction. This source fit did not transfer.

Percent changes below are relative to exact endpoint persistence; negative is
better.

| Development case | CD change | Track change | Balanced improvement |
| --- | ---: | ---: | ---: |
| `single_lift_sloth` | -0.452% | +0.362% | +0.045% |
| `double_lift_sloth` | +2.038% | +2.122% | -2.080% |
| `double_stretch_sloth` | +2.635% | +2.787% | -2.711% |

The equal-case aggregate worsens CD by `1.407%` and track error by `1.757%`.
No development case improves both metrics, and balanced improvement is
`-1.582%`, below the locked positive `1%` gate. The machine-readable result is
`results/sota/diagnostics/pgrd_unrolled_v1/summary.json`, SHA-256
`f9a2dbd25b2e6a26dc385ec8d7be8425c38a8ccb1c5c8886b2c7f8f9fe8b0a7b`.
The trained checkpoint SHA-256 is
`492db973cabb1987253b95fe7e3a568dd14157b4b668e3839178dc3119b245a1`.

## Interpretation

Recursive short-horizon training does not repair PGRD transfer to the
PhysTwin action regimes. It improves its registered source objective but
systematically underperforms the simpler persistent Bayesian endpoint anchor
on the untouched double-action tails. Together with the zero-shot and
temporal-head-only failures, this closes the current PGRD-adapter family.

The next credible route is a stronger physical backbone whose own causal
rollout already approaches the published state of the art, followed by a
Bayesian update that is evaluated as an addition rather than asked to replace
missing dynamics. A future PGRD revisit would require a graph- and
actuation-conditioned architecture trained on substantially broader physical
episodes, not another adjustment of this adapter.
