# Native PGRD temporal-head transfer

Run date: 2026-07-18

Status: the frozen three-action development gate failed. The exploratory
19-case future cohort remains sealed.

## Registered hypothesis

The zero-shot PGRD adapter failed because its residual output was trained
around PGRD's simulator rather than PhysTwin. This successor keeps the pinned
PGRD spatial encoder, retrains its temporal residual transformer on 18
PhysTwin source prefixes, and preserves the exact dense Bayesian endpoint
anchor. It is the smallest native-training test suggested by the zero-shot
diagnostic.

The protocol is frozen in
`configs/sota/phystwin_pgrd_native_development_v1.json`. Source supervision
ends at each released training boundary. The three sloth development tails
are excluded from fitting and can reject, but not confirm, the family. No
future frame is available to the trainer or gate.

## Implementation boundary

- PGRD commit `e294d96723054f77a1cfdd3c2c052de7b7cd9ce3` and sloth
  checkpoint SHA-256
  `79cc402835b73d6f7dc38a59ea37531f52ea3d2909d434ed9a2a8673509e073c`
  initialize the model.
- The PGRD spatial encoder is frozen; only the temporal transformer is
  trainable.
- Teacher-forced features use observed prefix states and the matching
  PhysTwin next state at a fixed 10 Hz cadence.
- Losses are averaged per frame, so dense points are not treated as
  independent episodes.
- A robust Smooth-L1 objective, L2 starting-point penalty, and 10 mm residual
  cap are fixed before validation.
- Validation uses one shared model and no target-specific fallback.
- The dense endpoint anchor is preserved exactly; only its predicted future
  change is interpolated from the sampled points.

## Frozen result

Training used 18 source cases and 422 cadence-aligned source frames. The mean
training loss fell from `0.05357` to `0.02911`, confirming that the temporal
head learned the source residual targets. That fit did not transfer.

Percent changes below are relative to exact endpoint persistence; negative is
better.

| Development case | CD change | Track change | Balanced improvement |
| --- | ---: | ---: | ---: |
| `single_lift_sloth` | -0.001% | -1.233% | +0.617% |
| `double_lift_sloth` | +1.320% | +1.418% | -1.369% |
| `double_stretch_sloth` | +1.923% | +3.704% | -2.813% |

The equal-case aggregate worsens CD by `1.081%` and track error by `1.296%`.
Only one of three cases improves both metrics, and balanced improvement is
`-1.188%`, below the locked positive `1%` gate. The machine-readable result is
`results/sota/diagnostics/pgrd_native_v1/summary.json`, SHA-256
`e409f1e921a68cc5a6219bf8cef3ca78420a4439a92b0c0ac9aececb553e719f`.

## Interpretation

Freezing PGRD's spatial representation while adapting only temporal dynamics
is insufficient. The head can fit source residuals, but its corrections do
not transfer across PhysTwin's double-action regimes. This closes temporal-
head-only adaptation; it does not reject learned residual physics generally.

A credible successor must change the spatial representation or simulator
interface itself. The next family must be registered separately and should
condition explicitly on the PhysTwin graph and realized controller/contact
motion. Retuning this head, choosing per-target trust, or opening the 19-case
future after this failure is forbidden.
