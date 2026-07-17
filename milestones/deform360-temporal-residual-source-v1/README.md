# Deform360 temporal residual source v1

This milestone tests whether a PGRD-inspired causal temporal residual is the
missing ingredient on top of the trusted reusable physical arm. It uses only
the 27 already-open source episodes and holds out `092-squirrel` as an outer
object fold.

## Model

The residual carries a per-node GRU state through an eight-frame causal context
and is trained with an eight-step open-loop loss. Its outputs are equivariant
residual velocity, heteroscedastic variance, and utility probability. A nested
calibration set chooses the utility threshold. Rejection returns the physical
trajectory exactly per node.

## Result

Training ran for 5,000 steps over 18 episodes and reduced its source objective,
but the deterministic residual did not transfer:

| Arm | Future track (m) | Future CD (m) | Late track (m) | Late CD (m) |
| --- | ---: | ---: | ---: | ---: |
| Trusted physical fallback | **0.003449** | **0.002750** | **0.005402** | **0.003910** |
| Deterministic temporal residual | 0.007655 | 0.005006 | 0.011948 | 0.007247 |
| Gated temporal residual | **0.003449** | **0.002750** | **0.005402** | **0.003910** |

The deterministic arm worsens future track error by 121.93% and future CD by
82.04%. On the four inner calibration episodes, full admission has combined
relative score 1.326 and a maximum degradation of 80.25%. Calibration selects
`utility_threshold=1.1`, so the gated output is exact abstention.

## Interpretation

This is a source-development negative, not evidence that temporal residuals
never help. It establishes that additional temporal capacity, by itself, does
not solve cross-object transfer in the current low-data setting. The result
retires this architecture before any confirmatory outcome is opened. The next
experiment scales the admitted automatic-registration and trusted-physics arm.

No penguin held episode, PokeFlex target, or newly selected Deform360 object was
opened for this milestone.
