# Teacher-centered part-pair spring refit

Run date: 2026-07-18

Status: source gate failed; family rejected without opening future metrics.

## Hypothesis

The global hierarchy failed because useful stiffness movement is strongly
case-dependent. The shared MatPhys graph-part decoder failed for another
reason: every source spring saturated at the same lower bound, so it learned
uniform softening rather than parts.

This family keeps the released per-edge spring field and fits only a small
continuous correction:

```text
log(k_e) = log(k_e^PhysTwin) + beta_pair(part(i), part(j)).
```

The unordered endpoint pair distinguishes within-part springs from cross-part
springs. All controller springs retain one separate scale. L2 shrinkage keeps
the offsets near the released teacher without imposing the saturated
factor-two `tanh` cap.

This is inspired by the spatial-associativity and piecewise-topology findings
reported by NeuSpring. It is not a reproduction: the graph is unchanged and
cross-part stiffness is a continuous topology proxy.

## Causal boundary

`bpt-build-phystwin-prefix` writes a fitting payload containing only the
permitted object/controller prefix and one unobserved hold sentinel. Mutating
any future object trajectory leaves all three fitting payloads byte-identical.
The source runner receives the previously locked DINO graph partition but no
future RGB, depth, masks, tracks, object points, or controls.

The five source cases, fit split, optimization settings, exact-teacher
fallback, and three-of-five acceptance rule are frozen in
`configs/sota/phystwin_part_pair_source_v1.json`. No future metric may be
opened unless this prefix-only gate passes.

## Source result

The gate was applied mechanically with
`bpt-gate-phystwin-part-pair-source`. The machine-readable result is archived
at `results/sota/phystwin_part_pair_source_v1_summary.json` with SHA-256
`5d565b9e64e1c2b4132ede41066a2a57d24aacbbafe84cfc766279f70db7204e`.
All refits used code commit `03a05e49749ac6661f9da9b00e9375ad42960897`
and official PhysTwin commit `2b6630528141b9cba5a7677c8b88b2129b4a8390`.

| Source case | Balanced improvement | CD change | Track change | Maximum absolute object log scale | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `single_lift_sloth` | +0.0116% | -0.0049% | -0.0183% | 0.000091 | teacher |
| `single_lift_zebra` | +0.0006% | -0.0016% | +0.0005% | 0.000370 | teacher |
| `single_lift_cloth` | +0.0630% | +0.1494% | -0.2754% | 0.000147 | teacher |
| `single_lift_rope` | +0.0372% | +0.0128% | -0.0873% | 0.000188 | teacher |
| `single_lift_dinosor` | +0.0030% | +0.0000% | -0.0061% | 0.000157 | teacher |

No case clears the locked 0.1% balanced-improvement floor while preserving
both metrics. The acceptance count is therefore `0/5`, below the required
`3/5`. The all-candidate validation mean is `10.108 mm` CD and `18.513 mm`
track error; exact-teacher fallback gives `10.100 mm` and `18.547 mm`.
Consequently the selected stack is byte-for-byte the all-teacher stack.

The zero-update smoke trajectory is exactly identical to the runner's teacher
baseline. Its `0.220 mm` vector RMSE against the historical released replay is
well below the measured Warp replay floor. The negative is therefore not an
identity-path failure.

## Interpretation

This result rejects the locked five-part continuous correction, not spatially
varying mechanics in general. Every fitted object log scale remains below
`3.8e-4`; the differentiable prefix objective effectively leaves the released
spring field unchanged. Retuning the partition, learning rate, regularizer, or
source cases after seeing this validation is forbidden.

Any successor should be registered as a new family and should address the two
mechanisms absent here:

1. derivative-free or otherwise scale-robust fitting of a smooth spatial
   spring field; and
2. explicit piecewise topology proposals rather than treating cross-part
   stiffness as a tiny continuous perturbation.

The successor must repeat the same future-blind source gate. This failed family
does not justify a released-future or independent-cohort run.
