# Teacher-centered part-pair spring refit

Run date: 2026-07-18

Status: source protocol locked; source validation not yet opened.

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
