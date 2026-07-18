# Per-object zero-order topology and spring field

Lock date: 2026-07-18

Status: nested source-transfer gate failed; family rejected without opening
future metrics.

## Why this successor exists

The fixed density-matched sparse topology improved the sloth development case
but failed all four transfer objects. That rejects one universal edge-density
profile. It does not test the mechanism used by NeuSpring: object-specific
zero-order topology identification followed by a spatial spring field.

This family keeps the released PhysTwin graph and per-edge spring field as an
exact candidate, then adds three deterministic search arms:

1. `topology_only`: region-specific radius and maximum-neighbour multipliers;
2. `field_only`: one global object scale, one controller scale, and five
   region log-stiffness scales on the exact topology; and
3. `joint`: all topology and field coordinates together.

An object edge receives the average log-scale of its endpoint regions. Every
changed topology first preserves total released object stiffness, so topology
search is not merely a proxy for uniform softening. Eight Latin-hypercube
candidates are frozen per arm, plus the exact teacher, for 25 official Warp
replays per object.

## Nested information boundary

The runner never receives the complete released prefix during search. For an
outer source split ending at frame `T_fit`, a second typed prefix artifact is
created with exactly frames `[0, T_fit)` plus one unobserved hold sentinel.
Candidates are selected only on the final quarter of that fit-only artifact.

```text
released observation prefix
|---------------- outer fit ----------------|---- sealed source suffix ----|
|------------ inner fit -----------| selection |
                                     ^ search sees nothing to the right
```

After selection, the chosen topology artifact is rerun once on the outer
prefix. Only the locked source gate may then read the sealed suffix. Exact
teacher fallback is decided inside the nested fit artifact, never from outer
validation.

## Development result

On `single_lift_sloth`, selecting from all observed fit frames chose
`field_005`: it improved fit CD/track by `3.76%/3.40%` but degraded the outer
suffix by `2.64%/0.77%`. That is direct overfitting evidence.

The nested selector chose `field_003`. It improved fit CD/track by
`1.37%/3.87%` and the untouched outer suffix by `4.22%/3.13%`. The selected
candidate changes only the spatial spring field, while topology-only and joint
arms remain explicit ablations in the same bank.

## Frozen source gate

The machine-readable lock is
`configs/sota/phystwin_zero_order_topology_field_source_v1.json`. The transfer
objects are zebra, cloth, rope, and dinosaur. The family passes only with:

1. a non-teacher nested selection in at least three of four objects;
2. outer-suffix CD and track improvements in at least three of four objects;
3. improvements in both equal-case aggregate metrics; and
4. at least `3%` aggregate balanced improvement.

Failure rejects the complete family without opening any 19-case future metric.
Passing permits only an exploratory run on that previously examined cohort;
independent confirmation remains necessary for a state-of-the-art claim.

## Source result

The nested selector chose a non-teacher candidate in three of four objects but
failed the locked transfer gate. The machine-readable result is archived at
`results/sota/phystwin_zero_order_topology_field_source_v1_summary.json` with
SHA-256
`fffd52f44bd8aab66c0ba65623fe6b0dc68bef6acb1be7b4b1943fc4f27e0d92`.
All replays used Bayesian-PhysTwin commit `95dc8b4` and official PhysTwin commit
`2b66305`.

| Transfer case | Nested selection | Outer CD change | Outer track change | Both improve |
| --- | --- | ---: | ---: | --- |
| `single_lift_zebra` | exact teacher | 0.00% | 0.00% | no |
| `single_lift_cloth` | `joint_000` | -10.91% | +0.34% | no |
| `single_lift_rope` | `field_005` | +6.29% | +6.92% | no |
| `single_lift_dinosor` | `topology_004` | +5.26% | +29.68% | no |

The selected stack records `0/4` two-metric wins. Equal-case CD improves from
`10.925` to `10.291 mm` (`-5.80%`), but track error worsens from `20.895` to
`21.649 mm` (`+3.61%`). Balanced improvement is `1.10%`, below the locked `3%`
threshold, so the gate selects the exact teacher and records
`future_metrics_opened=false`.

## Interpretation

Object-specific zero-order search can materially improve one metric, as the
cloth CD result shows, but late-fit selection does not yet identify candidates
that jointly transfer in CD and material-point tracking. The next diagnostic
must separate proposal support from selection failure: rerun the frozen bank on
these already-open source suffixes and measure its two-metric oracle. If the
oracle wins, improve the future-blind selector; if it does not, replace the
piecewise field model rather than expanding the same search.

No exploratory 19-case future run is justified for this family.
