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

## Post-gate diagnostics

All diagnostics below use only the already-open four source suffixes. They are
not a second selection opportunity and do not authorize a 19-case future run.

The bank has limited oracle headroom. The balanced future oracle improves the
equal-object source aggregate by `6.54%` CD and `5.62%` track error. Requiring
the oracle choice to improve both metrics gives `6.62%` and `4.59%`, but only
cloth and dinosaur contain a non-teacher candidate that improves both metrics;
rope and zebra fall back to the released teacher. The complete oracle record is
`results/sota/phystwin_zero_order_topology_field_source_v1_oracle.json`.

A Bayesian ensemble does not solve the selection problem. Its strongest
aggregate softmax policy improves CD by `5.42%` and track error by `3.02%`, but
it records only one of four two-metric wins and is confidently wrong on rope.
The best policy by win count reaches only two of four wins and improves the
aggregate by `2.96%` CD and `1.41%` track error. An action-matched selector also
reaches only two of four wins; its best policy improves CD by `1.05%` while
worsening track error by `0.47%`.

Finally, composing each selected spring field with the fixed Bayesian anchor
does not explain the anchor discrepancy. Cloth improves both metrics, but its
required anchor RMS shrinks by only `0.14%`. Dinosaur and zebra do not improve
both metrics and require a *larger* anchor correction. The effects are additive
rather than mechanistically explanatory.

## Interpretation

The family fails for both proposal support and future-blind selection. A
piecewise static spring field can help particular objects, but neither
ensembling, action matching, nor composition with the endpoint anchor makes it
transfer reliably. This closes the static zero-order field branch on the
released cases. A credible successor must model a correction that evolves
during the rollout rather than enlarging this candidate bank.

No exploratory 19-case future run is justified for this family.
