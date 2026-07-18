# Per-object zero-order topology and spring field

Lock date: 2026-07-18

Status: nested source-transfer protocol locked; transfer suffixes not inspected
for this family at lock time.

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
