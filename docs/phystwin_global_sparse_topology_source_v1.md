# Density-matched sparse PhysTwin topology

Lock date: 2026-07-18

Status: source transfer gate failed; family rejected without opening future
metrics.

## Hypothesis

The failed part-pair refit kept the released graph fixed and learned object
log-scale changes below `3.8e-4`. NeuSpring's strongest ablation instead changes
both topology and a spatial spring field. This successor tests the cheapest
missing mechanism first: a sparser object graph around the released PhysTwin
teacher.

The fixed candidate applies the following rule to all five previously locked
DINO graph regions:

```text
object radius             = 0.80 * released radius
object maximum neighbours = round(0.75 * released maximum neighbours)
controller topology       = unchanged
```

Removing edges also lowers aggregate stiffness. To isolate connectivity from
uniform softening, retained object springs are multiplied by

```text
sum(released object spring values)
---------------------------------- .
sum(transferred candidate values)
```

No spring, damping, collision, or discrepancy parameter is optimized. The
official nonlinear Warp backend simply replays the typed candidate artifact.

## Development smoke

`single_lift_sloth` was used to define the candidate and is excluded from the
transfer gate. The all-ones topology artifact reproduced the exact-teacher
trajectory byte for byte (SHA-256
`7a7d10acae79c9a61573722b1025e6836b2072bd5b4f1d27197861cc3e7a2b75`).

| Development topology | Fit CD (mm) | Fit track (mm) | Untouched suffix CD (mm) | Untouched suffix track (mm) |
| --- | ---: | ---: | ---: | ---: |
| Exact identity | 2.778 | 3.416 | 6.799 | 9.157 |
| Sparse, raw teacher values | 3.408 | 4.019 | 6.234 | 7.913 |
| Sparse, total-stiffness matched | 3.093 | 3.728 | 6.489 | 8.489 |
| Dense, total-stiffness matched | 2.897 | 3.571 | 7.957 | 10.705 |

The density-matched sparse graph improves the untouched suffix by `4.55%` CD
and `7.30%` track error. The normalized dense control degrades both metrics.
This isolates a connectivity signal more cleanly than the raw sparse result,
although it remains one inspected development interaction.

## Frozen transfer gate

The machine-readable lock is
`configs/sota/phystwin_global_sparse_topology_source_v1.json`. The transfer
panel is:

```text
single_lift_zebra
single_lift_cloth
single_lift_rope
single_lift_dinosor
```

For every case, only the released observation prefix plus one unobserved hold
sentinel may enter the runner. The final 25% of that prefix is a sealed source
transfer suffix. The gate verifies input hashes, exact identity behavior,
connectedness, absence of isolated nodes, strict edge reduction, and equal
total object stiffness before reading its metrics.

The fixed family passes only when:

1. candidate CD and track error both improve in at least three of four cases;
2. equal-case candidate means improve both metrics; and
3. the aggregate balanced relative improvement is at least `3%`.

There is no validation-selected per-case fallback. Failure rejects the complete
family and retains the exact released teacher.

## Claim boundary

Passing this gate permits an exploratory run on the previously examined
19-case cohort. It does not establish state of the art. An SOTA claim still
requires a separately locked independent evaluation and comparison under the
same metric contract.

No future PhysTwin, MotionCrafter, Prob4D, or VGGT observation may be used to
fit or select this topology. Prob4D remains a later observation model; it is
not part of this physical-backbone source test.

## Source result

The fixed candidate failed the locked transfer gate. The machine-readable
result is archived at
`results/sota/phystwin_global_sparse_topology_source_v1_summary.json` with
SHA-256
`a2160efaf9ba643b05ef0465f3b81630cc73f1bb87e0091c2f5665af99605f37`.
The replays used Bayesian-PhysTwin commits `11044f4` and `6d03d8c` and official
PhysTwin commit `2b66305`. The second Bayesian-PhysTwin commit only makes an
unchanged topology's density normalization exactly zero; candidate artifacts
and trajectories were not altered after their runs.

| Transfer case | Identity CD (mm) | Candidate CD (mm) | Identity track (mm) | Candidate track (mm) | Balanced change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_lift_zebra` | 4.509 | 4.835 | 8.398 | 8.286 | -2.94% |
| `single_lift_cloth` | 28.730 | 32.109 | 60.152 | 95.684 | -35.42% |
| `single_lift_rope` | 4.543 | 4.914 | 7.261 | 7.716 | -7.21% |
| `single_lift_dinosor` | 5.916 | 9.038 | 7.768 | 17.445 | -88.67% |

The candidate records `0/4` two-metric wins. Equal-case CD rises from
`10.925` to `12.724 mm` (`+16.47%`) and track error rises from `20.895` to
`32.283 mm` (`+54.50%`). The gate therefore selects the exact teacher and
records `future_metrics_opened=false`.

## Interpretation

The development sloth result does not transfer as a universal edge-density
prior. This rejects the fixed global `0.8/0.75` topology profile, not topology
inference itself. The large differences across objects support the narrower
hypothesis that topology and spring fields must be identified per object or
region, as in NeuSpring, rather than imposed globally.

No exploratory 19-case future run is justified for this family. A successor
must provide a future-blind per-object selector, retain exact-teacher fallback,
and pass a new source gate before any released future metrics are opened.
