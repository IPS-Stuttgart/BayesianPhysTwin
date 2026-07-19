# MatPhys spring proposal in the pinned PhysTwin runtime

## Status and boundary

This is an **exploratory bridge diagnostic**, not an independent SOTA result.
The five target futures were opened in the earlier MatPhys family experiment.
They may diagnose simulator parity, transfer, and selector behavior, but they
must not tune a confirmatory method or support a preregistered claim.

The frozen Causal4D claim and its real protocol are unchanged.

## Question

The earlier MatPhys fork produced encouraging validation scores but failed
simulator parity. This audit asks a narrower question:

> Does the source-trained MatPhys spring field still help when only `spring_Y`
> is transferred into the pinned released PhysTwin/Warp runtime?

That distinction is load-bearing. The candidate is a physical-parameter
proposal; the official runtime, initial state, controls, collision parameters,
and evaluation path remain those of released PhysTwin.

## Implementation

`run_matphys_causal.py` can now export the complete positive spring field as a
NumPy array in addition to the existing summary. The additive manifest field
preserves compatibility with existing consumers.

`bpt-build-phystwin-spring-overlay` validates shape, finiteness, and positivity,
then creates a checkpoint that replaces only `spring_Y`. It records hashes of
the source checkpoint, candidate field, output checkpoint, and summary.

The registered source-training exporter remained byte-identical on the server
(`ba4cd9a...`). Field extraction used a separate diagnostic copy
(`824007e...`), so it did not invalidate the frozen training audit.

## Selector

Selection uses only the held-out prefix interval
`[fit_end_frame, train_end_frame)`. The inherited rule accepts the candidate
only when:

1. neither validation CD nor validation track error regresses; and
2. their mean normalized score improves by at least 0.1%.

Rejected candidates fall back to the exact released checkpoint. No future
frame participates in selection.

## Exploratory result

All values below are millimetres. `B` is the released checkpoint, `C` the
MatPhys spring proposal, and the final column records the prefix decision.

| Case | Prefix CD B/C | Prefix track B/C | Future CD B/C | Future track B/C | Select |
|---|---:|---:|---:|---:|---|
| `single_clift_cloth_1` | 3.581 / 3.675 | 5.076 / 4.810 | 6.011 / 5.398 | 11.678 / 10.668 | baseline |
| `single_clift_cloth_3` | 1.940 / 1.998 | 7.284 / 7.861 | 4.164 / 4.661 | 13.210 / 15.369 | baseline |
| `double_lift_cloth_1` | 4.207 / 4.268 | 4.619 / 4.656 | 13.265 / 11.566 | 23.214 / 22.716 | baseline |
| `double_lift_zebra` | 5.054 / 4.686 | 7.867 / 5.941 | 15.086 / 12.406 | 30.989 / 25.710 | candidate |
| `single_push_sloth` | 9.921 / 35.708 | 15.802 / 66.454 | 13.719 / 88.761 | 22.587 / 137.169 | baseline |

The equal-case means are:

| Stack | Future CD (mm) | Future track (mm) | CD change | Track change |
|---|---:|---:|---:|---:|
| Released baseline | 10.449 | 20.336 | - | - |
| Ungated proposal | 24.558 | 42.326 | +135.03% | +108.15% |
| Prefix-selected + exact fallback | **9.913** | **19.280** | **-5.13%** | **-5.19%** |
| Future both-metric oracle | 9.450 | 18.978 | -9.56% | -6.67% |

The zebra gain survives the pinned runtime: future CD falls by 17.76% and
track error by 17.03%. The same proposal is catastrophic on sloth, where the
prefix gate rejects it decisively. This makes exact fallback a method
requirement rather than presentation polish.

Two rejected cases improve both future metrics. That is an oracle diagnostic,
not permission to relax the selector after seeing their futures. It indicates
that a source-trained, prefix-calibrated trust-region selector is a worthwhile
independent hypothesis.

## Interpretation

The fastest credible route toward stronger raw accuracy is now:

1. generate a spring-field proposal using source-only representation learning;
2. replay proposal candidates in the pinned upstream simulator;
3. select or interpolate them from a target prefix under a frozen rule;
4. retain an exact released-checkpoint fallback; and
5. apply Bayesian state/discrepancy inference after the physical prior passes.

This is safer than an ungated neural residual and more direct than another
output-space correction. The present 5.2% mean improvement is promising but
does not yet close the published benchmark gap, and the cohort is neither
independent nor broad enough to claim SOTA.

## Next gate

Before an independent run, freeze on source data:

- a small log-space trust-region grid between released and proposed springs;
- a prefix-only selector with exact fallback;
- replay-variance handling in the selection score; and
- Bayesian uncertainty propagation for accepted spring fields.

The independent target must remain unopened until that package is fixed. A
larger run is justified because the pinned selector produced a material,
two-metric gain while blocking a six-fold failure. It is not justified to run
the current ungated proposal or to claim that the MatPhys fork itself beat the
released runtime.
