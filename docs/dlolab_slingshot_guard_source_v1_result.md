# Slingshot exact-fallback guard source result v1

## Decision

**Source gate failed. No fresh-world guarded study is authorized.**

The one registered source-development attempt completed normally on all 19 already-open
calibration worlds. All three active-prefix batches passed native QA, all candidate
decisions sealed before the parent calibration reward was read, and there were no
technical failures or replacements.

## Result

| Arm | Mean reward | Gain vs blind | 95% paired CI | Harmed worlds |
| --- | ---: | ---: | ---: | ---: |
| Blind prior | 7.035400 | 0.000000 | [0, 0] | 0/19 |
| Active MAP | 7.008758 | -0.026642 | [-0.046057, -0.009416] | 10/19 |
| Active Bayes | 7.022242 | -0.013159 | [-0.025323, -0.002190] | 7/19 |
| Cross-fitted guard | 7.035400 | 0.000000 | [0, 0] | 0/19 |

The cross-fitted guard chose exact fallback in every held-out fold and in the full
19-world fit. Relative to unguarded active Bayes, it recovered `0.013159` mean reward,
with paired 95% CI `[0.002190, 0.025323]`, and removed all seven harmed worlds.

This is not evidence that the active update improved the task. It is evidence that the
baseline-relative admission rule detected insufficient source support and preserved the
incumbent exactly.

## Why the gate rejected every update

Before source outcomes were read, the frozen candidate family produced between zero and
45 nonblind sensor decisions. After opening the already-registered calibration rewards,
every candidate with at least one update had negative mean gain:

- unpenalized active Bayes: `-0.013159` over 45 updates;
- moderate posterior-spread guards: approximately `-0.00391` to `-0.00445`;
- most selective nontrivial guard: `-0.003890` over four updates.

Thus exact fallback was not caused only by the delete-one stability threshold. No
nontrivial member of the frozen bank had positive in-sample mean gain.

## Interpretation

Together with fresh-world v2, this sharpens the controlled conclusion:

1. posterior integration is materially better than plug-in MAP;
2. the finite model bank is nevertheless misspecified enough that both can select
   harmful actions;
3. posterior spread inside that bank does not identify the model-form error;
4. an exact-fallback guard prevents regression when source transfer evidence is absent.

The last point is useful methodology, but it is not a positive control-performance
result. It does not justify a fresh-world v3 run or a SOTA, official-benchmark,
real-world, perception, material-identification, or physical-safety claim.

## Custody

- Frozen implementation: `38c6b950`.
- Result ID: `acc990372587512fd46c8dc485e5299718c9690490033c120a6efb4536259645`.
- One attempt; no retry or replacement.
- 3/3 prefix batches and 19/19 source worlds accounted for.
- Parent calibration reward read only after the decision barrier passed.
- Protected targets, held-v8, DLO4/DLO5, official DLO3 evaluation, GPUs, robots, and
  new recordings were not accessed.
- Verification is a second implementation/arithmetic check, not independent human
  review.
