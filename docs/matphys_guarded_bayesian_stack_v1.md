# Guarded learned-physics plus Bayesian stack

## Status

This is an **exploratory five-case result after future opening**. It is the
strongest local SOTA hypothesis found so far, but it is neither a full-cohort
comparison nor independent evidence.

The result does not modify the frozen Causal4D claim.

## Method

The stack has two complete, independently usable families:

1. **Incumbent:** released PhysTwin trajectory followed by the frozen
   validation-selected Bayesian anchor or last-residual correction.
2. **Proposal:** source-trained MatPhys spring proposal injected as `spring_Y`
   into the pinned upstream PhysTwin runtime, prefix-gated against the exact
   checkpoint, then followed by the same Bayesian/last-residual overlay.

The incumbent is always available as an exact fallback. A proposal family may
replace it only if:

- the zero-change replay remains within 1 mm coordinate RMSE of the released
  trajectory over the simulated known-action rollout;
- its validation score improves by at least 0.1%; and
- neither validation CD nor validation track error regresses.

The replay-stability check compares simulator outputs, not observations. It
therefore does not inspect held-out geometry or tracks. It prevents a numerically
unstable replay family from winning on a short prefix and diverging later.

## Why the stability gate is necessary

`double_lift_cloth_1` is the decisive control. Its proposal family improves the
prefix by 0.50% and would be selected by an ordinary validation gate, but its
zero-change replay differs from the released trajectory by 5.08 mm coordinate
RMSE. The future then degrades from 6.95/14.38 mm to 13.27/23.21 mm CD/track.

`single_push_sloth` has 5.25 mm identity-replay drift and is also excluded.
Cloth-3 and zebra have 0.41 and 0.18 mm drift and pass. This numerical gate is
orthogonal to whether the learned physical proposal itself is accurate.

## Exploratory result

Lower is better; values are equal-case means in millimetres.

| Method | Future CD | Future track | Change vs incumbent |
|---|---:|---:|---:|
| Incumbent family | 8.2627 | 16.0830 | - |
| Proposal family everywhere | 9.2025 | 17.7120 | +11.37% / +10.13% |
| Guarded family selector | **7.7382** | **15.4517** | **-6.35% / -3.92%** |

The selector retains the incumbent on `single_clift_cloth_1`,
`double_lift_cloth_1`, and `single_push_sloth`. It selects the proposal on:

| Case | Incumbent CD/track | Selected CD/track | Prefix basis |
|---|---:|---:|---|
| `single_clift_cloth_3` | 4.239 / 13.667 | 4.064 / 13.142 | both validation metrics improve |
| `double_lift_zebra` | 14.397 / 26.025 | 11.950 / 23.393 | both validation metrics improve |

Numerically, 7.738 mm is below MatPhys's published 8 mm CD and 15.452 mm is
about 3% above its published 15 mm track result. That is **not a head-to-head
SOTA result**: the external number is a full-22 paper value, whereas this is a
previously examined five-case subset.

## Novel insight

The useful composition is not a fixed blend and not a wholesale simulator
replacement:

```text
source-trained physical proposal
-> pinned-runtime identity and stability check
-> prefix-only exact fallback
-> Bayesian discrepancy update
-> cross-family fallback to the incumbent
```

Each layer addresses a failure observed locally:

- the source model contributes a large physical-prior gain on transferable
  cases;
- the pinned replay removes fork-specific simulator artifacts;
- the identity gate catches numerical instability;
- prefix selection rejects object-specific proposal failures; and
- Bayesian anchoring removes remaining low-rank discrepancy.

## Path to a real SOTA test

The five cases are exhausted. The next credible experiment is grouped
out-of-fold evaluation over all 22 PhysTwin cases:

1. group cases by physical object;
2. train the spring proposer without the held-out object's interactions;
3. export a complete spring field for each out-of-fold case;
4. run the identity replay and proposal in the pinned runtime;
5. apply the frozen within- and cross-family gates;
6. fit the unchanged Bayesian overlay from the permitted prefix;
7. aggregate all 22 cases and compare paired trajectories with MatPhys and
   NeuSpring paper values; and
8. report replay spread and calibration, not only point estimates.

The 1 mm threshold must be frozen from source-only replay controls before that
run. An external sealed cohort is even stronger, but it should not replace the
full-22 out-of-fold comparison because published SOTA numbers use that
benchmark.

## Evidence

- gate summary: `results/sota/diagnostics/matphys_guarded_bayesian_stack_v1/summary.json`
- gate summary SHA-256: `e4e07a5616062df6e5b5405eed75035bee3d93235ed9bb24944103f6834bf160`
- incumbent overlay SHA-256: `cbf917e2dde31fe91551838e0f9fe9bc2092fc63b25265e2b51e8b173a58fffd`
- proposal overlay SHA-256: `3f3e685d5ac1a90e720b9014165aa8ebd26463ec1f1a46a5817663fe44820a28`
- stability-control manifest SHA-256: `3da6cd832d07f85981eb2cf2727b6d921fef3d62594376e99845e93d236ffd5f`
