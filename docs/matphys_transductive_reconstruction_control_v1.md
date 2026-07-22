# MatPhys all-frame reconstruction control

## Question

The published MatPhys recipe reports lower PhysTwin metrics than our causal
held-future methods, but its public per-case script passes `--fit_all_frames`.
That setting optimizes each case on both the released train interval and the
released test interval. This control reproduces that information regime before
we compare numbers or add a Bayesian smoother.

It does **not** test future prediction. Its only question is:

> Under MatPhys's public all-frame, per-case fitting regime, can the pinned
> upstream implementation reproduce the published reconstruction point on the
> same 22 PhysTwin cases?

## Hard boundary

The runner
[`scripts/remote/run_matphys_transductive_reconstruction.py`](../scripts/remote/run_matphys_transductive_reconstruction.py)
requires `--acknowledge-future-observations` for both training and export. Every
artifact declares:

```text
future_observations_used: true
released_test_outcomes_used_in_objective: true
claim_boundary: offline per-case reconstruction control only
```

The artifact contract is separate from the causal MatPhys and
Bayesian-PhysTwin external-backbone contracts. It must never be supplied to a
predictive rollout, a sealed selector, or a Causal4D result.

## Frozen upstream recipe

The control uses the pinned public MatPhys commit and its per-case settings:

- one model per case;
- 200 epochs, evaluation every 10 epochs;
- tracking and geometry weights of 1;
- render loss 0;
- gradient scale 1000 and clip 5;
- residual log-stiffness scale 1 and soft clamp 0.25;
- acceleration smoothing 0.01;
- `--fit_all_frames` and `--save_best_only`;
- seed 42 and learning rate `3e-4`.

Training and export reject any other MatPhys commit or a dirty tracked source
tree. The audit also binds the exact PhysTwin teacher checkpoint loaded by the
upstream dataset, even though its associated teacher-loss weight is zero.

The numeric physics-prior flag is retained for command parity even though the
current upstream implementation documents it as ignored. Visualization is
disabled because it does not affect optimization.

MatPhys's public video loader lexicographically sorts PNG names before uniform
sampling. The wrapper intentionally preserves that behavior and records every
selected path and hash rather than silently correcting it.

## Execution order

1. Run one epoch on one development case with `--eval-every 1`.
2. Verify finite checkpoints, recorded all-frame access, trajectory export, and
   official metric evaluation.
3. Estimate the 22-case runtime from the measured epoch time.
4. Run the full control only if the smoke is stable and computationally
   reasonable.
5. Aggregate official train/test metrics while labeling both intervals as
   fitted reconstruction data.

Only after an upstream reproduction succeeds is it meaningful to test a
Bayesian all-frame smoother. The causal held-future table remains the primary
Bayesian-PhysTwin result and is reported separately.
