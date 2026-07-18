# Deform360 causal contact transport: source-development protocol

## Purpose

The raw reusable-PhysTwin source arm showed that a dense Euclidean spring graph
can transmit local gripper motion too broadly. This addendum tests a narrower
alternative: transport the frame-zero material points only where contact is
causally supported by the known gripper trajectory and aperture.

This is a source-development diagnostic. It is not a replacement for
Bayesian-PhysTwin, a direct Deform360 leaderboard result, or part of the frozen
Causal4D main-paper claim.

## Information boundary

The candidate grid was specified after inspecting outcomes from source episodes
1, 3, and 4. Those episodes are therefore exploratory method-development data.
The remaining source episodes are used for source-only transfer diagnostics.

For a held episode, the method may use:

- frame-zero object points;
- the known future robot trajectory;
- the known future gripper aperture.

It may not use future object geometry, future tactile observations, a held
outcome-derived contact schedule, or any PokeFlex target data. A newly acquired
contact receives zero transport gain in this version, so it falls back exactly
to persistence until a separately locked transition model demonstrates source
transfer.

## Candidate family

Each causally active initial gripper induces a rigid transform relative to its
onset pose. A point at onset distance `d` receives transport weight

```text
gain * exp(-d / (base_scale + growth * gripper_travel)).
```

The grid contains translation and SE(3) transport, three base scales, four
growth factors, and two initial-contact gains. Exact persistence is included
once, for 49 candidates total. Multiple grippers are fused conservatively: the
transport direction is a weighted average and its total influence is bounded
by the strongest group. Duplicating a correlated gripper block therefore does
not increase confidence or displacement.

## Selection and gate

Candidates are scored with the existing independent Deform360 metric contract:
equal weight on future identity-aware track error and Chamfer distance, each
normalized by persistence. The pooled candidate is selected from source
episodes only. Leave-one-action-out folds also compare pooled selection against
exact persistence and against the median candidate selected from one source
episode.

The development-held outcome remains sealed unless all source gates pass:

- pooled selection beats persistence in at least two thirds of source folds;
- pooled selection beats the single-source median in at least two thirds;
- mean leave-one-action-out normalized score is at most 0.98.

Failure is informative: it means that this fixed causal transport family does
not transfer well enough to justify held evaluation. Action-conditioned expert
selection may then be developed on source data, but it must receive a new lock
and repeat the same transfer test before any held outcome is opened.

## Claim boundary

PGRD and DeformMaster already establish that physics-plus-neural-residual models
can improve deformable simulation. The possible contribution here is different:
causal contact realization, source-validated model-form selection, and exact
persistence fallback when an expert lacks support. Any state-of-the-art claim
still requires an independently locked multi-object panel and parity with the
official Deform360 split, horizon, and evaluator.
