# Dynamic Pairwise Belief: Open27 Source V1

## Status

Implementation-locked source-development protocol. No Open27 outcome from the
new 64-identity measurement pool may be computed before this protocol and its
implementation are committed. This study may use only the 27 already-open
Deform360 episodes across five objects. It must not inspect held-v8, any fresh
object, or any sealed PokeFlex target.

## Question

The strongest executable source result is a dual physical/persistence
backbone with pairwise-consensus RBF updates. It reaches 7.441 mm
object-balanced hidden identity RMSE and 6.795 mm hidden Chamfer, but it uses a
fixed set of 16 frame-zero identities. Separate source studies show that
fixed-identity observation providers often lose support as the action unfolds.

This experiment asks whether AllTracker's already-dense causal flow can support
a larger 64-identity frame-zero pool from which a small trustworthy subset is
selected independently at each update:

```text
64 causal material candidates
  -> residual-independent multiview/action support
  -> nuisance-aware preselection of 24
  -> exact pairwise correspondence consensus
  -> nuisance-aware active selection of 16
  -> recursive dual-backbone RBF update
  -> physical response and continuation guards
  -> corrected continuation or exact selected-backbone fallback
```

This is an online Bayesian-PhysTwin observation experiment. It does not revive
V12, V13, V14, DEFORM-DLO2, or any held-v8 attempt.

## Frozen Measurement Path

The existing raw-camera AllTracker builder is run with `--center-count 64`.
Its default remains 16, preserving the frozen earlier path. For update frame
`u`, the tracker reads only RGB frames `[0, u]`. No future image, target
trajectory, manual future identity, or outcome manifest enters measurement
construction.

The observation model uses only residual-independent cues:

- number of triangulation-inlier cameras;
- median multiview reprojection error;
- registered action support;
- frame-zero geometry and nuisance-aware information gain.

At least three inlier views are required. Metric variance has a fixed
`(5 mm)^2` floor. Total reliability mass is capped at eight effective samples,
so duplicating correlated evidence cannot create unbounded confidence. The
state innovation is processed once by the recursive Student-t update; it is
never recycled as prior perception reliability.

## Frozen Candidate

At frames 19, 38, and 57:

1. Build a rank-4 physical-response basis from the causal simulated prefix.
2. Admit finite candidates with at least three inlier views, action support of
   at least 0.10, and positive residual-independent reliability.
3. Select at most 24 candidates by nuisance-marginalized information gain.
4. Apply the exact pairwise gate with the existing 30 mm plus 10% strain
   envelope, at least nine inliers, and at least 70% consensus.
5. Require at least three causal motion centers, 0.5 mm physical motion,
   0.5 mm observed motion, and a Huber physical-agreement gain of at least 0.40.
6. Select at most 16 active inliers by nuisance-marginalized information gain.
7. Update separate recursive RBF beliefs for the physical and persistence
   backbones using calibrated metric variance and capped prior reliability.
8. Accept only if the decoded correction has nonnegative cosine with the
   physical continuation and no more than twice its RMS magnitude.
9. On any rejection or numerical failure, preserve the selected backbone
   byte-for-byte and leave the recursive belief state unchanged.

Archive row order is not evidence. Pool identities and their diagnostics are
canonicalized by material ID before any tie-broken selection.

## Frozen Arms

| Arm | Purpose |
| --- | --- |
| Fixed-16 selected backbone | No-update control from the established provider |
| Fixed-16 pairwise RBF | Strong source comparator with separate backbone beliefs |
| Dynamic-pool selected backbone | No-update control under the new support rules |
| Dynamic 16-of-64 guarded RBF | New candidate |

Every arm is scored on the same hidden identities and future frames. All 64
pool identities are excluded permanently from both directions of both hidden
metrics, including the fixed-16 controls.

## Advancement Gates

A genuinely fresh preregistered evaluation is justified only if all gates pass:

1. both co-primary object-balanced metrics improve by at least 1% over the
   fixed-16 pairwise RBF comparator;
2. both object-clustered 95% difference intervals have upper endpoints below
   zero;
3. late hidden identity RMSE improves;
4. at least four of five object means improve jointly;
5. no object regresses by more than 2% on either primary metric;
6. every rejection is byte-identical to its selected backbone;
7. every acceptance passes the frozen multiview, physical-motion, agreement,
   direction, and magnitude guards;
8. all 64 observed identities are excluded from every score.

Failure closes this exact source arm. Open27 outcomes may diagnose the frozen
method but may not tune its thresholds.

## Claim Boundary

Open27 is already outcome-open. A positive result would demonstrate source
headroom and justify a new independent protocol; it would not establish SOTA,
calibration, or non-regression. A negative result would be a valid stop. No
held-v8 or other prospective artifact is authorized by this protocol.

The target-free predictor is
`predict_dynamic_pairwise_belief_arrays`. The outcome-opening evaluator is
`evaluate_dynamic_pairwise_source`.
