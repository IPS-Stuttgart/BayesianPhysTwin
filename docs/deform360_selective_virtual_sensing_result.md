# Deform360 Selective Virtual Sensing V1 Result

## Status

This is the frozen result of protocol
`deform360-selective-virtual-sensing-v1`. The complete prediction cohort was
sealed before any selected future target was opened. One of 24 locked episodes
failed the target-free frame-zero reconstruction quality check and was excluded
without replacement. The remaining 23 episodes cover all 12 objects, with four
objects in each of the filament, sheet, and volumetric strata.

The prospective paper threshold fails decisively. These outcomes may be used
for diagnosis and future method development, but not for selecting a
replacement arm or claiming confirmation on this cohort.

## Prospective Result

Metrics are object-balanced means over hidden material points and future-only
frames after each causal update. Lower is better.

| Method | Hidden identity RMSE (mm) | Hidden Chamfer (mm) |
| --- | ---: | ---: |
| Persistence | **0.384** | **0.214** |
| Sealed pairwise-clique RBF update | 6.780 | 6.377 |
| Ungated raw-backbone RBF control | 13.058 | 13.326 |
| Independent CPD control | 13.647 | 13.564 |

Relative to persistence, the sealed primary regresses by 1665.49% in identity
RMSE and 2873.34% in Chamfer. All 12 object means regress on both metrics. The
object-clustered 95% intervals for the primary-minus-persistence difference are
`[3.371, 9.802] mm` and `[3.259, 9.575] mm`, respectively, and the one-sided
object sign-test value is 1.0 for both.

At episode level, the primary has zero joint wins, one bit-exact fallback tie,
and 22 joint regressions. The confirmation gate therefore fails every
performance condition on both co-primary metrics. The aggregate result SHA-256
is
`f8336eedb1eae69dec47b64ea2deaae1c80cf5d44fc66f52b937aa6a5476ee84`.

## What Failed

The action-only window selector did not ensure informative object motion. The
persistence identity error is below 1 mm in 22 of 23 evaluated episodes and has
a median of only 0.153 mm. Persistence is therefore an unusually strong
baseline for this cohort, while any coherent camera correction has ample room
to inject error.

The pairwise-distance clique rejects many identity-inconsistent tracks, but it
does not detect a spatially coherent 3D bias. A biased center set can preserve
pairwise distances and pass the gate. The RBF then spreads that accepted bias to
hidden material points, precisely where the protocol scores it.

The primary remains substantially better than the ungated RBF and CPD controls.
That supports the value of correspondence rejection, but not the hypothesis
that the accepted camera corrections improve a digital twin.

## Post-Open Mechanism Check

After all outcomes were open, the sealed primary was first regenerated
bit-for-bit in every case. Three fixed diagnostics were then applied only to
explain the failure:

| Diagnostic arm | Identity RMSE (mm) | Chamfer (mm) | Identity change vs persistence | Chamfer change vs persistence |
| --- | ---: | ---: | ---: | ---: |
| Sealed primary | 6.780 | 6.377 | +1665.49% | +2873.34% |
| Empirical-Bayes innovation shrinkage | 5.786 | 5.396 | +1406.75% | +2415.96% |
| At least three inlier views | 1.247 | 1.111 | +224.71% | +417.99% |
| Three views plus shrinkage | 0.960 | 0.681 | +149.92% | +217.33% |

The combined diagnostic often falls back exactly and removes most of the
damage, but it improves zero object means over persistence. More views and
innovation shrinkage mitigate the failure without solving it. These are
post-open diagnostics and cannot replace the sealed primary or support a new
claim. Their result SHA-256 is
`1216df16d0855950090196468561ffc96b7dd4df39a9d77563a5b0c2afc66eb1`.

## Common-Mode Ambiguity

The failure has a simple identifiability explanation. Work relative to a fixed
baseline trajectory and write a camera-derived innovation as

```text
y = d + b + e,
```

where `d` is the true state discrepancy, `b` is coherent observation bias, and
`e` is noise. Assume the camera observation model is invariant between worlds
with the same `d + b`. Let an update rule use the complete camera record,
including any confidence, reprojection residual, ensemble variance, or
uncertainty statistic computed from that record.

**Proposition.** If arbitrary coherent bias is allowed, no camera-only update
rule can be uniformly non-worsening relative to the baseline and also strictly
improve some nonzero discrepancy under squared error.

**Proof.** Choose a nonzero vector `u`. World A has `d = u, b = 0`; world B has
`d = 0, b = u`. Couple the noise identically. Both worlds then produce the same
camera evidence and every camera-measurable uncertainty statistic, so the rule
must emit the same correction `a` in both. In world B the baseline already has
zero error. Non-worsening therefore requires `||a||^2 <= 0`, hence `a = 0`.
The same zero correction cannot strictly improve the baseline in world A.
Therefore the two requirements cannot hold simultaneously. QED.

This is a uniform impossibility result, not a claim that cameras are useless.
It says that a safety guarantee requires assumptions or evidence that break the
ambiguity. Two-view triangulation is only determined, and a third view exposes
inconsistency only when its errors are sufficiently independent. An innovation
magnitude or uncertainty estimated from the same biased observations is still
a function of indistinguishable evidence.

## Research Decision

Do not tune another camera-only threshold, confidence head, or overlap stitcher
on this opened cohort. The next paper candidate should be a guarded Bayesian
state update with three explicit ingredients:

1. A physical and action-conditioned prior that predicts both state motion and
   where an observation innovation should be supported.
2. A latent shared-bias model, for example camera, time, and low-rank spatial
   modes, so coherent innovation is not automatically interpreted as state.
3. A source-calibrated upper confidence bound on loss relative to the unchanged
   baseline. Apply an update only when the bound certifies negative regret;
   otherwise return the baseline bit-for-bit.

The study window must also require target-free evidence of actual deformation,
not gripper path alone. Contact support, causal multiview motion, and predicted
physical response should be frozen on open source objects before selecting a
fresh-object cohort.

This yields a coherent paper question: **when can a deformable digital twin
safely believe its cameras under common-mode bias?** A strong paper would pair
the proposition and this prospective failure with a positive, independently
locked evaluation of the bias-aware guarded update. The current negative result
alone is not a state-of-the-art claim or a finished paper.

Bayesian-PhysTwin should own that method and paper story. Prob4D should remain a
versioned observation and calibration feeder, exposing positions,
uncertainties, support, camera provenance, and checksums through a narrow
adapter. Reopen Prob4D as an independent method paper only if a genuinely
independent sensing source, such as sparse depth, LiDAR, or tactile anchors,
produces a calibrated advantage over decoded camera-only baselines.

## Execution Note

Two outcome builders were accidentally overlapped on GPU 0. The
`125-rabbit-ep0008` attempt exhausted GPU memory, and the concurrent
`049-ball-ep0005` attempt was deliberately stopped so recovery could run
serially. Neither failed attempt produced a completed outcome manifest. Both
cases passed from the unchanged sealed inputs, protocol, and commit when rerun
serially. The original statuses and log hashes are retained in the execution
audit; no failed-attempt log is used as scientific evidence.

## Evidence

- `results/sota/deform360_selective_virtual_sensing_v1/prediction_cohort_seal.json`
- `results/sota/deform360_selective_virtual_sensing_v1/prospective_aggregate.json`
- `results/sota/deform360_selective_virtual_sensing_v1/posthoc_mechanism_diagnostic.json`
- `results/sota/deform360_selective_virtual_sensing_v1/quality_failure.json`
- `results/sota/deform360_selective_virtual_sensing_v1/evaluations/`
- `results/sota/deform360_selective_virtual_sensing_v1/outcome_manifests/`
- `results/sota/deform360_selective_virtual_sensing_v1/prediction_seals/`
- `results/sota/deform360_selective_virtual_sensing_v1/execution_audit.json`
