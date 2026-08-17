# DEFORM DLO2 local-residual official v7 result

## Decision

The frozen one-shot gate passed. On all 14 released DEFORM DLO2 evaluation
trajectories, the all-train physical model plus the fixed local residual reached
`7.8606 mm` mean coordinate L1. This is below the published DEFORM DLO2 value
of `9.7 mm` and improves the identically trained physical checkpoint by
`10.13%` (`8.7470 mm` to `7.8606 mm`). The candidate won all `14/14` paired
cases, and its worst candidate-to-baseline ratio was `0.9419`.

The canonical with-replacement compatibility draw reached `8.5037 mm`, also
below `9.7 mm`. The all-14 and canonical-draw comparisons therefore pass the
locked benchmark-specific claim gate. A public paper and repository search on
2026-08-17 found no lower reported value on this exact released DLO2 operator.
This supports a best-published-result claim for this benchmark, not a universal
claim over DLO prediction, other objects, or other benchmarks.

## Results

| Operator | Physical baseline | Local-residual candidate | Change |
|---|---:|---:|---:|
| All 14 unique evaluation trajectories | 8.7470 mm | **7.8606 mm** | -10.13% |
| Canonical released-loader draw | 9.5834 mm | **8.5037 mm** | -11.26% |
| Published DEFORM DLO2 reference | - | 9.7000 mm | candidate -12.33% |

The candidate horizon errors were `6.4793/8.2828/8.8195 mm` for the
early/middle/late thirds. The matching physical-checkpoint errors were
`7.2395/9.2177/9.7837 mm`, so the gain persists through the late horizon rather
than coming from one short segment.

Action-aware persistence was `52.8753 mm`; it is reported as a diagnostic, not
used as the principal comparison. The principal paired baseline is the same
6,400-update physical model used inside the candidate.

## Calibration

The all-train covariance was reused unchanged. At nominal 90% marginal
coordinate coverage, the official evaluation obtained `94.00%` coverage and
coordinate NEES `0.894`. Late-horizon coverage was `92.36%` with NEES `1.022`.
No target calibration or variance rescaling was performed.

## Custody

- Source revision: `2cdbff202b2b000a96c6eddf9e750999ee9f6e75`
- Source archive SHA-256: `f0e2ac2d1166f1e95e2ca7ba70ee109a63cc7e110687c7a863cefb0cd8322d69`
- All-train protocol SHA-256: `39566ca4542f1a0afff84458123580be92ca3c95af5bda396351ed702e7d4739`
- Official protocol SHA-256: `7761b3a8084fad60278eb82d5aa6bd96d3a3b7a6ebe2ad5d7c74d13ce8a0abcb`
- All-train result SHA-256: `3bc1fa4cb95398b2b7df588896bba619c19424615fa5c087c7a6a65eb6f725d0`
- Official result SHA-256: `c5009e47072adf4e72547cd11848d47bcb1259c3f591c8210cf53dc329ba6856`
- Prediction SHA-256: `431c778022bfb7b602512e5e6c2132a3f42e5959c959368e5203059bd2ce223b`
- Runtime: Python 3.10.12, Torch 2.0.1+cu118, CUDA 11.8
- Target selection: none
- Target calibration: none
- Retry or case replacement: none
- Prob4D used: false

The official evaluator read all 14 expected cases once, then exited normally.
The result and recomputed comparison and uncertainty summaries match exactly.

## Claim boundary

The published DEFORM loader samples with replacement after a preceding training
draw. Its filesystem glob order was not specified, so the canonical sorted-name
draw is a source-code compatibility view rather than a reconstruction of the
paper machine's path order. We therefore report both the stable all-14 mean and
the frozen canonical draw; both are below the paper's `9.7 mm` DLO2 value.

The scored rollout contains 498 future frames after the two causal input states.
The training budget is the preregistered 6,400-update, horizon-50 route, not a
compute-matched reproduction of the released DEFORM training script. Different
DLO datasets, manipulation benchmarks, and sensing tasks are not comparable to
this number.

Compact evidence is in
`results/sota/deform_dlo2_local_residual_official_v7/summary.json`. The primary
reference audit remains
`results/sota/deform_dlo2_official_eval_v2/reference_operator_audit.json`.
